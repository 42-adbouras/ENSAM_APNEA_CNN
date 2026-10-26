from __future__ import annotations

import io
import os
import threading
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import Session, select

import store
from db import Patient, Recording, get_session, init_db
from patients import read_samples
from pipeline import FS, ApneaModel, analyse

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/models/model_int8.tflite"))
RECORDING_FS = float(os.environ.get("RECORDING_FS", FS))
# The ESP32's sampling rate. A night file is bare samples with no header to say it, so the rate
# is set once for the device here; a night's duration is its sample count divided by it.
STATIC_DIR = Path(os.environ.get("STATIC_DIR", Path(__file__).parent / "static"))
# The built React app (web/dist): index.html and its assets/ folder.
MODELS_DIR = MODEL_PATH.parent
# The folder the night page's model dropdown lists; MODEL_PATH is the model it starts on.


def _parse_thresholds(text: str) -> dict[str, float]:
    pairs = (pair.split("=", 1) for pair in text.split(",") if pair.strip())
    return {name.strip(): float(value) for name, value in pairs}
    # "model_int8.tflite=0.23,model_float32.tflite=0.20" -> {"model_int8.tflite": 0.23, ...}


THRESHOLDS = _parse_thresholds(os.environ.get("MODEL_THRESHOLDS", "model_int8.tflite=0.23"))
# One decision threshold per model file. A model missing from it is listed but cannot be run.
if MODEL_PATH.name not in THRESHOLDS:
    raise RuntimeError(f"MODEL_THRESHOLDS has no threshold for {MODEL_PATH.name}")
THRESHOLD = THRESHOLDS[MODEL_PATH.name]
SMOOTH_WINDOW = int(os.environ.get("SMOOTH_WINDOW", "5"))

model = ApneaModel(MODEL_PATH)
init_db()
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
# No /docs, /redoc or /openapi.json routes.
app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets", check_dir=False), name="assets")
# Vite's hashed JavaScript and CSS files, referenced by index.html as /assets/...

_loaded: dict[str, tuple[int, ApneaModel]] = {MODEL_PATH.name: (MODEL_PATH.stat().st_mtime_ns, model)}
_loaded_lock = threading.Lock()
_hashes: dict[str, tuple[int, str]] = {}


def _model_files() -> dict[str, Path]:
    return {p.name: p for p in sorted(MODELS_DIR.glob("*.tflite")) if not p.name.startswith(".")}


def _model_sha256(path: Path) -> str:
    mtime = path.stat().st_mtime_ns
    cached = _hashes.get(path.name)
    if cached is None or cached[0] != mtime:
        cached = _hashes[path.name] = (mtime, store.sha256_of(path.read_bytes()))
    return cached[1]
    # Hashed once per version of the file: a file replaced on disk has a new mtime.


def _model(name: str) -> tuple[ApneaModel, float, Path]:
    """A model file of the models folder, loaded, its threshold and its path."""
    path = _model_files().get(name)
    if path is None:
        raise HTTPException(404, f"no model named {name!r}")
    if name not in THRESHOLDS:
        raise HTTPException(409, f"no threshold for {name!r}: add it to MODEL_THRESHOLDS")
    mtime = path.stat().st_mtime_ns
    with _loaded_lock:
        cached = _loaded.get(name)
        if cached is None or cached[0] != mtime:
            try:
                _loaded[name] = (mtime, ApneaModel(path))
            except (ValueError, RuntimeError) as exc:
                raise HTTPException(409, f"model {name!r} cannot be loaded: {exc}") from exc
        return _loaded[name][1], THRESHOLDS[name], path
    # Each file is loaded once and kept; a file replaced on disk has a new mtime and is loaded
    # again. The name is looked up among the listed files, never joined onto the folder.


MODEL_QUERY = Query(MODEL_PATH.name)
SESSION = Depends(get_session)


@app.get("/")
@app.get("/patient/{patient}")
@app.get("/patient/{patient}/night/{night}")
def page(patient: str | None = None, night: str | None = None) -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
    # One self-contained page, no CDN: it fetches the /api routes below and works offline, on a
    # clinic network with no internet access. /patient/{id} returns the same page, which reads
    # the patient from its own address; without this route, reloading or bookmarking a patient
    # would ask the server for a path it does not know.


def _patient(session: Session, patient_id: int) -> Patient:
    row = store.patient(session, patient_id)
    if row is None:
        raise HTTPException(404, f"no patient with id {patient_id}")
    return row


def _night(session: Session, patient: Patient, file: str) -> Recording:
    row = next((r for r in store.nights(session, patient) if r.file_name == file), None)
    if row is None:
        raise HTTPException(404, f"no night named {file!r} for patient {patient.id}")
    return row


def _latest(session: Session, nights: list[Recording], model_name: str) -> dict:
    path = _model_files().get(model_name)
    if path is None:
        return {}
    return store.latest_analyses(session, nights, _model_sha256(path), THRESHOLDS.get(model_name),
                                 SMOOTH_WINDOW)
    # A night counts as analysed with a model only by this exact file (its checksum), with the
    # current threshold and smoothing; after a change it shows as not analysed.


@app.get("/api/models")
def models() -> dict:
    return {"default": MODEL_PATH.name,
            "models": [{"name": name, "size_bytes": path.stat().st_size,
                        "threshold": THRESHOLDS.get(name)}
                       for name, path in _model_files().items()]}


@app.get("/api/patients")
def patients(session: Session = SESSION) -> list[dict]:
    listed = []
    for patient in store.patients(session):
        count = len(store.nights(session, patient))
        listed.append({"id": patient.id, "name": patient.code, "nights": count})
    return listed


class NewPatient(BaseModel):
    name: str


@app.post("/api/patients", status_code=201)
def new_patient(body: NewPatient, session: Session = SESSION) -> dict:
    try:
        patient = store.new_patient(session, body.name)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"id": patient.id, "name": patient.code}


@app.get("/api/patients/{patient}")
def patient_nights(patient: int, model: str = MODEL_QUERY, session: Session = SESSION) -> dict:
    row = _patient(session, patient)
    nights = store.nights(session, row)
    latest = _latest(session, nights, model)
    listed = []
    for number, night in enumerate(nights, start=1):
        analysis = latest.get(night.id)
        listed.append({"file": night.file_name, "number": number, "samples": night.samples,
                       "duration_s": night.samples / night.sample_rate_hz,
                       "size_bytes": night.size_bytes,
                       "analysis": None if analysis is None
                       else {"minutes": analysis.minutes, **store.summary(analysis)}})
    return {"id": row.id, "name": row.code, "model": model, "sample_rate_hz": RECORDING_FS,
            "nights": listed}
    # Nights oldest first, numbered from 1. "analysis" is the summary of the newest analysis
    # with `model`, or None when the night has not been analysed with it.


@app.post("/api/patients/{patient}/nights", status_code=201)
def upload_night(patient: int, file: UploadFile = File(...),
                 modified_ms: int | None = Form(None, ge=0), session: Session = SESSION) -> dict:
    row = _patient(session, patient)
    name = Path(file.filename or "").name
    modified_ns = None if modified_ms is None else modified_ms * 1_000_000
    try:
        store.add_night(session, row, name, file.file.read(), modified_ns, RECORDING_FS)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except FileExistsError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"id": row.id, "file": name}
    # One night per request, stored in the database. modified_ms is the original file's
    # modification time, in milliseconds, as the browser reports it; without it the night is
    # dated by the upload.


@app.get("/api/patients/{patient}/nights/{night}/analysis")
def kept_analysis(patient: int, night: str, model: str = MODEL_QUERY,
                  session: Session = SESSION) -> dict:
    row = _patient(session, patient)
    recording = _night(session, row, night)
    analysis = _latest(session, [recording], model).get(recording.id)
    if analysis is None:
        raise HTTPException(404, f"{night!r} has not been analysed with {model!r}")
    return store.result(analysis, row.code, recording, model)
    # Returns the newest stored analysis without running the model.


@app.post("/api/patients/{patient}/nights/{night}/analysis")
def analyse_night(patient: int, night: str, model: str = MODEL_QUERY,
                  session: Session = SESSION) -> dict:
    row = _patient(session, patient)
    recording = _night(session, row, night)
    chosen, threshold, model_path = _model(model)
    data = store.content(session, recording)
    if store.sha256_of(data) != recording.sha256:
        raise HTTPException(409, f"the stored content of {night!r} does not match its checksum")
    try:
        ecg = read_samples(data)
        result = {"patient": row.code, "night": recording.file_name, "model": model,
                  "sample_rate_hz": recording.sample_rate_hz,
                  **analyse(ecg, recording.sample_rate_hz, chosen, threshold, SMOOTH_WINDOW)}
    except ValueError as exc:
        raise HTTPException(422, f"could not analyse {night!r}: {exc}") from exc
    model_row = store.model_file(session, model_path, _model_sha256(model_path), chosen.quantized)
    store.save_analysis(session, recording, model_row, result)
    return result
    # The content is checked against its checksum before every analysis. Each run is saved as a
    # new analysis; GET on the same path then returns the newest one.


@app.get("/health")
def health(session: Session = SESSION) -> dict:
    session.exec(select(Patient.id).limit(1)).first()
    return {"status": "ok", "database": "ok", "model": MODEL_PATH.name, "threshold": THRESHOLD,
            "smooth_window": SMOOTH_WINDOW, "sample_rate_hz": FS,
            "input_quantization": {"scale": model.in_scale, "zero_point": model.in_zero}}
    # The query fails when the database is unreachable, which makes Docker's health check fail.


@app.post("/analyze")
async def analyse_upload(file: UploadFile = File(...),
                         fs: float = Query(FS, gt=0)) -> dict:
    raw = await file.read()
    try:
        if (file.filename or "").endswith(".npy"):
            ecg = np.load(io.BytesIO(raw), allow_pickle=False)
        else:
            ecg = np.loadtxt(io.BytesIO(raw), delimiter=",", ndmin=1)
    except Exception as exc:
        raise HTTPException(400, f"could not read {file.filename!r}: {exc}") from exc
    # allow_pickle=False: a pickled .npy can execute code when loaded.

    if ecg.ndim != 1:
        raise HTTPException(400, f"expected one ECG channel, got shape {ecg.shape}")
    try:
        return {"file": file.filename, "sample_rate_hz": fs,
                **analyse(ecg, fs, model, THRESHOLD, SMOOTH_WINDOW)}
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
