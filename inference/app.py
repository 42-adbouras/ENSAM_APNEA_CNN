from __future__ import annotations

import io
import json
import logging
import os
import re
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from patients import list_nights, list_patients, night_file, patient_dirs, read_night
from pipeline import FS, ApneaModel, analyse, has_ecg, load_physionet_record

MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/models/model_int8.tflite"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/apnea-ecg"))
PATIENTS_DIR = Path(os.environ.get("PATIENTS_DIR", "/patients"))
RECORDING_FS = float(os.environ.get("RECORDING_FS", FS))
# The ESP32's sampling rate. A night file is bare samples with no header to say it, so the rate
# is set once for the device here; a night's duration is its sample count divided by it.
STATIC_DIR = Path(__file__).parent / "static"
MODELS_DIR = MODEL_PATH.parent
# The folder the night page's model dropdown lists; MODEL_PATH is the model it starts on.
ANALYSES_DIR = Path(os.environ.get("ANALYSES_DIR", "/analyses"))
# Where analysis results are stored: <patient>/<night file>/<model file>.json


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

RECORD_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
# Record names become file paths, so anything like "../" is refused before it reaches the disk.

model = ApneaModel(MODEL_PATH)
app = FastAPI(title="Apnea inference", version="0.1")

_loaded: dict[str, tuple[int, ApneaModel]] = {MODEL_PATH.name: (MODEL_PATH.stat().st_mtime_ns, model)}
_loaded_lock = threading.Lock()


def _model_files() -> dict[str, Path]:
    return {p.name: p for p in sorted(MODELS_DIR.glob("*.tflite")) if not p.name.startswith(".")}


def _model(name: str) -> tuple[ApneaModel, float]:
    """A model file of the models folder, loaded, and its threshold."""
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
        return _loaded[name][1], THRESHOLDS[name]
    # Each file is loaded once and kept; a file replaced on disk has a new mtime and is loaded
    # again. The name is looked up among the listed files, never joined onto the folder.


MODEL_QUERY = Query(MODEL_PATH.name, description="a model file in the models folder")

log = logging.getLogger("uvicorn.error")


def _stored_path(patient: str, night: Path, model_name: str) -> Path:
    return ANALYSES_DIR / patient / night.name / f"{model_name}.json"
    # The three names come from folders and files actually listed, so none can be "..".


def _source(night: Path, model_name: str) -> dict | None:
    model_path = _model_files().get(model_name)
    if model_path is None:
        return None
    night_stat, model_stat = night.stat(), model_path.stat()
    return {"night_size": night_stat.st_size, "night_mtime_ns": night_stat.st_mtime_ns,
            "model_size": model_stat.st_size, "model_mtime_ns": model_stat.st_mtime_ns,
            "threshold": THRESHOLDS.get(model_name), "smooth_window": SMOOTH_WINDOW}
    # What a result was computed from. A stored result is returned only while this is unchanged,
    # so a replaced night or model file, or a new threshold, makes it count as not analysed.


def _kept_result(patient: str, night: Path, model_name: str) -> dict | None:
    source = _source(night, model_name)
    stored = _stored_path(patient, night, model_name)
    if source is None or not stored.is_file():
        return None
    try:
        kept = json.loads(stored.read_text())
    except (OSError, ValueError):
        return None
    return kept.get("result") if kept.get("source") == source else None


def _keep_result(patient: str, night: Path, model_name: str, result: dict) -> None:
    stored = _stored_path(patient, night, model_name)
    try:
        stored.parent.mkdir(parents=True, exist_ok=True)
        partial = stored.with_suffix(".partial")
        partial.write_text(json.dumps({"source": _source(night, model_name), "result": result}))
        os.replace(partial, stored)
    except OSError as exc:
        log.warning("analysis of %s/%s not stored: %s", patient, night.name, exc)
    # Written to a ".partial" file, then renamed over the old one in a single step, so a reader
    # never sees half a file. A failed write is logged and the result is still returned.


@app.get("/", include_in_schema=False)
@app.get("/patient/{patient}", include_in_schema=False)
@app.get("/patient/{patient}/night/{night}", include_in_schema=False)
def page(patient: str | None = None, night: str | None = None) -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
    # One self-contained page, no CDN: it fetches the /api routes below and works offline, on a
    # clinic network with no internet access. /patient/{id} returns the same page, which reads
    # the patient from its own address; without this route, reloading or bookmarking a patient
    # would ask the server for a path it does not know.


def _patient_folder(patient: str) -> Path:
    folder = patient_dirs(PATIENTS_DIR).get(patient)
    if folder is None:
        raise HTTPException(404, f"no patient named {patient!r}")
    return folder
    # The name is matched against the folders actually listed, never joined onto the path, so a
    # request for ".." or "../results" finds nothing.


@app.get("/api/models")
def models() -> dict:
    return {"default": MODEL_PATH.name,
            "models": [{"name": name, "size_bytes": path.stat().st_size,
                        "threshold": THRESHOLDS.get(name)}
                       for name, path in _model_files().items()]}


@app.get("/api/patients")
def patients() -> list[dict]:
    return list_patients(PATIENTS_DIR)


@app.get("/api/patients/{patient}")
def patient_nights(patient: str, model: str = MODEL_QUERY) -> dict:
    folder = _patient_folder(patient)
    nights = list_nights(folder, RECORDING_FS)
    for night in nights:
        result = _kept_result(patient, folder / night["file"], model)
        night["analysis"] = None if result is None else {"minutes": result["minutes"], **result["summary"]}
    return {"patient": patient, "model": model, "sample_rate_hz": RECORDING_FS, "nights": nights}
    # "analysis" is the summary of the kept result for `model`, or None when the night has not
    # been analysed with it.


def _night_path(patient: str, night: str) -> Path:
    path = night_file(_patient_folder(patient), night)
    if path is None:
        raise HTTPException(404, f"no night named {night!r} for patient {patient!r}")
    return path


@app.get("/api/patients/{patient}/nights/{night}/analysis")
def kept_analysis(patient: str, night: str, model: str = MODEL_QUERY) -> dict:
    result = _kept_result(patient, _night_path(patient, night), model)
    if result is None:
        raise HTTPException(404, f"{night!r} has not been analysed with {model!r}")
    return result
    # Returns the kept result without running the model.


@app.post("/api/patients/{patient}/nights/{night}/analysis")
def analyse_night(patient: str, night: str, model: str = MODEL_QUERY) -> dict:
    path = _night_path(patient, night)
    chosen, threshold = _model(model)
    try:
        ecg = read_night(path)
        result = {"patient": patient, "night": night, "model": model, "sample_rate_hz": RECORDING_FS,
                  **analyse(ecg, RECORDING_FS, chosen, threshold, SMOOTH_WINDOW)}
    except ValueError as exc:
        raise HTTPException(422, f"could not analyse {night!r}: {exc}") from exc
    _keep_result(patient, path, model, result)
    return result
    # Runs the model and keeps the result, which GET on the same path then returns.


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH.name, "threshold": THRESHOLD,
            "smooth_window": SMOOTH_WINDOW, "sample_rate_hz": FS,
            "input_quantization": {"scale": model.in_scale, "zero_point": model.in_zero}}


@app.get("/records")
def records() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.stem for p in DATA_DIR.glob("*.hea")
                  if (DATA_DIR / f"{p.stem}.dat").exists() and has_ecg(DATA_DIR, p.stem))


@app.get("/records/{name}")
def analyse_record(name: str) -> dict:
    if not RECORD_NAME.match(name) or not (DATA_DIR / f"{name}.hea").exists():
        raise HTTPException(404, f"no record named {name!r}")
    if not has_ecg(DATA_DIR, name):
        raise HTTPException(422, f"record {name!r} has no ECG channel")

    ecg, fs, labels = load_physionet_record(DATA_DIR, name)
    result = {"record": name, **analyse(ecg, fs, model, THRESHOLD, SMOOTH_WINDOW)}

    if labels is not None:
        labels = labels[:result["minutes"]]
        truth = np.array([s == "A" for s in labels])
        pred = np.array(result["apnea"][:len(labels)])
        result["expert_labels"] = labels
        result["agreement_with_expert"] = float(np.mean(pred == truth))
        # Apnea-ECG's expert annotations: shown next to the prediction for review, never used
        # to produce it.
    return result


@app.post("/analyze")
async def analyse_upload(file: UploadFile = File(...),
                         fs: float = Query(FS, gt=0, description="sampling rate of the upload, Hz")) -> dict:
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
