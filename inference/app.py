from __future__ import annotations

import io
import os
import re
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
THRESHOLD = float(os.environ.get("APNEA_THRESHOLD", "0.23"))
SMOOTH_WINDOW = int(os.environ.get("SMOOTH_WINDOW", "5"))
# APNEA_THRESHOLD belongs to the model file: it was chosen on the validation records for the
# int8 model, after 5-minute smoothing (0.23 for the model trained in src/model.ipynb). A
# retrained model needs its own value; reusing an old one silently shifts every decision.

RECORD_NAME = re.compile(r"^[A-Za-z0-9_-]{1,32}$")
# Record names become file paths, so anything like "../" is refused before it reaches the disk.

model = ApneaModel(MODEL_PATH)
app = FastAPI(title="Apnea inference", version="0.1")


@app.get("/", include_in_schema=False)
@app.get("/patient/{patient}", include_in_schema=False)
def page(patient: str | None = None) -> FileResponse:
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


@app.get("/api/patients")
def patients() -> list[dict]:
    return list_patients(PATIENTS_DIR)


@app.get("/api/patients/{patient}")
def patient_nights(patient: str) -> dict:
    return {"patient": patient, "sample_rate_hz": RECORDING_FS,
            "nights": list_nights(_patient_folder(patient), RECORDING_FS)}


@app.post("/api/patients/{patient}/nights/{night}/analysis")
def analyse_night(patient: str, night: str) -> dict:
    path = night_file(_patient_folder(patient), night)
    if path is None:
        raise HTTPException(404, f"no night named {night!r} for patient {patient!r}")
    try:
        ecg = read_night(path)
        return {"patient": patient, "night": night, "sample_rate_hz": RECORDING_FS,
                **analyse(ecg, RECORDING_FS, model, THRESHOLD, SMOOTH_WINDOW)}
    except ValueError as exc:
        raise HTTPException(422, f"could not analyse {night!r}: {exc}") from exc
    # POST, although nothing is stored: the request runs the model, and a GET could be fired by a
    # browser prefetching links. A full night takes about a second, so it runs while the doctor
    # waits and nothing is cached.


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
