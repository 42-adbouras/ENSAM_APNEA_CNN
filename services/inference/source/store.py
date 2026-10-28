"""
Patients, nights and analyses in the database.

Every patient's data lives here: a patient is a row, a night is a row of `recordings` with its
file's content in `recording_files`, and every analysis is kept. Nothing about patients is read
from or written to the file system.
"""
import hashlib
import threading
import uuid
from datetime import datetime, timezone

from sqlmodel import Session, col, select

from db import (Analysis, AnalysisMinute, Event, ModelFile, Patient, Recording, RecordingFile,
                now)
from patients import check_night, check_night_name, clean_patient_name, natural_key

SERVICE_VERSION = "0.3"
# Saved with every analysis, so a result can be traced to the code that produced it.

lock = threading.Lock()
# Held while a night is checked and added, so two requests never add the same one.


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _log(session: Session, action: str, **links) -> None:
    session.add(Event(action=action, **links))


def patients(session: Session) -> list[Patient]:
    """Every patient, in alphabetical order of name, then of creation."""
    return sorted(session.exec(select(Patient)), key=lambda p: (p.code.lower(), p.created_at))


def patient(session: Session, patient_id: uuid.UUID) -> Patient | None:
    return session.get(Patient, patient_id)


def new_patient(session: Session, name: str) -> Patient:
    """Adds a patient. ValueError for a name not allowed."""
    row = Patient(code=clean_patient_name(name))
    session.add(row)
    session.flush()
    _log(session, "patient_created", patient_id=row.id, detail="interface")
    session.commit()
    return row
    # Several patients may have the same name: a patient is identified by its id.


def nights(session: Session, patient: Patient) -> list[Recording]:
    """A patient's nights, oldest first. A night counts once its content is stored."""
    rows = session.exec(select(Recording).join(RecordingFile)
                        .where(Recording.patient_id == patient.id)).all()
    return sorted(rows, key=lambda r: (r.recorded_at, natural_key(r.file_name)))


def duplicate_of(session: Session, sha256: str) -> Recording | None:
    """A night, of any patient, whose stored content is exactly this."""
    return session.exec(select(Recording).join(RecordingFile)
                        .where(Recording.sha256 == sha256)).first()


def add_night(session: Session, patient: Patient, name: str, data: bytes, modified_ns: int | None,
              fs: float) -> Recording:
    """Stores a night's details and content.

    ValueError when the name or the content is not a night's. FileExistsError when the patient
    already has a night of that name, or when the same content is already stored.
    """
    check_night_name(name)
    samples = check_night(data, min_samples=int(60 * fs))
    sha256 = sha256_of(data)
    modified_ns = modified_ns if modified_ns is not None else int(now().timestamp() * 1e9)
    with lock:
        if any(r.file_name.lower() == name.lower() for r in nights(session, patient)):
            raise FileExistsError("this patient already has a night with this name")
        twin = duplicate_of(session, sha256)
        if twin is not None:
            raise FileExistsError(f"this night is already stored, as {twin.file_name} "
                                  f"of {twin.patient.code}")
        row = Recording(patient_id=patient.id, file_name=name, sha256=sha256,
                        size_bytes=len(data), mtime_ns=modified_ns, samples=samples,
                        sample_rate_hz=fs,
                        recorded_at=datetime.fromtimestamp(modified_ns / 1e9, timezone.utc))
        session.add(row)
        session.flush()
        session.add(RecordingFile(recording_id=row.id, content=data))
        _log(session, "night_added", patient_id=patient.id, recording_id=row.id, detail="upload")
        session.commit()
    return row
    # The details, the content and the event are committed together: all or nothing. modified_ns
    # is the original file's modification time; it sets the night's place in the order.


def content(session: Session, night: Recording) -> bytes:
    return session.get(RecordingFile, night.id).content


def model_file(session: Session, path, sha256: str, quantized: bool) -> ModelFile:
    """The row of this version of a model file, created the first time it is used."""
    row = session.exec(select(ModelFile).where(ModelFile.sha256 == sha256)).first()
    if row is None:
        row = ModelFile(file_name=path.name, sha256=sha256, size_bytes=path.stat().st_size,
                        quantized=quantized)
        session.add(row)
        session.flush()
    return row


def latest_analyses(session: Session, nights_: list[Recording], model_sha256: str,
                    threshold: float | None, smooth_window: int) -> dict[int, Analysis]:
    """recording id -> the newest analysis of its content with this model version, threshold
    and smoothing."""
    if threshold is None or not nights_:
        return {}
    rows = session.exec(
        select(Analysis, Recording).join(Recording).join(ModelFile)
        .where(col(Recording.sha256).in_({r.sha256 for r in nights_}),
               ModelFile.sha256 == model_sha256,
               Analysis.threshold == threshold,
               Analysis.smooth_window == smooth_window)
        .order_by(col(Analysis.created_at))).all()
    newest = {(rec.sha256, rec.sample_rate_hz): analysis for analysis, rec in rows}
    return {r.id: newest[key] for r in nights_ if (key := (r.sha256, r.sample_rate_hz)) in newest}
    # Matched by the night's checksum, not its row, so an analysis follows the content. Rows come
    # oldest first, so the dict keeps each night's newest analysis.


def save_analysis(session: Session, night: Recording, model: ModelFile, result: dict) -> Analysis:
    analysis = Analysis(recording_id=night.id, model_file_id=model.id,
                        threshold=result["threshold"], smooth_window=result["smooth_window"],
                        minutes=result["minutes"],
                        apnea_minutes=result["summary"]["apnea_minutes"],
                        minutes_per_hour=result["summary"]["apnea_minutes_per_hour"],
                        service_version=SERVICE_VERSION)
    session.add(analysis)
    session.flush()
    session.add_all(AnalysisMinute(analysis_id=analysis.id, minute=i, prob=p, prob_smoothed=s, apnea=a)
                    for i, (p, s, a) in enumerate(zip(result["prob"], result["prob_smoothed"],
                                                      result["apnea"])))
    _log(session, "night_analysed", patient_id=night.patient_id, recording_id=night.id,
         analysis_id=analysis.id)
    session.commit()
    return analysis
    # The analysis, its minutes and the event are committed together: all or nothing.


def summary(analysis: Analysis) -> dict:
    return {"apnea_minutes": analysis.apnea_minutes,
            "apnea_fraction": analysis.apnea_minutes / analysis.minutes,
            "apnea_minutes_per_hour": analysis.minutes_per_hour}


def result(analysis: Analysis, patient_code: str, night: Recording, model_name: str) -> dict:
    """A stored analysis in the shape POST .../analysis returns."""
    minutes = analysis.per_minute
    return {"patient": patient_code, "night": night.file_name, "model": model_name,
            "sample_rate_hz": night.sample_rate_hz, "minutes": analysis.minutes,
            "threshold": analysis.threshold, "smooth_window": analysis.smooth_window,
            "summary": summary(analysis),
            "prob": [m.prob for m in minutes],
            "prob_smoothed": [m.prob_smoothed for m in minutes],
            "apnea": [m.apnea for m in minutes]}
