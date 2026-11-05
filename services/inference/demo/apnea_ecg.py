"""
Demo patients from the Apnea-ECG Database (PhysioNet): three patients, the database's 35 test
recordings, x01 to x35.

Run by `make demo`, inside the running inference container, where data/apnea-ecg is mounted at
/apnea-ecg. For each night: download the recording (.dat) and its header (.hea) into /apnea-ecg
unless they are already there, check both against PhysioNet's SHA-256 list, convert the ECG to the
upload format (a header, then one mV value per line, at RECORDING_FS Hz) and add it through the
service's API, as Upload nights does. A night the patient already has is skipped, so a second run
adds nothing.
"""
from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

MIRRORS = ["https://physionet-open.s3.amazonaws.com/apnea-ecg/1.0.0/"] * 3 + [
    "https://physionet.org/files/apnea-ecg/1.0.0/"]
# Where each file is requested, in turn, until one answer is complete: PhysioNet's copy on Amazon
# S3 three times, then physionet.org. Both hold the same files and the same SHA256SUMS.txt.
CACHE = Path("/apnea-ecg")
API = os.environ.get("DEMO_API", "http://localhost:8000")
RECORDING_FS = float(os.environ.get("RECORDING_FS", "100"))

PATIENTS = {
    "Apnea-ECG control": ["x04", "x06", "x17", "x18", "x22", "x24", "x29", "x33", "x34", "x35"],
    "Apnea-ECG borderline": ["x03", "x10", "x11", "x12", "x16"],
    "Apnea-ECG apnea": ["x01", "x02", "x05", "x07", "x08", "x09", "x13", "x14", "x15", "x19",
                        "x20", "x21", "x23", "x25", "x26", "x27", "x28", "x30", "x31", "x32"],
}
# Patient name -> its nights. The records are grouped by Apnea-ECG's own classes, counted from the
# expert's minute labels (x*.apn): control, fewer than 5 apnea minutes; borderline, 5 to 99;
# apnea, 100 or more.

FIRST_NIGHT = datetime(2026, 10, 1, 7, 0, tzinfo=timezone.utc)
# A patient's night k is dated FIRST_NIGHT + k days; the service orders nights by this date.


# ── download ─────────────────────────────────────────────────────────────────

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(name: str, digest: str | None = None) -> bytes:
    problem = None
    for attempt, mirror in enumerate(MIRRORS):
        if attempt:
            time.sleep(5 * attempt)
        try:
            with urllib.request.urlopen(mirror + name, timeout=60) as response:
                data = response.read()
        except (OSError, http.client.HTTPException) as error:
            problem = error
            continue
        if digest is None or sha256(data) == digest:
            return data
        problem = "the download does not match PhysioNet's checksum"
    raise OSError(f"{name}: {problem}")
    # Each failed attempt (no connection, a cut transfer, a wrong checksum) waits 5 s longer than
    # the one before, then asks the next address of MIRRORS.


def checksums() -> dict[str, str]:
    lines = fetch("SHA256SUMS.txt").decode().splitlines()
    return {name: digest for digest, name in (line.split() for line in lines if line.strip())}
    # Each line is "<sha256> <file name>".


def cached(name: str, sums: dict[str, str]) -> Path:
    path = CACHE / name
    if path.exists() and sha256(path.read_bytes()) == sums[name]:
        return path
    data = fetch(name, sums[name])
    part = path.with_name(name + ".part")
    part.write_bytes(data)
    part.replace(path)
    if len(data) > 1e6:
        print(f"  {name} downloaded, {len(data) / 1e6:.0f} MB", flush=True)
    return path
    # A file is written under another name and renamed once complete, so an interrupted download
    # never leaves a partial file behind; a cached file that fails the checksum is downloaded again.


def recording(record: str, sums: dict[str, str]) -> Path | None:
    try:
        cached(f"{record}.hea", sums)
        return cached(f"{record}.dat", sums).with_suffix("")
    except OSError as error:
        print(f"  {record}: not downloaded, {error}", flush=True)
        return None
    # The record's path without extension, where its .hea and .dat sit next to each other; None
    # when either file could not be downloaded.


# ── WFDB -> CSV ──────────────────────────────────────────────────────────────

def ecg_mv(record: Path) -> tuple[np.ndarray, float]:
    """The ECG of an Apnea-ECG record, in mV, and its sampling rate."""
    header, signal = (line.split() for line in record.with_suffix(".hea").read_text().splitlines()[:2])
    if header[1] != "1" or signal[1] != "16":
        sys.exit(f"{record.name}: expected one signal in WFDB format 16")
    fs = float(header[2].split("/")[0].split("(")[0])
    gain_field, adc_zero = signal[2], int(signal[4])
    gain = float(gain_field.split("/")[0].split("(")[0])
    baseline = int(gain_field.split("(")[1].split(")")[0]) if "(" in gain_field else adc_zero

    digital = np.fromfile(record.with_suffix(".dat"), dtype="<i2")
    return (digital - baseline) / gain, fs
    # Header lines: "<record> <signals> <fs> <samples>", then the ECG's line: "<file> <format>
    # <gain>(<baseline>)/<units> <bits> <adc zero> <initial value> <checksum> <block size>
    # <description>". The baseline defaults to the ADC zero. Format 16: one little-endian 16-bit
    # integer per sample.


def night_csv(record: Path) -> bytes:
    mv, fs = ecg_mv(record)
    ratio = Fraction(RECORDING_FS / fs).limit_denominator(1000)
    mv = resample_poly(mv, ratio.numerator, ratio.denominator)
    out = io.BytesIO()
    np.savetxt(out, mv, fmt="%.3f", header="ecg_mv", comments="")
    return out.getvalue()
    # 100 Hz -> RECORDING_FS Hz, the rate the service assumes for every uploaded night.


# ── the service's API ────────────────────────────────────────────────────────

def call(method: str, path: str, body: bytes | None = None,
         content_type: str | None = None) -> tuple[int, object]:
    request = urllib.request.Request(API + path, data=body, method=method)
    if content_type:
        request.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")
    # An error status is returned like a success, with FastAPI's {"detail": ...} as the answer.


def wait_for_service() -> None:
    for _ in range(60):
        try:
            with urllib.request.urlopen(API + "/health", timeout=5):
                return
        except OSError:
            time.sleep(2)
    sys.exit(f"the inference service does not answer at {API}")
    # `make up` returns before the service is ready; this waits up to two minutes.


def new_patient(name: str) -> str:
    status, created = call("POST", "/api/patients", json.dumps({"name": name}).encode(),
                           "application/json")
    if status != 201:
        sys.exit(f"could not create {name!r}: {status} {created}")
    return created["id"]


def upload(patient: str, file_name: str, content: bytes, modified: datetime) -> tuple[int, object]:
    boundary = uuid.uuid4().hex
    body = b"".join([
        f'--{boundary}\r\nContent-Disposition: form-data; name="modified_ms"\r\n\r\n'
        f"{int(modified.timestamp() * 1000)}\r\n".encode(),
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
        f"Content-Type: text/csv\r\n\r\n".encode(),
        content,
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    return call("POST", f"/api/patients/{patient}/nights", body,
                f"multipart/form-data; boundary={boundary}")
    # The same two form fields the interface sends: the file and its date in milliseconds.


def main() -> None:
    wait_for_service()
    ids, missing = {}, {}
    for patient in call("GET", "/api/patients")[1]:
        ids.setdefault(patient["name"], patient["id"])
    for name, records in PATIENTS.items():
        stored = set()
        if name in ids:
            _, details = call("GET", f"/api/patients/{ids[name]}")
            stored = {night["file"] for night in details["nights"]}
        missing[name] = [record for record in records if f"{record}.csv" not in stored]
    # The first patient of each name is reused, so a second run adds to it instead of creating
    # another.

    needed = [record for records in missing.values() for record in records]
    paths = {}
    if needed:
        try:
            sums = checksums()
        except OSError as error:
            sys.exit(f"could not download the checksum list: {error}")
        print(f"Checking {len(needed)} recordings in data/apnea-ecg, downloading the missing ones",
              flush=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            paths = dict(zip(needed, pool.map(lambda record: recording(record, sums), needed)))
    # Six downloads run side by side.

    failed = []
    for name, records in PATIENTS.items():
        print(name, flush=True)
        for k, record in enumerate(records):
            file_name = f"{record}.csv"
            if record not in missing[name]:
                print(f"  {file_name} already there", flush=True)
                continue
            if paths.get(record) is None:
                print(f"  {file_name} not added, its download failed", flush=True)
                failed.append(record)
                continue
            if name not in ids:
                ids[name] = new_patient(name)
            content = night_csv(paths[record])
            status, answer = upload(ids[name], file_name, content, FIRST_NIGHT + timedelta(days=k))
            if status == 201:
                print(f"  {file_name} added", flush=True)
            elif status == 409:
                print(f"  {file_name} skipped: {answer['detail']}", flush=True)
            else:
                sys.exit(f"{file_name}: {status} {answer}")
        if name in ids:
            print(f"  /patient/{ids[name]}", flush=True)
    if failed:
        sys.exit(f"Not added, their download failed: {', '.join(failed)}. "
                 f"Run make demo again to add them.")
    # A patient is created with its first night, so a failed download never leaves an empty one.
    # 409: the same content is already stored, for another patient.


if __name__ == "__main__":
    main()
