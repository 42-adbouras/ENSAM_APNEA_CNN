from __future__ import annotations

import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import BinaryIO

import numpy as np

NIGHT_SUFFIX = ".csv"
PATIENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}")
NIGHT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,95}\.csv", re.IGNORECASE)
# Names accepted from the interface. Both start with a letter or digit, so neither can be hidden
# (".name"), "." or "..", and neither can hold a "/".


def _visible(path: Path) -> bool:
    return not path.name.startswith(".")
    # Skips hidden files such as .DS_Store and the "._NIGHT001.CSV" copies macOS leaves next to
    # files, which would otherwise pass as nights.


def patient_dirs(root: Path) -> dict[str, Path]:
    """Patient identifier -> its folder, in alphabetical order."""
    if not root.is_dir():
        return {}
    folders = sorted((p for p in root.iterdir() if p.is_dir() and _visible(p)),
                     key=lambda p: p.name.lower())
    return {p.name: p for p in folders}
    # Callers look a requested name up in this dict instead of joining it onto root, so a name
    # such as ".." matches nothing.


def _natural_key(name: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]
    # "NIGHT10.csv" -> ["night", 10, ".csv"]: digit runs compare as numbers, so NIGHT9 sorts before
    # NIGHT10. re.split with a group alternates text and digits, so the same positions always hold
    # the same type.


def night_files(folder: Path) -> list[Path]:
    """A patient's .csv files, oldest first."""
    files = [f for f in folder.iterdir()
             if f.is_file() and _visible(f) and f.suffix.lower() == NIGHT_SUFFIX]
    return sorted(files, key=lambda f: (f.stat().st_mtime_ns, _natural_key(f.name)))
    # Ordered by modification time, so the oldest recording is Night 1; files with the same time
    # are ordered by name.


def night_file(folder: Path, name: str) -> Path | None:
    """The night file called `name` in a patient's folder, or None."""
    return next((f for f in night_files(folder) if f.name == name), None)
    # Same rule as patient_dirs(): a requested name must equal a file actually listed.


def _is_header(first_line: bytes) -> bool:
    first_field = first_line.split(b",", 1)[0].strip()
    try:
        float(first_field)
    except ValueError:
        return True
    return False
    # Only the first field is tested, so a line such as "ecg_mv,lead_off" is a header while
    # "0.132,0" is a sample.


def read_night(path: Path) -> np.ndarray:
    """A night file -> its ECG samples, the first column, as float64."""
    with open(path, "rb") as f:
        header = _is_header(f.readline())
    return np.loadtxt(path, delimiter=",", usecols=0, skiprows=int(header), ndmin=1,
                      dtype=np.float64)
    # np.loadtxt raises ValueError on a line it cannot read as a number; the HTTP layer reports it.


@lru_cache(maxsize=4096)
def _count_samples(path: str, size: int, mtime_ns: int) -> int:
    newlines, first, last = 0, b"", b""
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            first = first or chunk
            newlines += chunk.count(b"\n")
            last = chunk
    if not first:
        return 0

    lines = newlines if last.endswith(b"\n") else newlines + 1
    # The final line counts even when the file does not end with a newline.
    if _is_header(first.split(b"\n", 1)[0]):
        lines -= 1
    return lines
    # Newlines are counted in 1 MB blocks with bytes.count(), without parsing any value.
    # size and mtime_ns are part of the cache key only, so a file replaced with new content is
    # counted again instead of served from the cache.


def count_samples(path: Path) -> int:
    stat = path.stat()
    return _count_samples(str(path), stat.st_size, stat.st_mtime_ns)


def create_patient(root: Path, name: str) -> Path:
    """Creates a new patient's folder. ValueError for a name not allowed, FileExistsError if taken."""
    name = " ".join(name.split())
    if not PATIENT_NAME.fullmatch(name):
        raise ValueError("a patient name is 1 to 64 letters, digits, spaces, _ or -, "
                         "starting with a letter or digit")
    if name.lower() in (existing.lower() for existing in patient_dirs(root)):
        raise FileExistsError(name)
    folder = root / name
    try:
        folder.mkdir()
    except FileExistsError:
        raise FileExistsError(name) from None
    return folder
    # " ".join(name.split()) drops spaces at both ends and turns each run of spaces into one, so
    # "John  Doe " becomes "John Doe". Names are compared without case, so "Patient_1" is refused
    # next to "patient_1" on every file system, not only on case-insensitive ones such as macOS's.


_HEAD_BYTES = 1 << 16


def _check_start(head: bytes) -> None:
    lines = head.splitlines()
    if len(head) == _HEAD_BYTES:
        lines = lines[:-1]
    lines = [line for line in lines if line.strip()]
    if lines and _is_header(lines[0]):
        lines = lines[1:]
    for line in lines:
        try:
            float(line.split(b",", 1)[0])
        except ValueError:
            text = line[:40].decode("utf-8", "replace")
            raise ValueError(f"not one ECG sample per line: {text!r}" if text.isprintable()
                             else "not a text file of ECG samples") from None
    # Reads the lines of the file's first 64 KB the way read_night() does: an optional header,
    # then one number per line in the first column. When the block is full, its last line is
    # skipped, since it may be cut in the middle.


def add_night(folder: Path, name: str, data: BinaryIO, modified_ns: int | None,
              min_samples: int) -> Path:
    """Saves an uploaded night file in a patient's folder.

    ValueError when the name or the content is not a night's; FileExistsError when the patient
    already has a file of that name, which is never replaced.
    """
    if not NIGHT_NAME.fullmatch(name):
        raise ValueError("a night file name must end in .csv and start with a letter or digit")
    if name.lower() in (f.name.lower() for f in folder.iterdir()):
        raise FileExistsError(name)

    head = data.read(_HEAD_BYTES)
    _check_start(head)

    target = folder / name
    partial = folder / f".{name}.partial"
    try:
        with open(partial, "wb") as out:
            out.write(head)
            shutil.copyfileobj(data, out, 1 << 20)
        samples = count_samples(partial)
        if samples < min_samples:
            raise ValueError(f"shorter than one minute ({samples} samples)")
        if modified_ns is not None:
            os.utime(partial, ns=(modified_ns, modified_ns))
        if target.exists():
            raise FileExistsError(name)
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return target
    # The file is written under a hidden ".partial" name, which night_files() skips, and renamed
    # once complete, so a half-written upload never appears as a night. os.utime gives it the
    # modification time of the original file, which sets its place in the night order.


def list_patients(root: Path) -> list[dict]:
    patients = []
    for name, folder in patient_dirs(root).items():
        files = night_files(folder)
        patients.append({
            "id": name,
            "nights": len(files),
            "latest_night": len(files) if files else None,
        })
    return patients


def list_nights(folder: Path, fs: float) -> list[dict]:
    nights = []
    for number, f in enumerate(night_files(folder), start=1):
        samples = count_samples(f)
        nights.append({
            "file": f.name,
            "number": number,
            "samples": samples,
            "duration_s": samples / fs,
            "size_bytes": f.stat().st_size,
        })
    return nights
