from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import numpy as np

NIGHT_SUFFIXES = {".csv", ".txt"}
NIGHT_NAME = re.compile(r"^NIGHT(\d+)$", re.IGNORECASE)
# Matched against the name without its suffix, ignoring case: NIGHT001.CSV and night1.csv are
# both night 1.


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


def night_number(path: Path) -> int | None:
    """The night's number from its file name, or None when the name is not a night's."""
    match = NIGHT_NAME.match(path.stem)
    if match is None or path.suffix.lower() not in NIGHT_SUFFIXES:
        return None
    return int(match.group(1))
    # int() drops the leading zeros, so NIGHT007 and NIGHT7 are both night 7.


def night_files(folder: Path) -> list[Path]:
    """A patient's night files, in the order of their numbers."""
    files = [f for f in folder.iterdir()
             if f.is_file() and _visible(f) and night_number(f) is not None]
    return sorted(files, key=lambda f: (night_number(f), f.name.lower()))
    # Sorted on the number, not on the name: as text, NIGHT10 would come before NIGHT9.


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


def list_patients(root: Path) -> list[dict]:
    patients = []
    for name, folder in patient_dirs(root).items():
        files = night_files(folder)
        patients.append({
            "id": name,
            "nights": len(files),
            "latest_night": night_number(files[-1]) if files else None,
        })
    return patients


def list_nights(folder: Path, fs: float) -> list[dict]:
    nights = []
    for f in night_files(folder):
        samples = count_samples(f)
        nights.append({
            "file": f.name,
            "number": night_number(f),
            "samples": samples,
            "duration_s": samples / fs,
            "size_bytes": f.stat().st_size,
        })
    return nights
