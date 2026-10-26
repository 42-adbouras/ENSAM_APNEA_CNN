"""
Patients' names and night recordings: which ones are accepted, and how a night's samples are read.

A night is the content of a .csv file: an optional header line, then one ECG sample per line, in
the first column. The content itself is stored in the database (store.py); this module only reads
bytes, with no web or database code.
"""
from __future__ import annotations

import io
import re

import numpy as np

PATIENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}")
NIGHT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.-]{0,95}\.csv", re.IGNORECASE)
# Both start with a letter or digit, so neither can be hidden (".name"), "." or "..", and neither
# can hold a "/".


def clean_patient_name(name: str) -> str:
    """The name as stored: spaces at both ends dropped, runs of spaces made one. ValueError if not allowed."""
    name = " ".join(name.split())
    if not PATIENT_NAME.fullmatch(name):
        raise ValueError("a patient name is 1 to 64 letters, digits, spaces, _ or -, "
                         "starting with a letter or digit")
    return name


def check_night_name(name: str) -> None:
    if not NIGHT_NAME.fullmatch(name):
        raise ValueError("a night file name must end in .csv and start with a letter or digit")


def natural_key(name: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", name)]
    # "NIGHT10.csv" -> ["night", 10, ".csv"]: digit runs compare as numbers, so NIGHT9 sorts before
    # NIGHT10. re.split with a group alternates text and digits, so the same positions always hold
    # the same type.


def _is_header(first_line: bytes) -> bool:
    first_field = first_line.split(b",", 1)[0].strip()
    try:
        float(first_field)
    except ValueError:
        return True
    return False
    # Only the first field is tested, so a line such as "ecg_mv,lead_off" is a header while
    # "0.132,0" is a sample.


def count_samples(data: bytes) -> int:
    if not data.strip():
        return 0
    lines = data.count(b"\n") + (0 if data.endswith(b"\n") else 1)
    return lines - _is_header(data.split(b"\n", 1)[0])
    # Newlines are counted without parsing any value; the final line counts even without a newline
    # after it.


_HEAD_BYTES = 1 << 16


def check_night(data: bytes, min_samples: int) -> int:
    """The number of samples in a night's content. ValueError when it is not a night's."""
    head = data[:_HEAD_BYTES]
    lines = head.splitlines()
    if len(data) > _HEAD_BYTES:
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
    samples = count_samples(data)
    if samples < min_samples:
        raise ValueError(f"shorter than one minute ({samples} samples)")
    return samples
    # Reads the lines of the first 64 KB the way read_samples() does: an optional header, then one
    # number per line in the first column. When the content is longer, the block's last line is
    # skipped, since it may be cut in the middle.


def read_samples(data: bytes) -> np.ndarray:
    """A night's content -> its ECG samples, the first column, as float64."""
    header = _is_header(data.split(b"\n", 1)[0])
    return np.loadtxt(io.BytesIO(data), delimiter=",", usecols=0, skiprows=int(header), ndmin=1,
                      dtype=np.float64)
    # np.loadtxt raises ValueError on a line it cannot read as a number; the HTTP layer reports it.
