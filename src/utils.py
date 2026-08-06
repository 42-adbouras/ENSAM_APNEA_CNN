# ────────────────────────────────────────────────────────────────
# PART 1 : LIBRARY IMPORTS
# ────────────────────────────────────────────────────────────────

from __future__ import annotations
# Allows the modern type annotation syntax (for example
# "np.ndarray | None") on older Python versions.
# No effect whatsoever on the computation.

import os
# Reads the environment variables
# Used here so the data location can differ between a local run
# and the Docker container, without touching the code

from pathlib import Path
# Builds file paths in a portable way
# Works the same under Linux (Docker) and under Windows

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt
# Signal filtering tools:
# butter      : designs a Butterworth filter (flat response in the passband)
# sosfiltfilt : applies that filter forward then backward (no phase shift)
# iirnotch    : designs a very narrow band-stop filter
# filtfilt    : applies that notch, also forward then backward

import numpy as np
# Library for mathematical operations on arrays
# Used to manipulate the ECG signal sample by sample

import wfdb
# Official PhysioNet library
# Reads the WFDB format: signal (.dat) + annotations (.apn, .qrs)

# ────────────────────────────────────────────────────────────────
# PART 2 : DATA LOCATION AND SAMPLING RATE
# ────────────────────────────────────────────────────────────────

DATA_ROOT = Path(os.environ.get("DATA_PATH", "data"))
# Root of the data folder, read from the environment so that the same
# code runs in both places:
#
# local run : DATA_PATH is unset -> "data", relative to the project
#             root. The scripts are therefore launched from the root,
#             not from inside src/
# container : DATA_PATH=/data (set in .env) -> the volume docker-compose
#             mounts from ./data
#
# A single variable, so no path is ever hard-coded twice.

DATA_DIR = DATA_ROOT / "apnea-ecg"
# Folder holding the raw recordings downloaded from PhysioNet

RESULTS_ROOT = Path(os.environ.get("RESULTS_PATH", "results"))
# Where trained models and plots are written. Same env-driven trick as
# DATA_ROOT: "results" locally, /results (the mounted volume) in Docker —
# so nothing is written to the container's throwaway filesystem.

FS = 100
# Sampling rate of the Apnea-ECG dataset: 100 Hz
# In other words 100 measurement points per second
#
# This value drives every time-related computation in the project:
# a 60-second window = 60 × 100 = 6000 points
# It is checked on every record when loading (PART 6)

# ────────────────────────────────────────────────────────────────
# PART 3 : DATASET RECORD LISTS
# ────────────────────────────────────────────────────────────────

A_RECORDS = [f"a{i:02d}" for i in range(1, 21)]
# 20 recordings of group A: SEVERE apnea
# Builds the list ["a01", "a02", ..., "a20"]
# The {i:02d} format forces two digits: 1 -> "01"

B_RECORDS = [f"b{i:02d}" for i in range(1, 6)]
# 5 recordings of group B: MODERATE apnea
# ["b01", ..., "b05"]

C_RECORDS = [f"c{i:02d}" for i in range(1, 11)]
# 10 recordings of group C: NEARLY NORMAL subjects
# ["c01", ..., "c10"]
# Essential: without them the model would see almost nothing but apnea

X_RECORDS = [f"x{i:02d}" for i in range(1, 36)]
# 35 recordings of group X: the official PhysioNet TEST set
# ["x01", ..., "x35"]
# Never used during training

VAL_RECORDS = ["a05", "a10", "a15", "a16", "a17", "b04", "c04", "c08"]
# 8 recordings held out for VALIDATION
# Picked to cover all three groups: severe (a), moderate (b),
# nearly normal (c) — so validation mirrors the full dataset
#
# They are used to monitor training (EarlyStopping, threshold choice)
# without ever touching the test set

LEARNING_RECORDS = A_RECORDS + B_RECORDS + C_RECORDS
# The 35 annotated recordings available for learning
# 20 (a) + 5 (b) + 10 (c) = 35

# ────────────────────────────────────────────────────────────────
# PART 4 : TRAIN / VALIDATION / TEST SPLIT
# ────────────────────────────────────────────────────────────────

SPLITS = {
    "train": [r for r in LEARNING_RECORDS if r not in VAL_RECORDS],
    "val":   VAL_RECORDS,
    "test":  X_RECORDS,
}
# Split BY RECORD, never by minute
#
# Key methodological point:
# one patient must appear in one set only.
# Splitting by minute would put the heartbeats of the same heart in
# both training and test: the model would recognise the patient
# instead of recognising apnea, and the reported score would be
# artificially inflated.
#
# train : 27 recordings (35 - 8 held out for validation)
# val   :  8 recordings
# test  : 35 recordings of group x

WINDOW_SAMPLES = 60 * FS                    # 6000
# Length of one analysis window, expressed in measurement points
# 60 seconds × 100 Hz = 6000 points
#
# The 60 s duration is not a free choice: the Apnea-ECG dataset
# provides one label (apnea / normal) per minute of recording

# ────────────────────────────────────────────────────────────────
# PART 5 : ECG SIGNAL CLEANING
# ────────────────────────────────────────────────────────────────

def clean_ecg(x: np.ndarray, fs: int = FS) -> np.ndarray:
    # Denoises the raw ECG signal before any analysis
    #
    # An ECG recorded over a whole night contains, on top of the
    # heart: breathing, the sleeper's movements, the mains supply
    # and electrode noise. These have to be removed, otherwise the
    # model learns the noise instead of the apnea.
    #
    # x  : raw ECG signal (1-dimensional array)
    # fs : sampling rate, 100 Hz by default

    sos = butter(4, [0.5, 40], btype="band", fs=fs, output="sos")
    # Designs a 4th-order Butterworth BAND-PASS filter
    #
    # Keeps only the frequencies between 0.5 Hz and 40 Hz:
    # below 0.5 Hz -> baseline wander
    #                 (breathing, body movements)
    # above 40 Hz  -> muscle noise and electronic noise
    # in between   -> the useful ECG content (QRS complexes)
    #
    # output="sos" : "Second-Order Sections" form, numerically more
    # stable than the classic form for high-order filters

    x = sosfiltfilt(sos, x)
    # Applies the band-pass filter
    #
    # sosfiltfilt filters the signal twice: once forward, once
    # backward. The two phase shifts cancel out: the R peaks stay
    # exactly at their original position in time.
    # That matters here, because the peak positions are what the RR
    # intervals are computed from afterwards.

    b, a = iirnotch(50, Q=30, fs=fs)
    # Designs a NOTCH (band-stop) filter centred on 50 Hz
    # 50 Hz = mains frequency (Europe, Morocco)
    # This interference is picked up by the electrode cables
    #
    # Q=30 : quality factor — the larger Q, the narrower the notch.
    # Here 50 Hz is removed without touching the neighbouring frequencies

    x = filtfilt(b, a, x)
    # Applies the notch, again forward then backward
    # so the peaks keep their position in time

    x = np.nan_to_num(x, nan=np.nanmedian(x))
    # Replaces missing values (NaN) with the median of the signal
    #
    # The median is preferred over the mean: it is not pulled up by
    # the R peaks nor by artifacts.
    # Without this step, a single NaN would contaminate every
    # computation downstream.

    return np.clip(x, -5, 5)
    # Clips the signal between -5 and +5 mV
    #
    # A normal ECG stays well below that limit. Anything beyond it
    # is an artifact (disconnected electrode, sudden movement):
    # it is bounded rather than removed, so that the time axis is
    # never shifted.

# ────────────────────────────────────────────────────────────────
# PART 6 : LOADING ONE PHYSIONET RECORD
# ────────────────────────────────────────────────────────────────

def load_record(rec: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (ecg, ann_times, ann_labels)."""
    # Loads one complete recording of the dataset
    #
    # rec : record name, for example "a01"
    #
    # Returns three items:
    # ecg        : the raw ECG signal of the whole night
    # ann_times  : the start instant of each annotated minute,
    #              expressed as a sample number (0, 6000, 12000, ...)
    # ann_labels : the label of each minute — "A" = apnea, "N" = normal

    path = str(DATA_DIR / rec)
    # Base path without extension, for example "data/apnea-ecg/a01"
    # wfdb appends the .dat, .hea and .apn extensions itself

    signals, fields = wfdb.rdsamp(path)
    # Reads the signal and its metadata
    # signals : array (number_of_points, number_of_channels)
    # fields  : dictionary describing the record (fs, units, duration)

    assert fields["fs"] == FS, f"unexpected fs={fields['fs']} for {rec}"
    # Safety check: the sampling rate read back must indeed be 100 Hz
    #
    # If a record were sampled differently, every duration computed
    # downstream (windows, RR intervals) would be wrong without any
    # error being raised. Better to stop right here.

    ann = wfdb.rdann(path, "apn")
    # Reads the .apn annotation file
    # It holds the diagnosis established by the experts, minute by minute
    # This is the "ground truth" used to train and to evaluate the model

    return signals[:, 0], ann.sample, np.array(ann.symbol)
    # signals[:, 0] : first channel only — the ECG
    # ann.sample    : annotation positions, as sample numbers
    # ann.symbol    : "A" / "N" symbols, converted to a numpy array
    #                 so that vectorised operations can be used later
