"""
Minimal raw-ECG preprocessing raw records -> (N, 6000) one-minute windows.

Produces exactly the four arrays modele_CNN_commente.py loads from '../data':
X_train_apnea.npy, y_train_apnea.npy, X_test_apnea.npy, y_test_apnea.npy
"""

# ────────────────────────────────────────────────────────────────
# PART 1 : LIBRARY IMPORTS
# ────────────────────────────────────────────────────────────────

import numpy as np
# Library for mathematical operations on arrays
# Used here for slicing, normalising and array stacking

from utils import (DATA_ROOT, FS, SPLITS, WINDOW_SAMPLES,
                   clean_ecg, load_record)
# Everything shared with the rest of the pipeline is imported, never copied:
# DATA_ROOT      : root of the data folder (local "data" or /data in Docker)
# FS             : sampling rate, 100 Hz
# SPLITS         : train / val / test split, by record
# WINDOW_SAMPLES : 6000 points = one 60 s minute
# clean_ecg      : band-pass 0.5-40 Hz + 50 Hz notch, then clip to +-5 mV
# load_record    : reads the ECG and its apnea annotations
#
# Importing rather than copying is what guarantees this script cannot
# drift away from the main pipeline.
#
# No wfdb import here, and no .qrs beat annotations: this model reads the
# ECG waveform itself, so nothing has to be detected beat by beat.

# ────────────────────────────────────────────────────────────────
# PART 2 : OUTPUT FOLDER AND FILE NAMES
# ────────────────────────────────────────────────────────────────

OUT_DIR = DATA_ROOT / "processed"
# The four arrays land in data/processed/, kept apart from the raw WFDB
# records download.py writes to data/apnea-ecg/. The consumer loads them
# from SAVE_PATH = '../data/processed' (launched from src/), which resolves
# to the same folder both locally and in the container.
#
# Built from DATA_ROOT, so they land inside the mounted volume when the
# script runs in the container (/data/processed), and stay on the host.

SUFFIX = "_apnea"
# The consumer loads 'X_train_apnea.npy', not 'X_train.npy'.
#
# That suffix is also what keeps this script from colliding with the
# RR-tachogram pipeline: its arrays share the same data/processed/ folder
# under the plain names (X_train.npy), so the two representations coexist
# on disk and neither can silently overwrite the other's frozen results.

# ────────────────────────────────────────────────────────────────
# PART 3 : REPRESENTATION CONSTANTS
# ────────────────────────────────────────────────────────────────

SEQ_LEN = WINDOW_SAMPLES    # 6000
# Length of one sample: 60 s x 100 Hz = 6000 raw ECG points
#
# Not a free choice: the Apnea-ECG dataset gives one expert label per
# minute, so the window has to be exactly the minute that was labelled.
#
# Divisible by 2 four times over (6000 -> 3000 -> 1500 -> 750 -> 375),
# which is what the CNN's four successive MaxPooling1D(2) need.

# The array is saved 2-D, (N, 6000), with NO channel axis: the consumer adds
# it itself with X_train[..., np.newaxis] before building the model. Saving
# (N, 6000, 1) here would make it (N, 6000, 1, 1) there, and Conv1D would reject it.
N_CHANNELS = 1
# One channel only: the ECG waveform. Kept as a named constant because
# PART 9 uses it to state what the saved shape must NOT be.

TRAIN_RECORDS = SPLITS["train"] + SPLITS["val"]
# The 35 annotated learning records: a01-a20 (severe), b01-b05 (moderate),
# c01-c10 (nearly normal).
#
# The two splits are recombined on purpose. utils.SPLITS holds a proper
# by-record train/val separation, but this consumer does its own
# validation_split=0.2 inside model.fit and therefore expects the whole
# learning set in one array. Handing it only SPLITS["train"] would silently
# shrink training to 27 records and leave 8 unused.
#
# Consequence worth knowing: the consumer's internal split is by MINUTE,
# so the same sleeper appears on both sides of it. That inflates its
# val_auc, which is only used for EarlyStopping — the reported score
# comes from the x records below, which stay untouched.

TEST_RECORDS = SPLITS["test"]
# The 35 records of group x: the official PhysioNet test set.
# Never seen during training. 17 248 labelled minutes in total.

ZSCORE_PER_WINDOW = True
# Whether each window is centred and scaled by its own statistics.
#
# Why per-window and not fixed constants: the absolute height of an ECG
# is electrode placement, skin impedance and body build, not physiology.
# Two sleepers with identical rhythms can differ by a factor of three in
# millivolts. Left alone, that offset is the loudest thing in the input,
# and with only 35 training sleepers the network learns to recognise the
# PATIENT rather than the apnea.
#
# What survives the z-score is the SHAPE of the minute — the QRS
# morphology and its modulation — which is what actually carries the event.
#
# Set to False to feed the millivolts as clean_ecg leaves them: PART 9
# checks whichever of the two was chosen.

LABEL_DTYPE = np.float32
# Labels are stored as floats, not as the single byte 0/1 would need.
#
# This is not a style choice, it is what the consumer's loss function
# requires. Its focal_loss multiplies y_true by tf.math.log(y_pred)
# directly, without casting anything itself. Keras only auto-casts y_true
# to the prediction dtype when the two are BOTH floating or BOTH integer —
# so an integer label array reaches the loss unconverted and TensorFlow
# refuses the multiplication:
#
#   TypeError: Input 'y' of 'Mul' Op has type float32
#              that does not match type int8 of argument 'x'.
#
# float32 matches the model output, so the cast is a no-op and the loss runs.
# sklearn's confusion_matrix / roc_auc_score / f1_score all accept 0.0 and
# 1.0 exactly as they accept 0 and 1, and Keras' class_weight lookup casts
# the labels to int64 itself, so nothing downstream notices the difference.

EPS = 1e-6
# Added to the standard deviation before dividing.
# A flat window (disconnected electrode) has std = 0, and 0/0 would put
# NaN into the array — which turns the whole training loss into NaN
# and destroys every weight of the network in one step.

SHUFFLE_SEED = 42
# Seed for the one shuffle applied to the training set in PART 8.
# Same value the consumer fixes for random / numpy / tensorflow, so the
# whole chain stays reproducible from one number.

# ────────────────────────────────────────────────────────────────
# PART 4 : ONE LABELLED MINUTE -> ONE (6000,) WINDOW
# ────────────────────────────────────────────────────────────────


def window(ecg_clean: np.ndarray, start: int) -> np.ndarray:
    """The 60 s starting at `start` -> a (6000,) row.

    ecg_clean : the cleaned ECG of the whole recording
    start     : first sample of the LABELLED minute
    """
    # Core function of the script: cuts one labelled minute out of the
    # night and puts it on the scale the CNN expects.

    segment = ecg_clean[start:start + SEQ_LEN].astype(np.float32)
    # A plain slice, no interpolation: the ECG is already on a regular
    # 100 Hz grid, so the minute is read exactly as it was recorded.
    # float32 : the precision used by TFLite Micro on the ESP32.

    if ZSCORE_PER_WINDOW:
        segment = (segment - segment.mean()) / (segment.std() + EPS)
    # Subtract the mean, divide by the standard deviation of THIS minute.
    # See PART 3 for why the absolute level is deliberately discarded.

    return np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    # Final safety net: any NaN or infinity left is forced to 0.
    # clean_ecg already replaces NaN and clips the signal, so this should
    # never fire — it costs nothing and it is the last line of defence.


# ────────────────────────────────────────────────────────────────
# PART 5 : ONE RECORD -> ALL ITS USABLE MINUTES
# ────────────────────────────────────────────────────────────────


def sequencize(ecg_clean: np.ndarray, times: np.ndarray, labels: np.ndarray):
    """All usable minutes of one record -> (n, 6000) windows and (n,) labels."""
    # Walks through a full night, minute by minute
    #
    # ecg_clean : the cleaned ECG of the whole recording
    # times     : start of each annotated minute
    # labels    : "A" or "N" for each of those minutes

    windows, y = [], []
    # Accumulators: the segments on one side, the labels on the other

    for start, label in zip(times, labels):
        # zip pairs each start instant with its label

        if start + WINDOW_SAMPLES > len(ecg_clean) or label not in ("A", "N"):
            continue
        # Two reasons to skip a minute:
        # 1. it runs past the end of the recording — the window would be
        #    shorter than 6000 points and np.stack below would refuse it
        # 2. its label is neither "A" nor "N" — an unusable annotation

        windows.append(window(ecg_clean, start))
        y.append(1 if label == "A" else 0)
        # Every in-window A/N minute yields a (6000,) row — none are dropped.
        # Label converted to a number for the network:
        # "A" (apnea) -> 1 | "N" (normal) -> 0

    # stack (not concatenate) — it creates the sample axis: 489 x (6000,) -> (489, 6000)
    return np.stack(windows).astype(np.float32), np.asarray(y, dtype=LABEL_DTYPE)
    # np.concatenate would have glued the minutes end to end into one
    # (2934000,) line, losing the minute boundaries entirely.


# ────────────────────────────────────────────────────────────────
# PART 6 : ONE SPLIT -> ALL ITS RECORDS COMBINED
# ────────────────────────────────────────────────────────────────


def process_split(records: list[str]):
    """Every record in one split, concatenated along the sample axis."""
    # Processes a whole list of records (learning set or test set)
    # and gathers the result into a single pair of arrays

    X_parts, y_parts = [], []
    for rec in records:
        ecg, times, labels = load_record(rec)
        # Reads the raw signal and its expert annotations

        X, y = sequencize(clean_ecg(ecg), times, labels)
        # The full chain for one record:
        # clean_ecg  -> band-pass, notch, clip
        # sequencize -> converts every labelled minute into a (6000,) row

        print(f"  {rec}: {len(y):4d} windows ({int(y.sum()):3d} apnea)")
        # Progress line: number of minutes kept, and how many are apnea
        # y.sum() counts the 1s, since normal minutes are worth 0.
        # int() because the labels are floats (see LABEL_DTYPE) and the
        # :3d format only accepts an integer.

        X_parts.append(X)
        y_parts.append(y)

    return np.concatenate(X_parts), np.concatenate(y_parts)
    # concatenate here, not stack: the per-record axis of minutes is
    # exactly what has to be merged, so that 35 records give a single
    # array of N minutes.


# ────────────────────────────────────────────────────────────────
# PART 7 : SAFETY CHECKS ON THE PRODUCED DATA
# ────────────────────────────────────────────────────────────────


def validate(X: np.ndarray, y: np.ndarray) -> None:
    """Fail loudly on the data bugs that would otherwise train silently."""
    # These checks are deliberately assertions and not warnings:
    # a data bug that goes through silently costs hours of training
    # on data that means nothing. Better to crash immediately.

    assert X.ndim == 2 and X.shape[1] == SEQ_LEN, f"bad shape {X.shape}"
    # Must be exactly (N, 6000), two dimensions.
    # X.ndim == 2 is the check that matters: a saved (N, 6000, 1) would
    # become (N, 6000, 1, 1) once the consumer adds its own channel axis,
    # and Conv1D would reject it with an unhelpful rank error.

    assert len(X) == len(y), "X and y are out of step"
    # As many labels as minutes: a shift by one would mean every
    # sample is trained against the wrong answer

    assert not np.isnan(X).any() and not np.isinf(X).any(), "NaN/Inf in X"
    # No NaN and no infinity: a single one turns the loss into NaN
    # and destroys the whole network

    assert set(np.unique(y)) <= {0, 1}, f"labels outside 0/1: {np.unique(y)}"
    # Only "A" and "N" minutes were kept, so nothing else can appear.
    # A third value would mean an annotation symbol slipped through PART 5.
    # The labels are floats, but 0.0 == 0 and 1.0 == 1, so the comparison
    # against the integer set holds.

    assert np.issubdtype(y.dtype, np.floating), \
        f"labels must be floating, got {y.dtype}"
    # The check that protects the consumer's focal_loss: see LABEL_DTYPE.
    # An integer label array only fails once training has already started,
    # several minutes in, with a TypeError from deep inside autograph.

    assert 0.25 <= y.mean() <= 0.55, f"apnea fraction {y.mean():.3f} out of range"
    # The share of apnea must stay near the expected 38%
    # A value far off would signal a split gone wrong or annotations
    # misread — not a real property of the dataset

    # The amplitude check depends on which scaling PART 3 selected. Either way
    # it is the one check that catches a normalisation that silently did nothing.
    if ZSCORE_PER_WINDOW:
        assert abs(float(X.mean())) < 0.05, \
            f"z-scored windows not centred (mean {X.mean():.3f})"
        # Each window was centred on its own mean, so the mean over the
        # whole split must sit on zero. A drift away from it means the
        # normalisation was skipped or applied along the wrong axis.

        assert 0.5 <= float(X.std()) <= 2.0, \
            f"z-scored windows badly scaled (std {X.std():.3f})"
        # Same idea for the spread: each window was divided by its own
        # standard deviation, so the pooled one must land near 1.
    else:
        assert np.abs(X).max() <= 5.0 + EPS, "raw ECG outside the +-5 mV clip"
        # clean_ecg clips to +-5 mV, so nothing may exceed it.
        # Anything beyond means the clipping step was bypassed.


# ────────────────────────────────────────────────────────────────
# PART 8 : ONE SPLIT -> ITS TWO FILES ON DISK
# ────────────────────────────────────────────────────────────────


def save_split(name: str, records: list[str], shuffle: bool) -> None:
    """Process, check and write X_<name>_apnea.npy and y_<name>_apnea.npy."""

    print(f"\n[{name}] {len(records)} records")
    X, y = process_split(records)
    # Full conversion of the split

    if shuffle:
        order = np.random.default_rng(SHUFFLE_SEED).permutation(len(y))
        X, y = X[order], y[order]
    # Why the training set is shuffled here, once, before it is saved:
    #
    # Keras' validation_split takes the LAST 20% of the array as given,
    # before any shuffling of its own. Records are concatenated in order
    # (a, then b, then c), so that last 20% would be almost entirely the
    # c records — the nearly normal sleepers. The consumer would then
    # monitor val_auc on a set that barely contains apnea, and
    # EarlyStopping would restore weights chosen on a meaningless score.
    #
    # A fixed seed keeps the shuffle identical from one run to the next.
    # The test set is never shuffled: it is only ever read in full, and
    # leaving it in record order keeps it comparable across experiments.

    validate(X, y)
    # Checks BEFORE saving: nothing invalid ever reaches the disk

    np.save(OUT_DIR / f"X_{name}{SUFFIX}.npy", X)
    np.save(OUT_DIR / f"y_{name}{SUFFIX}.npy", y)
    # Saves in numpy format, read directly by modele_CNN_commente.py

    size_mb = X.nbytes / 1e6
    print(f"  -> X_{name}{SUFFIX}.npy {X.shape}  "
          f"apnea={y.mean():.3f}  {size_mb:.0f} MB")
    # Summary line: final shape, share of apnea, and size on disk.
    # The size is printed because raw ECG is bulky — roughly 400 MB per
    # split, against 13 MB for the same minutes as an RR tachogram.


# ────────────────────────────────────────────────────────────────
# PART 9 : MAIN ENTRY POINT
# ────────────────────────────────────────────────────────────────


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Creates the output folder if needed
    # exist_ok=True : running the script twice raises no error

    save_split("train", TRAIN_RECORDS, shuffle=True)
    save_split("test", TEST_RECORDS, shuffle=False)
    # Two splits only, not three: the consumer carves its own validation
    # set out of the training array with validation_split=0.2.
    # Produces 4 files: X_train_apnea, y_train_apnea, X_test_apnea, y_test_apnea


if __name__ == "__main__":
    main()
    # Runs main() only when the file is launched directly.
    # If another script imports this one, nothing runs — the functions
    # simply become available.