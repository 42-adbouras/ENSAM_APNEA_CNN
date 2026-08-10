"""
Raw-ECG preprocessing: WFDB records -> (N, 6000) one-minute windows.

Produces THREE splits, each as exactly two arrays in data/processed/:

    X_<split>_apnea.npy   (N, 6000) float32   the windows themselves
    y_<split>_apnea.npy   (N,)      float32   0 = normal, 1 = apnea

    split     records                                   minutes
    train     27  (a/b/c minus the validation hold-out)  13 158
    val        8  (a05 a10 a15 a16 a17 b04 c04 c08)       3 865
    test      35  (x01-x35, the official PhysioNet set)  17 248

WHY THREE SPLITS ON DISK
------------------------
Earlier versions wrote only train and test, with the validation records folded
into the training array, and left the consumer to carve validation back out with
a boolean mask. That worked, but it put a methodological decision -- the one this
project got wrong for its entire first life -- inside the training script where
it could be quietly skipped.

Writing three splits makes the by-record separation a property of the data on
disk. A consumer cannot accidentally train on validation records, because they
are not in the file it loads.

The split is BY RECORD, never by minute. One patient appears in exactly one
split. Splitting by minute puts the same sleeping body on both sides, and the
model learns to recognise the PATIENT rather than the apnea: the original model
scored 0.9725 validation AUC that way against a real test AUC of 0.849. With the
by-record split, validation AUC 0.918 predicted test AUC 0.899 almost exactly.

WHAT THIS SCRIPT NO LONGER SAVES
--------------------------------
Two sidecars were dropped, in this order, and both losses are real.

quality_<split>_apnea.npy held per-window signal diagnostics measured BEFORE
normalisation -- raw standard deviation in mV, and the fraction of samples
sitting on the +-5 mV clip. That measurement can only be taken here: the z-score
in PART 4 divides every window by its own spread, so afterwards a minute of
electrode noise is indistinguishable from a minute of real ECG. It went because
nothing ever read it; its one consumer was a training switch that stayed off for
every run this project made.

The record identity of each window went next -- first as a records_ array, then
as a manifest of per-record counts. It existed for per-record scoring, and
losing it means pooled averages are now the only view available. That matters:
the run that motivated the feature had a pooled AUC of 0.892 while its worst
single recording scored 0.165, well below chance on a patient who would have
worn the device. Nothing downstream can surface that any more.

Both are recoverable from git history if either is wanted back; the reasoning
above is the part worth re-reading before doing so.

WHAT HAS NOT CHANGED
--------------------
Nothing is dropped from the DATA. The x records are the published benchmark
quoted over all 17 248 of their minutes, a real device has to answer for every
minute of the night including the noisy ones, and dropping minutes is a
training-time decision rather than a preprocessing one.
"""

# ────────────────────────────────────────────────────────────────
# PART 1 : LIBRARY IMPORTS
# ────────────────────────────────────────────────────────────────

import numpy as np
# Library for mathematical operations on arrays
# Used here for slicing, normalising and array stacking

from utils import (FS, PROCESSED_DIR, SPLITS, SUFFIX, WINDOW_SAMPLES,
                   clean_ecg, load_record)
# Everything shared with the rest of the pipeline is imported, never copied:
# FS             : sampling rate, 100 Hz
# PROCESSED_DIR  : where the arrays land (data/processed, or /data/processed)
# SPLITS         : train / val / test split, by record
# SUFFIX         : "_apnea", what keeps this pipeline's files distinct
# WINDOW_SAMPLES : 6000 points = one 60 s minute
# clean_ecg      : band-pass 0.5-40 Hz + 50 Hz notch, then clip to +-5 mV
# load_record    : reads the ECG and its apnea annotations
#
# PROCESSED_DIR and SUFFIX in particular are imported rather than restated:
# this script writes those files and model.py reads them, so a private copy of
# either constant in either file is a chance for the writer and the reader to
# disagree about the path.
#
# No wfdb import here, and no .qrs beat annotations: this model reads the ECG
# waveform itself, so nothing has to be detected beat by beat.

# ────────────────────────────────────────────────────────────────
# PART 2 : REPRESENTATION CONSTANTS
# ────────────────────────────────────────────────────────────────

SEQ_LEN = WINDOW_SAMPLES    # 6000
# Length of one sample: 60 s x 100 Hz = 6000 raw ECG points
#
# Not a free choice: the Apnea-ECG dataset gives one expert label per minute, so
# the window has to be exactly the minute that was labelled.
#
# Divisible by 2 four times over (6000 -> 3000 -> 1500 -> 750 -> 375), which is
# what the CNN's four downsampling stages need.

# The array is saved 2-D, (N, 6000), with NO channel axis: the consumer adds it
# itself with X[..., np.newaxis] before building the model. Saving (N, 6000, 1)
# here would make it (N, 6000, 1, 1) there, and Conv1D would reject it.

SPLIT_RECORDS = {
    "train": SPLITS["train"],   # 27 records: a/b/c minus the validation hold-out
    "val":   SPLITS["val"],     #  8 records: a05 a10 a15 a16 a17 b04 c04 c08
    "test":  SPLITS["test"],    # 35 records: x01-x35, never seen during training
}
# utils.SPLITS is the single source of truth for who goes where. The validation
# eight were chosen to span all three severity groups -- severe (a), moderate
# (b), near-normal (c) -- so validation mirrors the dataset instead of
# over-representing easy sleepers.

ZSCORE_PER_WINDOW = True
# Whether each window is centred and scaled by its own statistics.
#
# Why per-window and not fixed constants: the absolute height of an ECG is
# electrode placement, skin impedance and body build, not physiology. Two
# sleepers with identical rhythms can differ by a factor of three in millivolts.
# Left alone, that offset is the loudest thing in the input, and with only 27
# training sleepers the network learns to recognise the PATIENT rather than the
# apnea.
#
# What survives the z-score is the SHAPE of the minute -- the QRS morphology and
# its modulation -- which is what actually carries the event.

LABEL_DTYPE = np.float32
# Labels are stored as floats, not as the single byte 0/1 would need.
#
# Keras only auto-casts y_true to the prediction dtype when the two are BOTH
# floating or BOTH integer. A custom loss that multiplies y_true by a float
# tensor without casting first will therefore reject an integer label array:
#
#   TypeError: Input 'y' of 'Mul' Op has type float32
#              that does not match type int8 of argument 'x'.
#
# float32 matches the model output, so the cast is a no-op and the loss runs.
# sklearn's metrics accept 0.0/1.0 exactly as they accept 0/1.

EPS = 1e-6
# Added to the standard deviation before dividing. A flat window (disconnected
# electrode) has std = 0, and 0/0 would put NaN into the array -- which turns the
# whole training loss into NaN and destroys every weight in one step.

CLIP_MV = 5.0
# The limit clean_ecg() clips to. Repeated here only so PART 5 can assert the
# un-normalised path stays inside it; the clipping itself happens in utils.


# ────────────────────────────────────────────────────────────────
# PART 3 : ONE LABELLED MINUTE -> ONE (6000,) WINDOW
# ────────────────────────────────────────────────────────────────


def window(ecg_clean: np.ndarray, start: int) -> np.ndarray:
    """The 60 s starting at `start` -> a (6000,) row."""
    # Core function of the script: cuts one labelled minute out of the night and
    # puts it on the scale the CNN expects.

    segment = ecg_clean[start:start + SEQ_LEN].astype(np.float32)
    # A plain slice, no interpolation: the ECG is already on a regular 100 Hz
    # grid, so the minute is read exactly as it was recorded.
    # float32 : the precision TFLite Micro uses on the ESP32.

    if ZSCORE_PER_WINDOW:
        segment = (segment - segment.mean()) / (segment.std() + EPS)
    # Subtract the mean, divide by the standard deviation of THIS minute.
    # See PART 2 for why the absolute level is deliberately discarded.

    return np.nan_to_num(segment, nan=0.0, posinf=0.0, neginf=0.0)
    # Final safety net: any NaN or infinity left is forced to 0. clean_ecg
    # already replaces NaN and clips, so this should never fire -- it costs
    # nothing and it is the last line of defence.


# ────────────────────────────────────────────────────────────────
# PART 4 : ONE RECORD -> ALL ITS USABLE MINUTES
# ────────────────────────────────────────────────────────────────


def sequencize(ecg_clean: np.ndarray, times: np.ndarray, labels: np.ndarray):
    """All usable minutes of one record -> (windows, labels)."""
    windows, y = [], []
    # Accumulators kept strictly in step: index i of each list describes the same
    # minute. PART 6 asserts that this stayed true.

    for start, label in zip(times, labels):
        if start + WINDOW_SAMPLES > len(ecg_clean) or label not in ("A", "N"):
            continue
        # The ONLY two reasons a minute is ever skipped:
        # 1. it runs past the end of the recording -- the window would be shorter
        #    than 6000 points and np.stack below would refuse it
        # 2. its label is neither "A" nor "N" -- an unusable annotation
        # Both are structural, never quality-based: see the module docstring.

        windows.append(window(ecg_clean, start))
        y.append(1 if label == "A" else 0)
        # "A" (apnea) -> 1 | "N" (normal) -> 0

    # stack (not concatenate) -- it creates the sample axis: 489 x (6000,) -> (489, 6000)
    return np.stack(windows).astype(np.float32), np.asarray(y, dtype=LABEL_DTYPE)
    # np.concatenate would have glued the minutes end to end into one (2934000,)
    # line, losing the minute boundaries entirely.


# ────────────────────────────────────────────────────────────────
# PART 5 : ONE SPLIT -> ALL ITS RECORDS COMBINED
# ────────────────────────────────────────────────────────────────


def process_split(records: list[str]):
    """Every record in one split -> (X, y)."""
    X_parts, y_parts, counts = [], [], []

    for rec in records:
        ecg, times, labels = load_record(rec)
        X, y = sequencize(clean_ecg(ecg), times, labels)
        # clean_ecg  -> band-pass, notch, clip
        # sequencize -> every labelled minute into a (6000,) row, plus its label

        assert len(y) > 0, f"{rec} yielded no usable minutes"
        # Checked per record because the split arrays no longer record who
        # contributed what. A record that exists on disk but produces nothing
        # would otherwise leave the split quietly smaller than intended, with
        # no trace of which patient went missing.

        print(f"  {rec}: {len(y):4d} windows ({int(y.sum()):3d} apnea)")
        # int() because the labels are floats (see LABEL_DTYPE) and :3d only
        # accepts an integer. This console line is now the ONLY place the
        # per-record breakdown appears -- nothing on disk carries it.

        X_parts.append(X)
        y_parts.append(y)
        counts.append(len(y))
        # How many minutes this record contributed. Once the arrays are
        # concatenated this is the only thing left that says where one
        # recording ends and the next begins.

    return (np.concatenate(X_parts), np.concatenate(y_parts),
            np.asarray(counts, dtype=np.int32))
    # concatenate, not stack: the per-record axis of minutes is exactly what has
    # to be merged, so N records give a single array of M minutes.


# ────────────────────────────────────────────────────────────────
# PART 6 : SAFETY CHECKS ON THE PRODUCED DATA
# ────────────────────────────────────────────────────────────────


def validate(X, y, counts) -> None:
    """Fail loudly on the data bugs that would otherwise train silently."""
    # Deliberately assertions and not warnings: a data bug that goes through
    # silently costs hours of training on data that means nothing.

    assert X.ndim == 2 and X.shape[1] == SEQ_LEN, f"bad shape {X.shape}"
    # Must be exactly (N, 6000). X.ndim == 2 is the check that matters: a saved
    # (N, 6000, 1) would become (N, 6000, 1, 1) once the consumer adds its own
    # channel axis, and Conv1D would reject it with an unhelpful rank error.

    assert len(X) == len(y), f"X and y out of step: {len(X)} vs {len(y)}"
    # Both describe the same minutes in the same order. A shift by one means
    # samples trained against the wrong answer.

    assert int(counts.sum()) == len(y), \
        f"record counts sum to {int(counts.sum())}, not {len(y)}"
    # counts is the record axis. If it does not add up, test.py smooths across
    # the wrong boundaries and silently averages two patients together.

    assert not np.isnan(X).any() and not np.isinf(X).any(), "NaN/Inf in X"
    # A single one turns the loss into NaN and destroys the whole network.

    assert set(np.unique(y)) <= {0, 1}, f"labels outside 0/1: {np.unique(y)}"
    # Only "A" and "N" minutes were kept. A third value would mean an annotation
    # symbol slipped through PART 4. Labels are floats, but 0.0 == 0.

    assert np.issubdtype(y.dtype, np.floating), \
        f"labels must be floating, got {y.dtype}"
    # Protects a custom focal loss: see LABEL_DTYPE. An integer label array only
    # fails once training has started, minutes in, from deep inside autograph.

    assert 0.25 <= y.mean() <= 0.55, f"apnea fraction {y.mean():.3f} out of range"
    # The share of apnea must stay near the expected 38%. Actual values are
    # train 0.401, val 0.318, test 0.380. A value far outside would signal a
    # split gone wrong or annotations misread.

    if ZSCORE_PER_WINDOW:
        assert abs(float(X.mean())) < 0.05, \
            f"z-scored windows not centred (mean {X.mean():.3f})"
        assert 0.5 <= float(X.std()) <= 2.0, \
            f"z-scored windows badly scaled (std {X.std():.3f})"
        # Each window was centred and divided by its own statistics, so the
        # pooled mean must sit on zero and the pooled spread near 1. Drift means
        # the normalisation was skipped or applied along the wrong axis. This is
        # the one check that catches a normalisation that silently did nothing.
    else:
        assert np.abs(X).max() <= CLIP_MV + EPS, "raw ECG outside the +-5 mV clip"


# ────────────────────────────────────────────────────────────────
# PART 7 : ONE SPLIT -> ITS THREE FILES ON DISK
# ────────────────────────────────────────────────────────────────


def save_split(name: str, records: list[str]) -> None:
    """Process, check and write one split.

    NOTHING IS SHUFFLED HERE. The training split used to be permuted at save
    time with a fixed seed, for a Keras `validation_split=0.2` that takes the
    LAST 20% of the array as-is: unshuffled, records concatenate a -> b -> c and
    that tail would have been almost entirely near-normal c records, making
    val_auc meaningless.

    Neither half of that is true any more. Validation is its own file and its own
    records, passed to Keras as explicit validation_data. And model.py's
    train_dataset() calls ds.shuffle() with a buffer the size of the whole
    training set and reshuffle_each_iteration=True, so the rows are fully
    permuted before every epoch regardless of the order they arrived in. The
    save-time shuffle was doing nothing.
    """

    print(f"\n[{name}] {len(records)} records")
    X, y, counts = process_split(records)

    validate(X, y, counts)
    # Checks BEFORE saving: nothing invalid ever reaches the disk.

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.save(PROCESSED_DIR / f"X_{name}{SUFFIX}.npy", X)
    np.save(PROCESSED_DIR / f"y_{name}{SUFFIX}.npy", y)
    np.save(PROCESSED_DIR / f"counts_{name}{SUFFIX}.npy", counts)

    print(f"  -> X_{name}{SUFFIX}.npy {X.shape}  apnea={y.mean():.3f}  "
          f"{X.nbytes / 1e6:.0f} MB")
    print(f"  -> y_{name}{SUFFIX}.npy {y.shape}  "
          f"+ counts_{name}{SUFFIX}.npy ({len(counts)} records)")
    # Size is printed because raw ECG is bulky -- roughly 400 MB for the test
    # split, against 13 MB for the same minutes as an RR tachogram.


# ────────────────────────────────────────────────────────────────
# PART 8 : MAIN ENTRY POINT
# ────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"window = {SEQ_LEN} samples = {SEQ_LEN // FS} s at {FS} Hz")

    seen: set[str] = set()
    for name, records in SPLIT_RECORDS.items():
        overlap = seen & set(records)
        assert not overlap, f"record in more than one split: {sorted(overlap)}"
        seen |= set(records)
        save_split(name, records)
    # The overlap assertion is the whole methodological point of this script,
    # enforced across all three splits before anything is written. One patient,
    # one split.

    print(f"\n{len(seen)} records across {len(SPLIT_RECORDS)} disjoint splits")


if __name__ == "__main__":
    main()
    # Runs main() only when the file is launched directly. If another script
    # imports this one, nothing runs -- the functions simply become available.
