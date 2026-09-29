"""
Apnea inference: raw single-lead ECG -> one apnea decision per minute.

    raw ECG (mV) -> resample to 100 Hz -> band-pass + notch -> clip
      -> 60 s windows, each z-scored -> int8 -> model_int8.tflite -> P(apnea)
      -> 5-minute moving average -> threshold -> A / N

The preprocessing is a copy of the one the model was trained on (the Dataset cell of
src/model.ipynb). It must stay identical: a model scores the windows it learned from, and any
difference in filtering or normalisation is a silent loss of accuracy, not an error.

This module has no web code, so it can be imported and tested on its own.
"""
from __future__ import annotations

import threading
from fractions import Fraction
from pathlib import Path

import numpy as np
import wfdb
from ai_edge_litert.interpreter import Interpreter, OpResolverType
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

FS = 100                        # the model's sampling rate
WINDOW_SAMPLES = 60 * FS        # one minute = one expert label = one inference
EPS = 1e-6
CLIP_MV = 5.0


# ── preprocessing: identical to training ─────────────────────────────────────

def clean_ecg(x: np.ndarray) -> np.ndarray:
    """Band-pass 0.5-40 Hz and 50 Hz notch, both zero-phase, then clip to +-5 mV."""
    sos = butter(4, [0.5, 40], btype="band", fs=FS, output="sos")
    x = sosfiltfilt(sos, x)
    b, a = iirnotch(50, Q=30, fs=FS)
    x = filtfilt(b, a, x)
    x = np.nan_to_num(x, nan=np.nanmedian(x))
    return np.clip(x, -CLIP_MV, CLIP_MV)
    # Zero-phase filtering (forward then backward) needs the whole recording at once. That is
    # fine here, where a complete night is analysed after the fact. A device scoring minutes as
    # they arrive cannot do it, and would need a causal filter and a model retrained on it.


def to_windows(ecg_clean: np.ndarray) -> np.ndarray:
    """Whole minutes from the start of the recording -> (N, 6000) float32, each z-scored."""
    n = len(ecg_clean) // WINDOW_SAMPLES
    w = ecg_clean[:n * WINDOW_SAMPLES].astype(np.float32).reshape(n, WINDOW_SAMPLES)
    w = (w - w.mean(axis=1, keepdims=True)) / (w.std(axis=1, keepdims=True) + EPS)
    return np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    # Minutes start at sample 0, 6000, 12000... which is also where Apnea-ECG's annotations
    # fall, and a trailing partial minute is dropped, as in training. Each window is z-scored
    # on its own: absolute amplitude reflects electrode placement, not breathing.


def resample_to_model_rate(ecg: np.ndarray, fs: float) -> np.ndarray:
    if fs == FS:
        return ecg
    ratio = Fraction(FS / fs).limit_denominator(1000)
    return resample_poly(ecg, ratio.numerator, ratio.denominator)
    # Polyphase resampling applies its own anti-aliasing low-pass, so a 250 or 500 Hz
    # recording arrives at 100 Hz without frequencies folding back into the ECG band.


def smooth(prob: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average over `window` minutes, edges padded with the edge value."""
    if window <= 1 or len(prob) == 0:
        return prob
    padded = np.pad(prob, window // 2, mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")
    # Apnea comes in runs of 5 to 30 minutes, but the model sees each minute alone. Averaging
    # neighbours removes isolated spikes. One call per recording, so two nights never mix.


# ── the model ────────────────────────────────────────────────────────────────

class ApneaModel:
    """The int8 TFLite model: z-scored windows in, P(apnea) per window out."""

    def __init__(self, model_path: Path):
        self.path = Path(model_path)
        self._interpreter = Interpreter(
            model_path=str(self.path),
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        # Plain built-in kernels. The default XNNPACK delegate cannot prepare this int8 graph,
        # and the reference kernels are the closest match to TFLite Micro on the ESP32.
        self._interpreter.allocate_tensors()
        self._lock = threading.Lock()
        # One interpreter holds one set of tensors, so two requests must not run it at once.

        inp = self._interpreter.get_input_details()[0]
        out = self._interpreter.get_output_details()[0]
        assert tuple(inp["shape"]) == (1, WINDOW_SAMPLES, 1), f"unexpected input {inp['shape']}"
        assert inp["dtype"] == np.int8 and out["dtype"] == np.int8, "expected a full-int8 model"
        self._in, self._out = inp["index"], out["index"]
        self.in_scale, self.in_zero = inp["quantization"]
        self.out_scale, self.out_zero = out["quantization"]

    def predict(self, windows: np.ndarray) -> np.ndarray:
        q = np.clip(np.round(windows / self.in_scale) + self.in_zero, -128, 127).astype(np.int8)
        probs = np.empty(len(windows), dtype=np.float32)
        with self._lock:
            for i, window in enumerate(q):
                self._interpreter.set_tensor(self._in, window.reshape(1, WINDOW_SAMPLES, 1))
                self._interpreter.invoke()
                raw = int(self._interpreter.get_tensor(self._out)[0, 0])
                probs[i] = (raw - self.out_zero) * self.out_scale
        return probs


# ── one recording, end to end ────────────────────────────────────────────────

def analyse(ecg_mv: np.ndarray, fs: float, model: ApneaModel,
            threshold: float, smooth_window: int) -> dict:
    ecg = resample_to_model_rate(np.asarray(ecg_mv, dtype=np.float64), fs)
    windows = to_windows(clean_ecg(ecg))
    if len(windows) == 0:
        raise ValueError(f"recording shorter than one minute ({len(ecg) / FS:.0f} s at {FS} Hz)")

    prob = model.predict(windows)
    prob_smoothed = smooth(prob, smooth_window)
    apnea = prob_smoothed > threshold

    hours = len(windows) / 60
    return {
        "minutes": len(windows),
        "threshold": threshold,
        "smooth_window": smooth_window,
        "summary": {
            "apnea_minutes": int(apnea.sum()),
            "apnea_fraction": float(apnea.mean()),
            "apnea_minutes_per_hour": float(apnea.sum() / hours),
        },
        "prob": [round(float(p), 4) for p in prob],
        "prob_smoothed": [round(float(p), 4) for p in prob_smoothed],
        "apnea": [bool(a) for a in apnea],
    }
    # "Apnea minutes per hour" is NOT the clinical AHI, which counts individual events of 10 s
    # or more. The model labels whole minutes, so this is its own index and is named as such.


def has_ecg(data_dir: Path, name: str) -> bool:
    return "ECG" in wfdb.rdheader(str(Path(data_dir) / name)).sig_name


def load_physionet_record(data_dir: Path, name: str) -> tuple[np.ndarray, float, list[str] | None]:
    """A PhysioNet Apnea-ECG record -> (ECG in mV, fs, expert labels per minute or None)."""
    path = str(Path(data_dir) / name)
    signals, fields = wfdb.rdsamp(path, channel_names=["ECG"])
    # The ECG channel by NAME, not position: Apnea-ECG's "r" records (a01r...) hold
    # respiration and SpO2 channels, and their first channel is not an ECG at all.
    labels = None
    if (Path(data_dir) / f"{name}.apn").exists():
        ann = wfdb.rdann(path, "apn")
        labels = [str(s) for s in ann.symbol]
    return signals[:, 0], float(fields["fs"]), labels
