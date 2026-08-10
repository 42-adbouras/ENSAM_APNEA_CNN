from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from model import MODEL_PATH
from utils import RESULTS_ROOT, load_split

THRESHOLD_CRITERION = "accuracy"   # "accuracy" | "youden" | "f1"
THRESHOLD_GRID = np.arange(0.05, 0.96, 0.01)

def load_model(path: Path):
    try:
        return tf.keras.models.load_model(str(path))
    except Exception:
        return tf.keras.models.load_model(str(path), compile=False)



def rates(y_true, y_prob, t) -> dict:
    y_pred = (y_prob > t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    return {"accuracy": (tp + tn) / len(y_true),
            "youden": sens + spec - 1.0,
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "sensitivity": sens,
            "specificity": spec}


def choose_threshold(y_true, y_prob, criterion: str = THRESHOLD_CRITERION):
    best_t, best_v, best_stats = 0.5, -np.inf, {}
    for t in THRESHOLD_GRID:
        s = rates(y_true, y_prob, t)
        if s[criterion] > best_v:
            best_t, best_v, best_stats = float(t), s[criterion], s
    return best_t, best_stats


def evaluate(y_true, y_prob, threshold) -> dict:
    y_pred = (y_prob > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float((tp + tn) / len(y_true)),
        "recall_apnea": float(tp / (tp + fn)) if tp + fn else 0.0,
        "recall_normal": float(tn / (tn + fp)) if tn + fp else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }




def print_results(metrics: dict, split: str) -> None:
    m = metrics
    print(f"\n{'='*55}")
    print(f"  RESULTS - {split.upper()} (threshold {m['threshold']:.2f})")
    print(f"{'='*55}")
    print(f"  AUC          : {m['auc']:.3f}")
    print(f"  Recall Apnea : {m['recall_apnea']:.3f}")
    print(f"  Recall Normal: {m['recall_normal']:.3f}")
    print(f"  Precision    : {m['precision']:.3f}")
    print(f"  F1           : {m['f1']:.3f}")
    print(f"  Accuracy     : {m['accuracy']*100:.1f}%")
    print(f"  FN           : {m['fn']}")
    print(f"  FP           : {m['fp']}")
    print("\n  Confusion matrix :")
    print("                  Pred Normal    Pred Apnea")
    print(f"  True Normal  :      {m['tn']:5d}         {m['fp']:5d}")
    print(f"  True Apnea   :      {m['fn']:5d}         {m['tp']:5d}")



def main(argv: list[str]) -> None:
    args = [a for a in argv if not a.startswith("--")]
    split = "test"
    if "--split" in argv:
        i = argv.index("--split")
        split = argv[i + 1] if i + 1 < len(argv) else "test"
        args = [a for a in args if a != split]

    model_path = Path(args[0]) if args else MODEL_PATH
    if not model_path.is_absolute() and not model_path.exists():
        model_path = RESULTS_ROOT / model_path
    if not model_path.exists():
        raise SystemExit(f"no such model: {model_path}\nRun:  make train")

    model = load_model(model_path)

    val = load_split("val")
    prob_val = model.predict(val["X"][..., np.newaxis], verbose=1).ravel()
    threshold, _stats = choose_threshold(val["y"], prob_val)

    target = val if split == "val" else load_split(split)
    prob = prob_val if split == "val" else model.predict(
        target["X"][..., np.newaxis], verbose=0).ravel()

    print_results(evaluate(target["y"], prob, threshold), split)


if __name__ == "__main__":
    main(sys.argv[1:])
