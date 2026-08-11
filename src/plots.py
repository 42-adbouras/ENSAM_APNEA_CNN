from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Selected before pyplot is imported. The container has no display, and the
# default interactive backend fails on import rather than at draw time.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from utils import RESULTS_ROOT  # noqa: E402

CURVES_PATH = RESULTS_ROOT / "training_curves.png"

# Fixed colour slots. Blue means "train" and orange means "validation" in
# every panel that has both, so the eye learns the pairing once. The two
# extra hues appear only where the series are two metrics rather than two
# splits, precisely so they cannot be mistaken for the train/val pair.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e8e7e3"
AXIS = "#d5d4d0"
TRAIN_C = "#2a78d6"      # blue
VAL_C = "#eb6834"        # orange
PREC_C = "#1baf7a"       # aqua
REC_C = "#eda100"        # yellow

PLOT_STYLE = {
    "figure.facecolor": SURFACE,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.fontsize": 8,
}


def _curve(ax, epochs, series, title, subtitle, ylabel, best_epoch,
           drawstyle="default"):
    """One panel: the named series, the restored-epoch rule, the chrome."""
    for label, values, color in series:
        if values is None:
            continue
        ax.plot(epochs, values, color=color, linewidth=1.8, label=label,
                drawstyle=drawstyle, solid_capstyle="round")

    ax.axvline(best_epoch, color=INK_2, linewidth=1, linestyle=(0, (4, 3)),
               zorder=1)

    ax.set_title(title, pad=20)
    ax.annotate(subtitle, xy=(0, 1.02), xycoords="axes fraction", ha="left",
                va="bottom", fontsize=8, color=INK_2)
    # The qualifying numbers sit under the title rather than inside the
    # plot: an annotation placed in data coordinates lands somewhere
    # different on every run, and eventually on top of a line.

    ax.set_ylabel(ylabel)
    ax.set_xlabel("Epoch")
    ax.set_xlim(epochs[0], epochs[-1])
    ax.grid(True)
    ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(loc="best")


def plot_history(history: dict, path: Path = CURVES_PATH) -> Path:
    """Four panels answering four different questions about the run.

    loss        Is it overfitting? A validation curve that turns upward
                while the training curve keeps falling is the classic
                picture, and the cue to raise DROPOUT_TRUNK or set
                AUGMENT = True.

    AUC         The only curve with mechanical consequences: both
                EarlyStopping and ReduceLROnPlateau monitor val_auc, and
                the weights on disk are whichever epoch maximised it. A
                jagged curve with a spiky maximum means those weights were
                a lucky epoch rather than a converged one -- which is
                exactly how three seeds end up scoring 0.83 / 0.83 / 0.91.

    prec/rec    Both are computed at Keras' fixed 0.5, while test.py goes
                on to choose a threshold near 0.2. The distance between
                the two is a direct reading of how far the sigmoid is
                from being calibrated.

    lr          Whether each plateau in the AUC panel was actually broken
                by a rate cut. If the curve above never recovers after the
                first reduction, REDUCE_LR_PATIENCE and EPOCHS are simply
                buying wasted epochs.

    The dashed rule in all four panels is the epoch EarlyStopping restored
    the weights to. Everything to the right of it was trained and then
    thrown away.

    Every series is read with .get, so a history missing a key (an older
    run, a renamed metric) loses one line instead of raising.
    """
    epochs = np.arange(1, len(history["loss"]) + 1)
    val_auc = history.get("val_auc")
    best = int(np.argmax(val_auc)) + 1 if val_auc else int(epochs[-1])
    lr = history.get("lr")

    with plt.rc_context(PLOT_STYLE):
        # rc_context rather than rcParams.update: a global style change
        # would follow into anything else drawing in the same process.
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7))

        _curve(axes[0, 0], epochs,
               [("Train", history.get("loss"), TRAIN_C),
                ("Validation", history.get("val_loss"), VAL_C)],
               "Loss", f"binary cross-entropy · {len(epochs)} epochs run",
               "Loss", best)

        auc_note = (f"best {max(val_auc):.4f} at epoch {best}"
                    if val_auc else "val_auc not recorded")
        _curve(axes[0, 1], epochs,
               [("Train", history.get("auc"), TRAIN_C),
                ("Validation", val_auc, VAL_C)],
               "AUC — the monitored metric",
               f"{auc_note} · weights restored to that epoch", "AUC", best)

        _curve(axes[1, 0], epochs,
               [("Precision", history.get("val_precision"), PREC_C),
                ("Recall", history.get("val_recall"), REC_C)],
               "Validation precision & recall",
               "at Keras' fixed 0.5 — not the threshold test.py picks",
               "Rate", best)

        _curve(axes[1, 1], epochs,
               [("Learning rate", lr, INK_2)],
               "Learning rate",
               (f"{lr[0]:.1e} → {lr[-1]:.1e}" if lr
                else "not recorded — is LearningRateLog in the callbacks?"),
               "Rate", best, drawstyle="steps-post")
        # steps-post, not a line: the rate is piecewise constant and a cut
        # lands between two epochs. Interpolating draws a ramp that never
        # happened.
        if lr:
            axes[1, 1].set_yscale("log")
            # Linear, a halving schedule reads as one drop and then a flat
            # line; log spaces the cuts evenly, which is what they are.

        fig.suptitle("Training history", x=0.02, ha="left", fontsize=12,
                     color=INK)
        fig.text(0.02, -0.01,
                 f"Dashed rule: epoch {best}, the weights EarlyStopping "
                 f"restored and saved. Later epochs were discarded.",
                 ha="left", fontsize=8, color=INK_2)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(path)
        plt.close(fig)

    return path
