from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Selected before pyplot is imported. The container has no display, and the
# default interactive backend fails on import rather than at draw time.

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

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
CHANCE_C = "#b6b5b1"     # grey, for the ROC diagonal

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


def _frame(ax, title, subtitle, xlabel, ylabel):
    """The chrome every panel shares: title, qualifying line, grid."""
    ax.set_title(title, pad=20)
    ax.annotate(subtitle, xy=(0, 1.02), xycoords="axes fraction", ha="left",
                va="bottom", fontsize=8, color=INK_2)
    # The qualifying numbers sit under the title rather than inside the
    # plot: an annotation placed in data coordinates lands somewhere
    # different on every run, and eventually on top of a line.

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.set_axisbelow(True)


def _curve(ax, epochs, series, title, subtitle, ylabel, best_epoch,
           drawstyle="default"):
    """One epoch panel: the named series, the restored-epoch rule, the chrome."""
    for label, values, color in series:
        if values is None:
            continue
        ax.plot(epochs, values, color=color, linewidth=1.8, label=label,
                drawstyle=drawstyle, solid_capstyle="round")

    ax.axvline(best_epoch, color=INK_2, linewidth=1, linestyle=(0, (4, 3)),
               zorder=1)

    _frame(ax, title, subtitle, "Epoch", ylabel)
    ax.set_xlim(epochs[0], epochs[-1])
    if len(series) > 1:
        ax.legend(loc="best")


def _roc(ax, y_true, y_prob):
    """The restored model's ROC on the validation records.

    Not an epoch panel: one curve, drawn from the predictions of the
    weights EarlyStopping restored, so it describes the model that was
    saved rather than any epoch along the way.

    The AUC printed here is sklearn's exact trapezoid over every distinct
    threshold. Keras' val_auc in the panel above is a 200-bin Riemann
    approximation of the same quantity, so the two agree to about three
    decimals and are not expected to match exactly -- a small gap is the
    two estimators, not two different models.
    """
    if y_true is None or y_prob is None:
        _frame(ax, "Validation ROC",
               "not drawn, plot_history() was called without val_roc",
               "False positive rate", "True positive rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return
        # Same tolerance the epoch panels give a missing history key: the
        # figure loses one panel instead of raising. This is the path taken
        # when the curves are redrawn from a saved history.json alone,
        # which carries no predictions.

    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    ax.plot([0, 1], [0, 1], color=CHANCE_C, linewidth=1,
            linestyle=(0, (4, 3)), zorder=1)
    # The chance diagonal: what a coin flip traces. The whole point of the
    # panel is the distance between it and the curve.

    ax.plot(fpr, tpr, color=VAL_C, linewidth=1.8, solid_capstyle="round",
            label=f"AUC {auc:.4f}", zorder=3)

    _frame(ax, "Validation ROC", "",
           "False positive rate", "True positive rate")
    # No operating point is marked. Keras' fixed 0.5 sits well down the
    # curve and test.py sweeps for its own near 0.2, but marking either
    # means naming a threshold the figure does not choose.
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    # Square by construction: a stretched ROC misreads, because the eye
    # judges it by how far the curve bulges toward the top-left corner.
    ax.legend(loc="lower right")


def plot_history(history: dict, val_roc: tuple | None = None,
                 path: Path = CURVES_PATH) -> Path:
    """Four panels answering four different questions about the run.

    loss        Is it overfitting? A validation curve that turns upward
                while the training curve keeps falling is the classic
                picture, and the cue to raise DROPOUT_TRUNK or add
                weight decay.

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

    ROC         Every operating point the saved model can be run at, not
                just the one Keras happened to report. The AUC panel gives
                a single number per epoch; this gives the whole trade-off
                behind the final number, which is what test.py sweeps when
                it picks a threshold. A curve that hugs the top-left has
                room to buy sensitivity cheaply; one that runs close to the
                diagonal has none, and no threshold will rescue it.

    The dashed rule in the three EPOCH panels is the epoch EarlyStopping
    restored the weights to. Everything to the right of it was trained and
    then thrown away. The ROC panel has no epoch axis: it is the restored
    model itself, and its dashed line is the chance diagonal instead.

    `val_roc` is (y_true, y_prob) for the validation records, predicted
    with the restored weights -- model.py passes it after fit() returns.
    Without it the ROC panel is drawn empty and labelled, which is what
    happens when the figure is redrawn from a saved history.json alone.

    The learning-rate panel this replaced is gone from the figure, not from
    the run: LearningRateLog still writes "lr" into history.json, so the
    schedule is still on disk for anyone who wants to look at it.

    Every series is read with .get, so a history missing a key (an older
    run, a renamed metric) loses one line instead of raising.
    """
    epochs = np.arange(1, len(history["loss"]) + 1)
    val_auc = history.get("val_auc")
    best = int(np.argmax(val_auc)) + 1 if val_auc else int(epochs[-1])

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
               "AUC, the monitored metric",
               f"{auc_note} · weights restored to that epoch", "AUC", best)

        _curve(axes[1, 0], epochs,
               [("Precision", history.get("val_precision"), PREC_C),
                ("Recall", history.get("val_recall"), REC_C)],
               "Validation precision & recall",
               "at Keras' fixed 0.5, not the threshold test.py picks",
               "Rate", best)

        _roc(axes[1, 1], *(val_roc if val_roc else (None, None)))

        fig.suptitle("Training history", x=0.02, ha="left", fontsize=12,
                     color=INK)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(path)
        plt.close(fig)

    return path
