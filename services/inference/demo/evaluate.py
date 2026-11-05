"""
Scores the two CinC 2018 models on the demo patients of `make demo` (Apnea-ECG's x01 to x35)
against the expert's minute labels, prints the results and draws them in a PNG.

Run by `make eval`, inside the running inference container, where data/apnea-ecg is mounted at
/apnea-ecg and results/ at /results. For each model and each night of the three demo patients, it
takes the service's stored analysis of that night with that model, or runs a new one when there
is none, as Analyse in the interface does. Every minute the expert scored is then compared with
the model's: the expert labels are the records' .apn files, downloaded into /apnea-ecg when
missing.
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

import numpy as np
from scipy.stats import rankdata

import apnea_ecg as demo

MODELS = ("model_cinc2018_float32.tflite", "model_cinc2018_int8.tflite")
FIGURE = Path("/results/evaluate.png")
SAMPLES_PER_MINUTE = 6000       # Apnea-ECG is sampled at 100 Hz
SKIP, NUM, SUB, CHN, AUX = 59, 60, 61, 62, 63
APNEA, NORMAL = 8, 1            # the codes of the "A" and "N" annotations
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
# matplotlib writes its font cache to MPLCONFIGDIR; the container's user has no home folder.


# ── expert labels ────────────────────────────────────────────────────────────

def apn_file(record: str) -> bytes:
    path = demo.CACHE / f"{record}.apn"
    if not path.exists():
        data = demo.fetch(path.name)
        part = path.with_name(path.name + ".part")
        part.write_bytes(data)
        part.replace(path)
    return path.read_bytes()
    # PhysioNet's SHA256SUMS.txt has no entry for x01.apn to x35.apn, so they are not checked.


def expert_labels(record: str) -> dict[int, int]:
    """Minute -> 1 apnea, 0 none, for every minute the expert scored."""
    words = np.frombuffer(apn_file(record), dtype="<u2").astype(np.int64)
    labels, sample, i = {}, 0, 0
    while i < len(words):
        code, value = words[i] >> 10, words[i] & 0x3FF
        if code == 0 and value == 0:
            break
        if code == SKIP:
            interval = (words[i + 1] << 16) | words[i + 2]
            sample += interval - (1 << 32) if interval >= 1 << 31 else interval
            i += 3
        elif code in (NUM, SUB, CHN):
            i += 1
        elif code == AUX:
            i += 1 + (value + 1) // 2
        else:
            sample += value
            if code in (APNEA, NORMAL):
                labels[sample // SAMPLES_PER_MINUTE] = int(code == APNEA)
            i += 1
    return labels
    # WFDB annotation file (MIT format): 16-bit little-endian words, each a 6-bit code and a 10-bit
    # value. An annotation's value is the samples since the previous one. SKIP adds a longer gap,
    # given as a 32-bit signed integer in the next two words, high half first. NUM, SUB and CHN
    # set fields of the previous annotation; AUX is followed by `value` bytes of text, padded to
    # whole words. Two zero bytes end the file.


# ── the service's analyses ───────────────────────────────────────────────────

def analysis(patient: str, record: str, model: str) -> tuple[int, object, bool]:
    path = f"/api/patients/{patient}/nights/{record}.csv/analysis?model={quote(model)}"
    status, result = demo.call("GET", path)
    if status != 404:
        return status, result, False
    status, result = demo.call("POST", path)
    return status, result, True
    # GET returns the newest stored analysis with this exact model file, threshold and smoothing,
    # or 404 when there is none; POST then runs the model and stores the analysis.


def auc(y: np.ndarray, p: np.ndarray) -> float:
    ranks = rankdata(p)
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    return float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    # The share of (apnea minute, normal minute) pairs in which the apnea minute has the higher
    # probability, ties counting half: the Mann-Whitney statistic.


def counts(y: np.ndarray, a: np.ndarray) -> dict[str, int]:
    return {"TN": int(((y == 0) & (a == 0)).sum()), "FP": int(((y == 0) & (a == 1)).sum()),
            "FN": int(((y == 1) & (a == 0)).sum()), "TP": int(((y == 1) & (a == 1)).sum())}
    # TN: normal minute called normal. FP: normal minute called apnea.
    # FN: apnea minute called normal. TP: apnea minute called apnea.


# ── figure ───────────────────────────────────────────────────────────────────

def draw(scores: dict[str, dict], title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    teal = LinearSegmentedColormap.from_list(
        "teal", ["#e6f4f4", "#c3e4e5", "#97cfd1", "#66b5ba", "#3d9aa1", "#1f7f88", "#14646e", "#0c4a52"])
    ink, ink_2, muted, surface = "#1f2933", "#52606d", "#9aa5b1", "#ffffff"
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 11})

    fig, axes = plt.subplots(1, len(scores), figsize=(10.5, 6.2), facecolor=surface)
    for ax, (model, s) in zip(np.atleast_1d(axes), scores.items()):
        tags = np.array([["TN", "FP"], ["FN", "TP"]])
        grid = np.array([[s["TN"], s["FP"]], [s["FN"], s["TP"]]])
        share = grid / grid.sum(axis=1, keepdims=True)
        ax.set_facecolor(surface)
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle((j + 0.015, i + 0.015), 0.97, 0.97,
                                           color=teal(share[i, j]), linewidth=0))
                text = "#ffffff" if share[i, j] > 0.5 else ink
                ax.text(j + 0.5, i + 0.24, tags[i, j], ha="center", va="center", color=text,
                        fontsize=12, fontweight="bold")
                ax.text(j + 0.5, i + 0.48, f"{grid[i, j]:,}", ha="center", va="center",
                        color=text, fontsize=22, fontweight="bold")
                ax.text(j + 0.5, i + 0.72, f"{share[i, j]:.0%} of the row", ha="center",
                        va="center", color=text, fontsize=10.5)
        ax.set_xlim(0, 2)
        ax.set_ylim(2, 0)
        ax.set_aspect("equal")
        ax.set_xticks([0.5, 1.5], ["no apnea", "apnea"], color=ink_2)
        ax.set_yticks([0.5, 1.5], ["no apnea", "apnea"], color=ink_2, rotation=90, va="center")
        ax.set_xlabel("Model", color=ink_2, labelpad=8)
        ax.set_ylabel("Expert", color=ink_2, labelpad=8)
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.set_title(f"{model}, threshold {s['threshold']:.2f}", color=ink, fontsize=12,
                     fontweight="bold", pad=46)
        ax.text(1, 2.12, f"accuracy {s['accuracy']:.1%}  ·  AUC {s['auc']:.3f}\nsensitivity "
                f"{s['sensitivity']:.2f}  ·  specificity {s['specificity']:.2f}", ha="center",
                va="top", color=ink_2, fontsize=11, linespacing=1.5)
        # Each cell is shaded by its share of the expert's row; text on dark cells is white.

    fig.suptitle(title, color=ink, fontsize=14, fontweight="bold", x=0.5, y=0.98)
    # fig.text(0.5, 0.02, "TN normal minute called normal · FP normal called apnea · FN apnea called "
    #          "normal · TP apnea called apnea. 5-minute smoothing.", ha="center", color=muted,
    #          fontsize=9.5)
    fig.subplots_adjust(left=0.07, right=0.97, top=0.78, bottom=0.19, wspace=0.25)
    fig.savefig(FIGURE, dpi=200, facecolor=surface)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    demo.wait_for_service()
    thresholds = {m["name"]: m["threshold"] for m in demo.call("GET", "/api/models")[1]["models"]}
    for model in MODELS:
        if thresholds.get(model) is None:
            sys.exit(f"{model} is not in services/inference/data/models or has no threshold in "
                     f"MODEL_THRESHOLDS (.env).")
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        sys.exit("The service's image has no matplotlib: run make rebuild first.")

    ids = {}
    for patient in demo.call("GET", "/api/patients")[1]:
        ids.setdefault(patient["name"], patient["id"])
    nights = []
    for name, records in demo.PATIENTS.items():
        if name not in ids:
            sys.exit(f"There is no patient named {name!r}: run make demo first.")
        stored = {night["file"] for night in demo.call("GET", f"/api/patients/{ids[name]}")[1]["nights"]}
        nights += [(name, record) for record in records if f"{record}.csv" in stored]
        absent = [record for record in records if f"{record}.csv" not in stored]
        if absent:
            print(f"{name} has no {', '.join(absent)}: run make demo to add them", flush=True)
    # (patient name, record) for every demo night the service has.

    try:
        labels = {record: expert_labels(record) for _, record in nights}
    except OSError as error:
        sys.exit(f"could not download the expert labels: {error}")

    print(f"Analysing {len(nights)} nights with {len(MODELS)} models; stored analyses are reused",
          flush=True)
    tasks = [(name, record, model) for name, record in nights for model in MODELS]
    with ThreadPoolExecutor(max_workers=len(MODELS)) as pool:
        answers = list(pool.map(lambda t: analysis(ids[t[0]], t[1], t[2]), tasks))
    results, new = {}, 0
    for (name, record, model), (status, result, ran) in zip(tasks, answers):
        if status != 200:
            sys.exit(f"{record}.csv with {model}: {status} {result}")
        results[(model, record)] = result
        new += ran
    print(f"{new} new analyses, {len(tasks) - new} reused", flush=True)
    # Nights go in the outer loop, so the analyses running side by side use different models:
    # the service runs one analysis at a time per model.

    scored = {}
    for model in MODELS:
        y, p, a, group = [], [], [], []
        for name, record in nights:
            r = results[(model, record)]
            for minute, label in labels[record].items():
                if minute < r["minutes"]:
                    y.append(label)
                    p.append(r["prob_smoothed"][minute])
                    a.append(int(r["apnea"][minute]))
                    group.append(name)
        scored[model] = tuple(np.array(v) for v in (y, p, a, group))
    # Minutes past the end of the analysed night (the last, incomplete minute) are left out.

    scores = {}
    for model in MODELS:
        y, p, a, _ = scored[model]
        scores[model] = {"threshold": thresholds[model], **counts(y, a),
                         "accuracy": float((a == y).mean()), "sensitivity": float(a[y == 1].mean()),
                         "specificity": float(1 - a[y == 0].mean()), "auc": auc(y, p)}

    y = scored[MODELS[0]][0]
    title = (f"Apnea-ECG x01 to x35: {len(nights)} nights, {len(y):,} minutes, "
             f"{y.mean():.1%} apnea")
    print(f"\n{title}\n")
    width = max(len(model) for model in MODELS)
    print(f"{'':{width}s}  threshold  accuracy  sensitivity  specificity    AUC"
          f"      TN      FP      FN      TP")
    for model, s in scores.items():
        print(f"{model:{width}s}  {s['threshold']:9.2f}  {s['accuracy']:8.1%}  {s['sensitivity']:11.2f}  "
              f"{s['specificity']:11.2f}  {s['auc']:.3f}  {s['TN']:6,d}  {s['FP']:6,d}  {s['FN']:6,d}  "
              f"{s['TP']:6,d}")
    # print("\nTN normal minute called normal, FP normal called apnea, FN apnea called normal, "
    #       "TP apnea called apnea")

    groups = list(demo.PATIENTS)
    print(f"\n{'Accuracy per patient':{width}s}" + "".join(
        f"{g.removeprefix('Apnea-ECG '):>12s}" for g in groups))
    for model in MODELS:
        y, _, a, group = scored[model]
        print(f"{model:{width}s}" + "".join(
            f"{(a[group == g] == y[group == g]).mean():12.1%}" for g in groups))

    draw(scores, title)
    print(f"\nsaved results/{FIGURE.name}")


if __name__ == "__main__":
    main()
