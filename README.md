# ENSAM_APNEA_CNN

Sleep apnea detection from a single-lead ECG using a 1D convolutional neural network.

The whole pipeline, from downloading the data to scoring the int8 model, is one notebook:
[src/model.ipynb](src/model.ipynb).

---

## Overview

| | |
|---|---|
| **Dataset** | [Apnea-ECG](https://physionet.org/content/apnea-ecg/1.0.0/) (PhysioNet, Moody & Mark, 2000) |
| **Signal** | Single-lead ECG, 100 Hz, full-night recordings |
| **Input** | One 60 s window = 6000 raw samples, z-scored per window |
| **Label** | One expert annotation per minute: `A` (apnea) = 1, `N` (normal) = 0 |
| **Model** | Dilated residual 1D CNN: 4-layer stem, 7 residual blocks (dilations 1 to 64), 69,646 parameters |
| **Loss** | Binary cross-entropy |
| **Deployment** | Full-int8 TFLite model, 124.5 KB |
| **Framework** | TensorFlow 2.20 (Keras 3) |

---

## Results

On the 35 official test recordings (`x01` to `x35`, 17,248 minutes), which play no part in
training or in choosing the threshold:

| Model | Threshold | AUC | Accuracy | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|---|
| Keras (`model.keras`) | 0.20 | 0.9209 | 85.56 % | 0.795 | 0.893 | 0.819 |
| int8 (`model_int8.tflite`) | 0.23 | 0.9198 | 85.70 % | 0.802 | 0.891 | 0.818 |

Quantisation to int8 makes the model about 4 times smaller at no measurable cost in accuracy.

Training ran 21 epochs; early stopping restored epoch 9 (best validation AUC 0.9105).

---

## Data split

The split is **by record, never by minute**: one patient appears in one set only.

| Split | Records | Minutes | Apnea |
|---|---|---|---|
| `train` | 27: `a01` to `a20`, `b01` to `b05`, `c01` to `c10`, minus the validation records | 13,158 | 40.1 % |
| `val` | 8: `a05 a10 a15 a16 a17 b04 c04 c08` | 3,865 | 31.8 % |
| `test` | 35: `x01` to `x35`, the official PhysioNet test set | 17,248 | 38.0 % |

The validation records cover all three groups of the dataset (clear apnea `a`, borderline `b`,
near-normal `c`), so validation contains the same mix as the data it stands in for.

---

## Repository layout

```
ENSAM_APNEA_CNN/
├── src/
│   └── model.ipynb          # the whole pipeline: data, model, training, conversion, test
├── data/                    # created by the notebook, git-ignored
│   ├── apnea-ecg/           # raw PhysioNet records (.hea, .dat, .apn, .qrs), ~420 MB
│   └── processed/           # X_ and y_ arrays for train, val and test, ~823 MB
└── results/
    ├── model.keras          # trained Keras model
    ├── history.json         # loss and metrics for every epoch
    ├── training_curves.png  # training figure
    ├── model_int8.tflite    # int8 model for the ESP32-S3
    └── model_float32.tflite # the same model unquantised, float32 throughout
```

---

## Setup

Python **3.11** and the following packages (the versions the notebook was run with):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install tensorflow==2.20.0 wfdb==4.3.1 numpy==2.4.6 scipy==1.17.1 \
            scikit-learn==1.9.0 matplotlib==3.11.1 ipykernel
```

Then open [src/model.ipynb](src/model.ipynb), select the `.venv` kernel, and run all cells. The
notebook finds the project folder on its own, whether the kernel starts in `src/` or at the root.

A first run downloads the dataset (~420 MB from PhysioNet) and builds the processed arrays. Later
runs skip both.

---

## The notebook, section by section

### Setup

Imports, folders (`data/apnea-ecg`, `data/processed`, `results`, created if missing), output file
paths, and a fixed seed (42) for Python, NumPy and TensorFlow, so a rerun on the same machine and
library versions produces the same model.

### Dataset

Runs only what is missing:

1. **Download.** Any record without its four files (`.hea` header, `.dat` samples, `.apn` expert
   labels, `.qrs` heartbeat positions) is fetched from PhysioNet. Only the 70 ECG recordings are
   downloaded; the respiration recordings (`a01r`...) have no ECG channel and are skipped.
2. **Preprocessing**, for each record:
   - a 4th-order Butterworth band-pass (0.5 to 40 Hz) removes baseline wander and high-frequency
     noise, and a 50 Hz notch removes mains interference. Both are zero-phase (forward then
     backward), so the heartbeats do not move in time;
   - missing samples are replaced by the median, and the signal is clipped to ±5 mV;
   - each labelled minute becomes one 6000-sample row, z-scored against its own mean and standard
     deviation. This removes the patient-specific amplitude and keeps the shape of the minute.

   A minute is dropped for only two reasons: it runs past the end of the recording, or its label
   is neither `A` nor `N`. Never for signal quality.

Output in `data/processed/`:

| File | Shape |
|---|---|
| `X_train_apnea.npy` / `y_train_apnea.npy` | (13158, 6000) / (13158,) float32 |
| `X_val_apnea.npy` / `y_val_apnea.npy` | (3865, 6000) / (3865,) float32 |
| `X_test_apnea.npy` / `y_test_apnea.npy` | (17248, 6000) / (17248,) float32 |

Delete `data/processed/` to force a rebuild.

### Model

```
Input (6000, 1)
  stem1  Conv1D(16, 15, stride 2) → BN → ReLU                6000 → 3000
  stem2  Conv1D(24, 7)  → BN → ReLU → MaxPool(2)             3000 → 1500
  stem3  Conv1D(40, 7)  → BN → ReLU → MaxPool(2)             1500 →  750
  stem4  Conv1D(32, 5)  → BN → ReLU → MaxPool(2)              750 →  375

  7 × residual block, dilation 1, 2, 4, 8, 16, 32, 64        (375, 32)
        Conv1D(32, 3, dilated) → BN → ReLU → SpatialDropout(0.10)
        Conv1D(32, 3, dilated) → BN → squeeze-excite(ratio 8)
        + shortcut → ReLU

  concat[ attention pool | global avg pool | global max pool ]   → (96,)
  Dense(64, relu) → Dropout(0.30)
  Dense(1, sigmoid)                                              → P(apnea)
```

The stem shrinks the minute to 375 timesteps; the dilated blocks then widen the receptive field to
**8225 samples (82.2 s)**. `build_model()` asserts that this covers the 60 s window, so a change to
the stem or the dilations that breaks the coverage fails at build time.

### Training

Adam (learning rate 1e-3), binary cross-entropy, batch size 64, up to 70 epochs. The training set
is reshuffled every epoch. Two callbacks watch the validation AUC:

- `EarlyStopping(patience=12, restore_best_weights=True)`: stops after 12 epochs without
  improvement and keeps the weights of the best epoch;
- `ReduceLROnPlateau(factor=0.5, patience=6)`: halves the learning rate after 6 epochs without
  improvement.

Saved to `results/`: `model.keras`, `history.json`, and `training_curves.png` (loss, AUC,
validation precision and recall, validation ROC curve). Training again overwrites these files.

### TensorFlow Lite conversion

Converts `model.keras` into two TFLite files, both with a **fixed input shape** `(1, 6000, 1)`
(a microcontroller allocates all its memory up front):

- `model_int8.tflite` (124.5 KB), **full int8**: weights, activations, input and output. The
  int8 ranges are calibrated on 500 real training windows;
- `model_float32.tflite` (295.8 KB), **unquantised**: float32 throughout, so it computes what
  `model.keras` computes. It takes the z-scored window and returns the probability directly.

### Test set

Scores `model.keras` and `model_int8.tflite` on the 35 test records. For each model:

1. predict on the validation records and **smooth** the probabilities with a 5-minute moving
   average, within each night separately (apnea comes in runs of several minutes, while the model
   scores each minute alone);
2. choose the threshold that maximises validation accuracy.
3. apply that threshold **once** to the smoothed test predictions, and report AUC, accuracy,
   sensitivity, specificity, precision and the confusion matrix.

