# Apnea review: the inference service

**Apnea review** is a small web application that runs the project's trained models on overnight
ECG recordings. You copy each patient's recordings into a folder, open the page in a browser, and
for every night you get a minute-by-minute estimate of when the patient had sleep apnea.

It runs on your own computer inside Docker. Nothing is sent to the internet.

---

## Contents

1. [Start it](#1-start-it)
2. [A tour of the interface](#2-a-tour-of-the-interface)
3. [Adding recordings](#3-adding-recordings)
4. [Models and thresholds](#4-models-and-thresholds)
5. [How a night is analysed](#5-how-a-night-is-analysed)
6. [Where results are stored](#6-where-results-are-stored)
7. [Settings](#7-settings)
8. [Commands](#8-commands)
9. [Code and API](#9-code-and-api)

---

## 1. Start it

### What you need

- **Docker Desktop** (or Docker Engine with Compose on Linux), installed and running.
- **GNU Make**.

Nothing else: Python, Node and the models are all inside the Docker image.

### First start

From the project folder, create the settings file, `.env` (only the first time):

```bash
MODEL_PATH=/models/model_int8.tflite
MODEL_THRESHOLDS=model_int8.tflite=0.23,model_float32.tflite=0.20
PATIENTS_DIR=/patients
ANALYSES_DIR=/analyses
RECORDING_FS=100
```

Each line is one setting, explained in [Settings](#7-settings). Then start the service:

```bash
make up    # builds the image (first time only) and starts the service
```

The first `make up` takes a few minutes, because it downloads and builds everything. Later starts take a few seconds.

Then open **http://localhost:8000** in your browser.

To check that it is running, `make ps` should show the container as `(healthy)`.

### Stopping it

```bash
make stop    # stop the service; make up starts it again
```

Stopping does not lose anything: recordings and results are ordinary files in the project folder.

---

## 2. A tour of the interface

The interface has three pages: **Patients**, then one **patient**, then one **night**.

### 2.1 The patients page

![The patients page](docs/screenshots/patients.png)

This is the first page. Each row is one patient, with:

- the number of nights recorded;
- the total recorded time;
- how many of those nights have been analysed **with the selected model**.

Type in **Search patients** to filter the list. The **Model** menu selects the model whose results are shown. Click a row to open the patient.

### 2.2 A patient's nights

![A patient's nights](docs/screenshots/patient.png)

At the top: the number of nights, the total recorded time and how many nights are analysed.

Below, one row per night, **newest at the top**. "Night 1" is always the oldest recording. Each row shows the file name, the night's length and its file size, then:

- **a result**, such as `43.6 min/h`, with a red bar (the bar is full at 60 minutes per hour), or
- **Not analysed**, when the selected model has not been run on that night yet.

Click a row to open the night.

**Analyse all nights** runs the model on every night that is not analysed yet, one after the other,
newest first. The button shows the progress and the row being analysed shows a spinner:

![Analyse all nights in progress](docs/screenshots/analysing.png)

Each night takes one to two seconds. Stay on the page until it finishes: leaving the page stops the run after the current night. Nights already done are kept.

### 2.3 Going to a patient from any page

On every page except the first, the header has a **Go to patient** box. Type part of a name, then click a match.

![The Go to patient box](docs/screenshots/search.png)

### 2.4 A night

When you first open a night that has not been analysed, the page looks like this:

![A night before analysis](docs/screenshots/night-not-analysed.png)

1. Check the **Model** menu (the threshold of the selected model is shown next to it).
2. Click **Analyse**. The result appears after one to two seconds.

The result is saved, so the next time anyone opens this night with this model it appears
immediately, even after a restart. Clicking **Analyse** again runs the model again and replaces the
saved result.

To move between nights, use the buttons at the top right, which name the previous and next night.
The links at the top (**Patients / patient_3 / Night 4**) take you back up.

### 2.5 Reading the result

![A night's result](docs/screenshots/night.png)

**The four figures**

| Figure | Meaning |
|---|---|
| **Apnea minutes** | Minutes flagged as apnea, out of the minutes analysed |
| **Minutes per hour** | Apnea minutes divided by the hours analysed |
| **Flagged periods** | Stretches of consecutive flagged minutes |
| **Longest period** | The longest of those stretches |

**The chart**

![The chart, with a minute selected](docs/screenshots/chart.png)

- The **red strip** at the top marks every flagged minute.
- The **green line** is the model's apnea probability for each minute, averaged over 5 minutes.
- The **dashed line** is the threshold: a minute is flagged when the green line is above it.
- The bottom axis is **hours since the recording started**.

Move the mouse over the chart to read any minute: its probability, whether it is flagged, and when
it is.

**The flagged periods**

Click **List the … flagged periods** under the chart to see each period's start, end and length:

<img src="docs/screenshots/periods.png" width="300" alt="The list of flagged periods">

Times are **hours:minutes since the recording started**, not the time of day: `2:10` means two hours
and ten minutes into the night. The recorder has no clock, so the time of day is not known.

**Download CSV** saves the night minute by minute, one row per minute:

```
minute,start,probability,probability_smoothed,apnea
1,0:00,0.0039,0.0039,0
2,0:01,0,0.0039,0
3,0:02,0.0078,0.0039,0
...
```

The file is named after the patient, the night and the model, for example
`patient_1_NIGHT001_model_int8.csv`.

**Print** prints the page without the header, menus and buttons. The list of flagged periods is
printed only if it is open.

### 2.6 Choosing a model

The **Model** menu lists every model file in `inference/data/models/`, with its threshold next to
it. A model without a threshold is shown greyed out with "(no threshold)" and cannot be selected

- The choice applies to every page, until you reload the page.
- Results are saved **separately for each model**. Switching models shows that model's own
  results, or "Not analysed with this model yet".

---

## 3. Adding recordings

#### [TODO]: file upload

Recordings live in `inference/data/patients/`. **One folder per patient, one `.csv` file per night:**

```
inference/data/patients/
├── patient_1/          # the folder name is the patient's name in the interface
│   ├── NIGHT001.CSV    # one night
│   ├── NIGHT002.CSV
│   └── NIGHT003.CSV
└── patient_2/
    └── ...
```

**To add a patient or a night:**

1. Create the patient's folder if it is new.
2. Copy the night files into it.
3. Reload the page. No restart is needed.

**Which files are valide nights.** Every file ending in `.csv` (or `.CSV`), whatever its name. Other files
are ignored.

**File format.**

- One ECG sample per line, in **millivolts**.
- Sampled at 100 Hz (another rate can be set with `RECORDING_FS`.
- An optional first line that is not a number, such as `ecg_mv`, is skipped as a header.
- If a line has several columns separated by commas, only the first is used.

```
ecg_mv
0.025
0.015
-0.010
...
```

---

## 4. Models and thresholds

The service runs the `.tflite` files in `inference/data/models/`:

| File | Size | Threshold | Notes |
|---|---|---|---|
| `model_int8.tflite` | 124.5 KB | 0.23 | Full int8: the model for the ESP32-S3. Selected by default |
| `model_float32.tflite` | 295.8 KB | 0.20 | The same network unquantised; same outputs as `model.keras` |

Both come from the training notebook, [src/model.ipynb](src/model.ipynb), which writes them to
`results/`.

**Every model needs its own threshold.** The model gives a probability for each minute, and the
threshold turns it into a yes or no. Each threshold was chosen on the validation recordings. The int8 and float32 files of the same network need different thresholds,
because quantisation shifts the probabilities slightly. The notebook's Test set cell prints the
threshold of `model.keras` (0.20) and of `model_int8.tflite` (0.23)

Thresholds are set in `.env`:

```
MODEL_THRESHOLDS=model_int8.tflite=0.23,model_float32.tflite=0.20
```

**To add a model:**

1. Copy the `.tflite` file into `inference/data/models/`.
2. Add `<file name>=<threshold>` to `MODEL_THRESHOLDS` in `.env`, separated from the others by a
   comma.
3. Run `make up`.

The model must take one 60-second window, input shape `(1, 6000, 1)`, and be either full int8
(int8 input and output) or float32.

---

## 5. How a night is analysed

The service prepares the signal exactly as the training notebook did. On Apnea-ECG's test
recordings it reproduces the training data to within 1.5e-6.

1. **Filtering.** Frequencies below 0.5 Hz (breathing and movement drift) and above 40 Hz (muscle
   and electrical noise) are removed, as is 50 Hz mains interference. The filters run forwards and
   backwards, so the heartbeats do not move in time.
2. **Clean-up.** Missing samples are replaced by the night's median, and values are limited to
   ±5 mV.
3. **Minutes.** The night is cut into 60-second windows from the start. A final incomplete minute
   is dropped. Each window is rescaled on its own (mean 0, standard deviation 1), so the model sees
   the shape of the signal, not its amplitude.
4. **Model.** One probability of apnea per minute.
5. **Smoothing.** Each minute's probability is averaged with the two minutes before and the two
   after (5 minutes in all). At the start and end of the night, the first or last value stands in
   for the missing minutes.
6. **Decision.** A minute is flagged as apnea when its averaged probability is above the model's
   threshold.

---

## 6. Where results are stored

Each analysis is saved as one JSON file:

```
inference/data/analyses/<patient>/<night file>/<model file>.json
```

for example `inference/data/analyses/patient_1/NIGHT001.CSV/model_int8.tflite.json`. You can open
these files in VS Code or any text editor.

A file has two parts: `source`, what the result was computed from, and `result`, the result
itself.

```json
{
  "source": {
    "night_size": 21094749,
    "night_mtime_ns": 1790682911130659956,
    "model_size": 127504,
    "model_mtime_ns": 1790682568331006693,
    "threshold": 0.23,
    "smooth_window": 5
  },
  "result": {
    "patient": "patient_1",
    "night": "NIGHT001.CSV",
    "model": "model_int8.tflite",
    "sample_rate_hz": 100.0,
    "minutes": 522,
    "threshold": 0.23,
    "smooth_window": 5,
    "summary": {
      "apnea_minutes": 378,
      "apnea_fraction": 0.7241379310344828,
      "apnea_minutes_per_hour": 43.44827586206897
    },
    "prob": [0.0039, 0.0, 0.0078, ...],
    "prob_smoothed": [0.0039, 0.0039, 0.0039, ...],
    "apnea": [false, false, false, ...]
  }
}
```

**`source`**

| Field | Meaning |
|---|---|
| `night_size` | Size of the night file, in bytes |
| `night_mtime_ns` | Modification time of the night file |
| `model_size` | Size of the model file, in bytes |
| `model_mtime_ns` | Modification time of the model file |
| `threshold` | The model's threshold in `MODEL_THRESHOLDS` |
| `smooth_window` | Minutes in the moving average (`SMOOTH_WINDOW`) |

A saved result is shown only while these six values match the current files and settings.
Otherwise the night shows "Not analysed" for that model.

**`result`**

| Field | Meaning |
|---|---|
| `patient` | The patient's folder name |
| `night` | The night's file name |
| `model` | The model's file name |
| `sample_rate_hz` | Sampling rate of the night file (`RECORDING_FS`) |
| `minutes` | Number of minutes analysed. A final incomplete minute is not counted |
| `threshold` | Threshold used for the decision |
| `smooth_window` | Minutes averaged in the smoothing |
| `summary.apnea_minutes` | Minutes flagged as apnea: **Apnea minutes** in the interface |
| `summary.apnea_fraction` | `apnea_minutes` divided by `minutes`, from 0 to 1 |
| `summary.apnea_minutes_per_hour` | `apnea_minutes` divided by the hours analysed: **Minutes per hour** in the interface |
| `prob` | One value per minute: the model's apnea probability, from 0 to 1, rounded to 4 decimals |
| `prob_smoothed` | One value per minute: `prob` after the 5-minute average, the green line of the chart |
| `apnea` | One value per minute: `true` when `prob_smoothed` is above the threshold, the red strip of the chart |

The three lists each hold `minutes` values; the first value is the first minute of the recording.

Results survive restarts and every `make` command, including `make clean`. The folder is not
committed to git, because it holds results derived from patients' recordings.

---

## 7. Settings

Settings are in `.env`, in the project folder, created at the [first start](#first-start).
After a change, run `make up`.

| Setting | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `/models/model_int8.tflite` | The model selected when the page opens |
| `MODEL_THRESHOLDS` | `model_int8.tflite=0.23` | The threshold of each model file, comma-separated |
| `PATIENTS_DIR` | `/patients` | The patients folder, inside the container |
| `ANALYSES_DIR` | `/analyses` | The results folder, inside the container |
| `RECORDING_FS` | `100` | Sampling rate of the night files, in Hz. Other rates are resampled to 100 Hz |
| `SMOOTH_WINDOW` | `5` | Minutes in the moving average. |

The paths are inside the container and must match the folders mounted in `docker-compose.yml`:

| Project folder | Inside the container |
|---|---|
| `inference/data/models/` | `/models` |
| `inference/data/patients/` | `/patients` |
| `inference/data/analyses/` | `/analyses` |

---

## 8. Commands

Run from the project folder.

| Command | What it does |
|---|---|
| `make up` | Start the service (builds the image the first time) |
| `make stop` | Stop the service |
| `make ps` | Show whether the service is running and healthy |
| `make logs` | Show the service's log as it runs |
| `make rebuild` | Rebuild and restart: needed after changing any code in `inference/` |
| `make shell` | Open a terminal inside the running container |
| `make down` | Remove the container. Recordings and results are kept |
| `make build` | Build the image without starting it |
| `make clean` | **Removes the container and every unused Docker image and cache on this computer**, including other projects'. Recordings and results are kept |

---

## 9. Code and API

### Code layout

```
inference/
├── app.py             # the web server (FastAPI): the API and the page
├── pipeline.py        # preprocessing and the TFLite model wrapper
├── patients.py        # reading the patients folder and the night files
├── requirements.txt   # Python packages of the image
├── Dockerfile         # builds the interface with Node, then serves it with Python
├── web/               # the interface: React 19 + TypeScript, built with Vite
└── data/
    ├── models/        # .tflite models
    ├── patients/      # recordings
    └── analyses/      # saved results (written by the service, not committed)
```

`pipeline.py` and `patients.py` do not depend on the web server, so they can be imported from any
Python script.

### HTTP API

The interface talks to the server through these routes. `model` is a file name from the models
folder; without it, the model named by `MODEL_PATH` is used.

| Method and path | Returns |
|---|---|
| `GET /api/models` | `{default, models: [{name, size_bytes, threshold}]}` |
| `GET /api/patients` | `[{id, nights, latest_night}]` |
| `GET /api/patients/{id}?model=` | The patient's nights, oldest first: `{patient, model, sample_rate_hz, nights: [{file, number, samples, duration_s, size_bytes, analysis}]}`. `analysis` is the saved summary for `model`, or `null` |
| `GET /api/patients/{id}/nights/{file}/analysis?model=` | The saved result, without running the model. `404` if not analysed with `model` |
| `POST /api/patients/{id}/nights/{file}/analysis?model=` | Runs the model, saves and returns the result |
| `GET /health` | The default model, its threshold and the smoothing. Used by Docker's health check |
| `POST /analyze?fs=100` | Analyses an uploaded file (`.npy`, or `.csv`/`.txt` with one value per line and no header) with the default model, without saving it |


Errors return `{"detail": "<message>"}` with `404` (unknown patient, night or model), `409` (model
without a threshold, or that cannot be loaded) or `422` (the night could not be analysed).

### Working on the interface

The interface is in `inference/web/`: React 19, TypeScript, React Router and TanStack Query, built
with Vite. Node is only needed to work on it, not to run the service.

For live reload, keep the service running (`make up`) and start Vite:

```bash
cd inference/web
npm install       # the first time
npm run dev       # http://localhost:5173, API calls are forwarded to the service on port 8000
```

`npm run build` checks the types and builds into `inference/web/dist/`. To ship a change, run
`make rebuild`.
