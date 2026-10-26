# Apnea review: the inference service

**Apnea review** is a small web application that runs the project's trained models on overnight ECG recordings. You upload each patient's recordings, open the page in a browser, and for every night you get a minute-by-minute estimate of when the patient had sleep apnea.

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
RECORDING_FS=100
DB_USER=apnea
DB_PASS=change-me
DB_NAME=apnea
DB_URL=postgresql://apnea:change-me@database:5432/apnea
```

Replace `change-me` with a password of your choice, in both lines.

Each line is one setting, explained in [Settings](#7-settings). Then start the service:

```bash
make up    # builds the image (first time only) and starts the service
```

The first `make up` takes a few minutes, because it downloads and builds everything. Later starts take a few seconds.

Then open **http://localhost:8000** in your browser.

The patient list starts empty: add patients and their nights as described in
[Adding recordings](#3-adding-recordings).

To check that it is running, `make ps` should show both containers, `apnea_database` and
`apnea_inference`, as `(healthy)`.

### Stopping it

```bash
make stop    # stop the service; make up starts it again
```

Stopping does not lose anything: patients, recordings and results are in the database, which keeps
its data in `services/database/data/`.

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

**Create patient** adds a new patient; see [Adding recordings](#3-adding-recordings).

### 2.2 A patient's nights

![A patient's nights](docs/screenshots/patient.png)

At the top: the number of nights, the total recorded time and how many nights are analysed.

Below, one row per night, **newest at the top**. "Night 1" is always the oldest recording. Each row shows the file name, the night's length and its file size, then:

- **a result**, such as `43.6 min/h`, with a red bar (the bar is full at 60 minutes per hour), or
- **Not analysed**, when the selected model has not been run on that night yet.

Click a row to open the night. **Upload nights** adds recordings to this patient; see
[Adding recordings](#3-adding-recordings).

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
immediately, even after a restart. Clicking **Analyse** again runs the model again: the new result
is shown, and the earlier ones stay in the database.

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

The **Model** menu lists every model file in `services/inference/data/models/`, with its threshold next to
it. A model without a threshold is shown greyed out with "(no threshold)" and cannot be selected

- The choice applies to every page, until you reload the page.
- Results are saved **separately for each model**. Switching models shows that model's own
  results, or "Not analysed with this model yet".

---

## 3. Adding recordings

Every patient and every night is stored in the database: the night's file, byte for byte, with its
details and its results. They are added from the interface.

**Create a patient.** On the patients page, click **Create patient**, type a name and click
**Create**. The name can use letters, digits, spaces, `_` and `-`, and starts with a letter or digit.
Several patients may have the same name: each one is told apart by a number, shown in its page's
address (`/patient/7`). The new patient's page then opens.

![Creating a patient](docs/screenshots/new-patient.png)

**Upload nights.** On a patient's page, click **Upload nights** and choose one or more `.csv`
files, for example from the SD card. They are sent one after the other; each night appears in the
list as soon as it arrives. A short report then says how many nights were uploaded, and which files
were not and why:

![Nights uploaded](docs/screenshots/upload.png)

A file is not uploaded when:

- its name does not end in `.csv` (or `.CSV`);
- the patient already has a night with the same name. An upload never replaces a recording;
- the same night is already stored, for this patient or another one;
- it is not one number per line (see the file format below);
- it holds less than one minute of signal.

Each uploaded night keeps the modification time of the original file, so the nights are numbered
in the order they were recorded, whatever the order in which they were uploaded.

**File format.**

- One ECG sample per line, in **millivolts**.
- Sampled at 100 Hz (another rate can be set with `RECORDING_FS`).
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

The service runs the `.tflite` files in `services/inference/data/models/`:

| File | Size | Threshold | Notes |
|---|---|---|---|
| `model_int8.tflite` | 124.5 KB | 0.23 | Full int8: the model for the ESP32-S3. Selected by default |
| `model_float32.tflite` | 295.8 KB | 0.20 | The same network unquantised; same outputs as `model.keras` |

Both come from the training notebook, [training/model.ipynb](training/model.ipynb), which writes them to
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

1. Copy the `.tflite` file into `services/inference/data/models/`.
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

Everything the service knows is kept in a **PostgreSQL database**, which Docker runs next to the
service: the patients, the nights with their recordings, every analysis with its minute-by-minute
values, the versions of the model files and a log of actions. Only the model files stay on disk.
The database's files are in `services/database/data/`; they survive every `make` command, including
`make clean`, and are not committed to git.

- Analysing a night again adds a new analysis; the interface shows the newest one.
- An analysis counts for a night when it was made on the same content, with the same version of the
  model file, the same threshold and the same smoothing. Otherwise the night shows "Not analysed".

**[DATABASE.md](DATABASE.md)** explains the tables, how to look at the data (`make db-shell`), and
how to back up, restore or empty the database.

---

## 7. Settings

Settings are in `.env`, in the project folder, created at the [first start](#first-start).
After a change, run `make up`.

| Setting | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `/models/model_int8.tflite` | The model selected when the page opens |
| `MODEL_THRESHOLDS` | `model_int8.tflite=0.23` | The threshold of each model file, comma-separated |
| `RECORDING_FS` | `100` | Sampling rate of the night files, in Hz. Other rates are resampled to 100 Hz |
| `SMOOTH_WINDOW` | `5` | Minutes in the moving average. |
| `DB_USER` | | The database's user, created with the database |
| `DB_PASS` | | That user's password. To change it later, see [DATABASE.md](DATABASE.md#12-changing-the-password) |
| `DB_NAME` | | The database's name |
| `DB_URL` | | How the service reaches the database: `postgresql://<user>:<password>@database:5432/<database>` |

The paths are inside the container and must match the folders mounted in `docker-compose.yml`:

| Project folder | Inside the container |
|---|---|
| `services/inference/data/models/` | `/models` |
| `services/database/data/` | `/var/lib/postgresql`, in the `database` container |

---

## 8. Commands

Run from the project folder.

| Command | What it does |
|---|---|
| `make up` | Start the service (builds the image the first time) |
| `make stop` | Stop the service |
| `make ps` | Show whether the two containers are running and healthy |
| `make logs` | Show the service's log as it runs |
| `make rebuild` | Rebuild and restart: needed after changing any code in `services/inference/` |
| `make shell` | Open a terminal inside the service's container |
| `make db-shell` | Open PostgreSQL's command line, to look at the tables |
| `make down` | Remove the containers. Recordings and the database are kept |
| `make build` | Build both images, the database's and the service's, without starting them |
| `make clean` | **Removes the container and every unused Docker image and cache on this computer**, including other projects'. Recordings and the database are kept |

---

## 9. Code and API

### Code layout

```
services/
├── inference/
│   ├── source/
│   │   ├── app.py         # the web server (FastAPI): the API and the page
│   │   ├── pipeline.py    # preprocessing and the TFLite model wrapper
│   │   ├── patients.py    # the rules for patient names and night files, reading a night's samples
│   │   ├── db.py          # the database tables (SQLModel) and the connection
│   │   └── store.py       # patients, nights and analyses in the database
│   ├── requirements.txt   # Python packages of the image
│   ├── Dockerfile         # builds the interface with Node, then serves it with Python
│   ├── web/               # the interface: React 19 + TypeScript, built with Vite
│   └── data/
│       └── models/        # .tflite models
└── database/
    ├── Dockerfile         # the database's image: PostgreSQL 18
    ├── setup/setup.sh     # starts PostgreSQL with the settings from .env
    └── data/              # the database's files (not committed)
```

`pipeline.py`, `patients.py`, `db.py` and `store.py` do not depend on the web server, so they can be imported from any Python script.

### HTTP API

The interface talks to the server through these routes. `model` is a file name from the models folder; without it, the model named by `MODEL_PATH` is used.

| Method and path | Returns |
|---|---|
| `GET /api/models` | `{default, models: [{name, size_bytes, threshold}]}` |
| `GET /api/patients` | `[{id, name, nights}]`. `id` is the patient's number, `name` its name |
| `POST /api/patients` | Creates a patient. JSON body `{"name": "<name>"}`. Returns `201` and `{id, name}` |
| `GET /api/patients/{id}?model=` | The patient and its nights, oldest first: `{id, name, model, sample_rate_hz, nights: [{file, number, samples, duration_s, size_bytes, analysis}]}`. `analysis` is the saved summary for `model`, or `null` |
| `POST /api/patients/{id}/nights` | Stores one night in the database: the `.csv` file as the form field `file` and, optionally, `modified_ms`, the original file's modification time in milliseconds. Returns `201` and `{id, file}`; `409` if the patient has a night of that name, or if the same content is already stored |
| `GET /api/patients/{id}/nights/{file}/analysis?model=` | The newest saved result, without running the model. `404` if not analysed with `model` |
| `POST /api/patients/{id}/nights/{file}/analysis?model=` | Runs the model, saves the result as a new analysis and returns it |
| `GET /health` | The default model, its threshold and the smoothing, after checking the database. Used by Docker's health check |
| `POST /analyze?fs=100` | Analyses a file sent as the form field `file`, with the default model, without saving it. A name ending in `.npy` is read as a one-dimensional NumPy array; any other file as text, one value per line in mV, no header. `fs` is the file's sampling rate (default 100) |


Errors return `{"detail": "<message>"}` with `400` (a patient name, file or upload that is not
accepted), `404` (unknown patient, night or model), `409` (a night name already used by the patient, a
night already stored, a night file changed since it was registered, or a model without a threshold
or that cannot be loaded) or `422` (the night could not be
analysed).
