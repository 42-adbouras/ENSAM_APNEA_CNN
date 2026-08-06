import wfdb
from utils import DATA_DIR

SAVE_DIR = "apnea-ecg"
DATA_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading '{SAVE_DIR}' -> {DATA_DIR}")
wfdb.dl_database(SAVE_DIR, dl_dir=str(DATA_DIR), overwrite=False)
