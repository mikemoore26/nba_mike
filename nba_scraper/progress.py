import json
import os
from .config import PROGRESS_FILE, DATA_DIR

def load_progress():
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE, "r") as f:
        data = json.load(f)
    return {(int(y), t) for y, t in data}

def save_progress(done_set):
    os.makedirs(DATA_DIR, exist_ok=True)
    data = [[y, t] for (y, t) in sorted(done_set)]
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)
