import numpy as np
import pandas as pd
import json
from pathlib import Path
from huggingface_hub import snapshot_download

DATA_DIR = snapshot_download(
    repo_id="davidbkinney/country-risk-data",
    repo_type="dataset"
)

def load_json(filename):
    path = Path(DATA_DIR) / filename
    with open(path, "r") as f:
        return json.load(f)

def load_csv(filename):
    path = Path(DATA_DIR) / filename
    return pd.read_csv(path)

def safe_mean(x):
    x = [v for v in x if v is not None and np.isfinite(v)]
    return float(np.mean(x)) if len(x) > 0 else None

def clean_nan(val):
    """
    Recursively finds float('nan') values in lists and dictionaries 
    and replaces them with None (which becomes standard JSON null).
    Uses the 'x != x' property of float('nan') to avoid importing math.
    """
    if isinstance(val, dict):
        return {k: clean_nan(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [clean_nan(x) for x in val]
    elif isinstance(val, float) and val != val:  # Fast NaN check
        return None
    return val


