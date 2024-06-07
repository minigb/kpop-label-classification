import json
from pathlib import Path

def load_json(fn):
    fn = Path(fn)
    assert fn.exists(), f"{fn} does not exist"
    with open (fn, 'r') as f:
        data = json.load(f)
    return data

def save_json(fn, data):
    fn = Path(fn)
    fn.parent.mkdir(parents=True, exist_ok=True)
    with open(fn, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)