import os
import json

def load_json(file_path: str) -> dict:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON not found: {file_path}")
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise ValueError(f"Invalid JSON: {file_path}") from e
    
def load_txt(file_path: str) -> str:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"TXT not found: {file_path}")
    try:
        with open(file_path, "r") as f:
            return f.read()
    except Exception as e:
        raise ValueError(f"Invalid TXT: {file_path}") from e