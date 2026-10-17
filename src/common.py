import json
from pathlib import Path

import yaml
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_params():
    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_classes():
    classes_path = PROJECT_ROOT / "data" / "prepared" / "classes.json"
    with open(classes_path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path):
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    return PROJECT_ROOT / p


def image_to_vector(path, image_size):
    img = Image.open(resolve_path(path)).convert("L")
    img = img.resize((image_size, image_size))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.flatten()


def load_xy(df, image_size):
    X = []
    y = []

    for _, row in df.iterrows():
        try:
            vec = image_to_vector(row["image_path"], image_size)
            X.append(vec)
            y.append(str(row["label"]))
        except Exception as e:
            print(f"Не удалось открыть файл {row.get('image_path')}: {e}")

    if len(X) == 0:
        return np.empty((0, image_size * image_size)), np.empty((0,), dtype=str)

    return np.vstack(X), np.asarray(y, dtype=str)