import json

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from common import PROJECT_ROOT


def main():
    model_path = PROJECT_ROOT / "models" / "model.joblib"
    bundle = joblib.load(model_path)

    model = bundle["model"]

    test_path = PROJECT_ROOT / "data" / "features" / "test_features.npz"
    data = np.load(test_path, allow_pickle=False)

    X = data["X"]
    y = data["y"]

    print(f"Тестовых объектов: {len(X)}")

    if len(X) == 0:
        raise SystemExit("Не удалось загрузить тестовые признаки.")

    predictions = model.predict(X)

    metrics = {
        "model": bundle.get("model_name"),
        "use_augmentation": bundle.get("use_augmentation"),
        "accuracy": float(accuracy_score(y, predictions)),
        "f1_macro": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y, predictions, average="weighted", zero_division=0)),
        "test_samples": int(len(y))
    }

    metrics_path = PROJECT_ROOT / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()