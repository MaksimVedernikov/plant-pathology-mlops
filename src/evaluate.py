import json

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from common import PROJECT_ROOT, load_xy


def main():
    model_path = PROJECT_ROOT / "models" / "model.joblib"
    bundle = joblib.load(model_path)

    model = bundle["model"]
    image_size = bundle.get("image_size", 32)

    test_df = pd.read_csv(PROJECT_ROOT / "data" / "prepared" / "test.csv")

    print(f"Тестовых изображений: {len(test_df)}")

    X, y = load_xy(test_df, image_size)

    if len(X) == 0:
        raise SystemExit("Не удалось загрузить тестовые изображения.")

    predictions = model.predict(X)

    metrics = {
        "model": bundle.get("model_name"),
        "use_augmentation": bundle.get("use_augmentation"),
        "accuracy": float(accuracy_score(y, predictions)),
        "f1_macro": float(f1_score(y, predictions, average="macro", zero_division=0)),
        "test_samples": int(len(y))
    }

    metrics_path = PROJECT_ROOT / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()