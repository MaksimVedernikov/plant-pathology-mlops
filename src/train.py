from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from common import PROJECT_ROOT, read_params, load_xy


def get_model(name, seed):
    name = name.lower()

    if name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed
        )

    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=seed
        )

    if name == "svm":
        return LinearSVC(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed
        )

    raise ValueError(f"Неизвестная модель: {name}")

def main():
    params = read_params()

    train_params = params.get("train", {})
    prepare_params = params.get("prepare", {})

    model_name = str(train_params.get("model", "logistic_regression"))
    use_augmentation = bool(train_params.get("use_augmentation", False))
    image_size = int(train_params.get("image_size", 32))
    seed = int(prepare_params.get("seed", 42))

    prepared_dir = PROJECT_ROOT / "data" / "prepared"
    augmented_dir = PROJECT_ROOT / "data" / "augmented"

    train_df = pd.read_csv(prepared_dir / "train.csv")

    if use_augmentation:
        aug_df = pd.read_csv(augmented_dir / "train.csv")
        if len(aug_df) > 0:
            train_df = pd.concat([train_df, aug_df], ignore_index=True)

    print(f"Модель: {model_name}")
    print(f"Аугментация: {use_augmentation}")
    print(f"Обучающих изображений: {len(train_df)}")

    X, y = load_xy(train_df, image_size)

    if len(X) == 0:
        raise SystemExit("Не удалось загрузить обучающие изображения.")

    model = get_model(model_name, seed)
    model.fit(X, y)

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    out_path = models_dir / "model.joblib"

    joblib.dump(
        {
            "model": model,
            "model_name": model_name,
            "use_augmentation": use_augmentation,
            "image_size": image_size
        },
        out_path
    )

    print(f"Модель сохранена в {out_path}")


if __name__ == "__main__":
    main()