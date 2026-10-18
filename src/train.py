import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from common import PROJECT_ROOT, read_params


def get_model(name, seed):
    name = name.lower()

    if name == "logistic_regression":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed
            ))
        ])

    if name == "random_forest":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1
            ))
        ])

    if name == "svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearSVC(
                max_iter=5000,
                class_weight="balanced",
                random_state=seed
            ))
        ])

    raise ValueError(f"Неизвестная модель: {name}")


def main():
    params = read_params()

    train_params = params.get("train", {})
    prepare_params = params.get("prepare", {})

    model_name = str(train_params.get("model", "logistic_regression"))
    use_augmentation = bool(train_params.get("use_augmentation", False))
    image_size = int(train_params.get("image_size", 128))
    seed = int(prepare_params.get("seed", 42))

    features_path = PROJECT_ROOT / "data" / "features" / "train_features.npz"

    data = np.load(features_path, allow_pickle=False)
    X = data["X"]
    y = data["y"]

    print(f"Модель: {model_name}")
    print(f"Аугментация: {use_augmentation}")
    print(f"Обучающих объектов: {len(X)}")

    if len(X) == 0:
        raise SystemExit("Не удалось загрузить обучающие признаки.")

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