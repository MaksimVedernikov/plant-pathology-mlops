import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models, transforms

from common import PROJECT_ROOT, read_params, resolve_path

FEATURE_DIR = PROJECT_ROOT / "data" / "features"


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(device):
    try:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        base = models.mobilenet_v2(weights=weights)
    except Exception:
        base = models.mobilenet_v2(pretrained=True)

    model = torch.nn.Sequential(
        base.features,
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten()
    )

    model.eval()
    model.to(device)
    return model


def get_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def extract_features(df, model, transform, device):
    X = []
    y = []

    for _, row in df.iterrows():
        try:
            img_path = resolve_path(row["image_path"])
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                features = model(tensor)

            if isinstance(features, tuple):
                features = features[0]

            features = features.detach().cpu().numpy().squeeze()

            X.append(features)
            y.append(str(row["label"]))

        except Exception as e:
            print(f"Не удалось обработать {row.get('image_path')}: {e}")

    if len(X) == 0:
        return np.empty((0, 0)), np.empty((0,), dtype=str)

    return np.vstack(X), np.asarray(y, dtype=str)


def main():
    params = read_params()
    train_params = params.get("train", {})

    image_size = int(train_params.get("image_size", 128))
    use_augmentation = bool(train_params.get("use_augmentation", False))

    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    prepared_dir = PROJECT_ROOT / "data" / "prepared"
    augmented_dir = PROJECT_ROOT / "data" / "augmented"

    train_df = pd.read_csv(prepared_dir / "train.csv")

    if use_augmentation:
        aug_df = pd.read_csv(augmented_dir / "train.csv")
        if len(aug_df) > 0:
            train_df = pd.concat([train_df, aug_df], ignore_index=True)

    test_df = pd.read_csv(prepared_dir / "test.csv")

    device = get_device()
    model = get_model(device)
    transform = get_transform(image_size)

    print(f"Устройство: {device}")
    print(f"Размер изображений: {image_size}")
    print(f"Аугментация: {use_augmentation}")
    print(f"Обучающих изображений: {len(train_df)}")
    print(f"Тестовых изображений: {len(test_df)}")

    X_train, y_train = extract_features(train_df, model, transform, device)
    X_test, y_test = extract_features(test_df, model, transform, device)

    np.savez_compressed(FEATURE_DIR / "train_features.npz", X=X_train, y=y_train)
    np.savez_compressed(FEATURE_DIR / "test_features.npz", X=X_test, y=y_test)

    print("Признаки сохранены в data/features")


if __name__ == "__main__":
    main()