import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageEnhance

from common import PROJECT_ROOT, read_params, resolve_path

PREP_DIR = PROJECT_ROOT / "data" / "prepared"
AUG_DIR = PROJECT_ROOT / "data" / "augmented"
AUG_IMG_DIR = AUG_DIR / "images"

try:
    FLIP_LR = Image.Transpose.FLIP_LEFT_RIGHT
    FLIP_TB = Image.Transpose.FLIP_TOP_BOTTOM
except AttributeError:
    FLIP_LR = Image.FLIP_LEFT_RIGHT
    FLIP_TB = Image.FLIP_TOP_BOTTOM


def augment_image(img):
    if random.random() < 0.5:
        img = img.transpose(FLIP_LR)

    if random.random() < 0.5:
        img = img.transpose(FLIP_TB)

    angle = random.uniform(-15, 15)
    img = img.rotate(angle, expand=False)

    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))

    return img


def main():
    params = read_params()
    seed = int(params.get("prepare", {}).get("seed", 42))
    use_augmentation = bool(params.get("train", {}).get("use_augmentation", False))
    num_per_image = int(params.get("augment", {}).get("num_per_image", 1))

    random.seed(seed)

    AUG_DIR.mkdir(parents=True, exist_ok=True)
    AUG_IMG_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(PREP_DIR / "train.csv")
    rows = []

    if use_augmentation:
        for idx, row in train_df.iterrows():
            img_path = resolve_path(row["image_path"])

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Не удалось открыть {img_path}: {e}")
                continue

            for i in range(num_per_image):
                aug_img = augment_image(img)
                rel_path = Path("data") / "augmented" / "images" / f"aug_{idx}_{i}.jpg"
                full_path = PROJECT_ROOT / rel_path

                aug_img.save(full_path, "JPEG")

                rows.append({
                    "image_path": rel_path.as_posix(),
                    "label": row["label"]
                })

    out_df = pd.DataFrame(rows, columns=["image_path", "label"])
    out_df.to_csv(AUG_DIR / "train.csv", index=False)

    print(f"Аугментация включена: {use_augmentation}")
    print(f"Создано аугментированных изображений: {len(out_df)}")


if __name__ == "__main__":
    main()