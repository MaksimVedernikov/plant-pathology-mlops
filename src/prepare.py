import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from common import PROJECT_ROOT, RAW_DIR, IMG_EXTS, read_params

PREP_DIR = PROJECT_ROOT / "data" / "prepared"
FORBIDDEN_FOLDER_NAMES = {"images", "image", "train", "test", "val", "valid", "validation"}


def find_csv():
    if not RAW_DIR.exists():
        return None

    csv_files = list(RAW_DIR.rglob("*.csv"))
    if not csv_files:
        return None

    # Сначала ищем именно train.csv
    for p in csv_files:
        if p.name.lower() == "train.csv":
            return p

    # Если нет, ищем любой train-файл, но не test и не submission
    for p in csv_files:
        name = p.name.lower()
        if "train" in name and "test" not in name and "submission" not in name:
            return p

    return None


def build_image_index():
    index = {
        "name": {},
        "stem": {},
    }

    if not RAW_DIR.exists():
        return index

    for path in RAW_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMG_EXTS:
            continue

        try:
            rel = path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()

        if path.name not in index["name"]:
            index["name"][path.name] = rel

        if path.stem not in index["stem"]:
            index["stem"][path.stem] = rel

    return index


def to_project_relative(path):
    path = Path(path).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def candidate_exists(candidate):
    candidate = Path(candidate)

    if candidate.is_absolute():
        if candidate.exists() and candidate.is_file():
            return candidate
        return None

    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    project_candidate = PROJECT_ROOT / candidate
    if project_candidate.exists() and project_candidate.is_file():
        return project_candidate.resolve()

    return None


def resolve_image(value, csv_dir, image_index):
    value = str(value).strip().replace("\\", "/")
    if not value:
        return None

    basename = Path(value).name
    stem = Path(basename).stem

    candidates = [
        value,
        basename,
        RAW_DIR / value,
        RAW_DIR / basename,
        RAW_DIR / "images" / value,
        RAW_DIR / "images" / basename,
        RAW_DIR / "Train" / value,
        RAW_DIR / "Train" / basename,
        RAW_DIR / "train" / value,
        RAW_DIR / "train" / basename,
        csv_dir / value,
        csv_dir / basename,
    ]

    for candidate in candidates:
        full_path = candidate_exists(candidate)
        if full_path is not None:
            return to_project_relative(full_path)

    if basename in image_index["name"]:
        return image_index["name"][basename]

    if stem in image_index["stem"]:
        return image_index["stem"][stem]

    return None


def find_image_column(df):
    preferred = [
        "image_id",
        "image",
        "image_path",
        "filename",
        "file",
        "path",
        "image_name",
        "file_name",
        "id",
    ]

    for col in preferred:
        if col in df.columns:
            return col

    for col in df.columns:
        try:
            has_image_ext = df[col].astype(str).str.contains(
                r"\.(jpg|jpeg|png|bmp|webp)",
                case=False,
                regex=True
            ).any()
            if has_image_ext:
                return col
        except Exception:
            continue

    return df.columns[0]

def find_label_column(df):
    exact_candidates = [
        "label",
        "labels",
        "class",
        "class_name",
        "target",
        "target_label",
        "y"
    ]

    for col in exact_candidates:
        if col in df.columns:
            return col

    for col in df.columns:
        if col in {"image_id", "image", "filename", "file", "path", "id"}:
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            if any(word in col for word in ["label", "class", "target", "disease"]):
                return col

    return None

def collect_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    img_col = find_image_column(df)
    label_col = find_label_column(df)

    if label_col is not None and df[label_col].notna().any():
        labels = df[label_col].astype(str)
    else:
        candidates = []
        for col in df.columns:
            if col == img_col:
                continue
            if col in {"id", "index"}:
                continue
            if any(word in col for word in
                   ["image", "file", "path", "name", "index", "level", "unnamed"]):
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                candidates.append(col)

        if len(candidates) >= 2:
            onehot = df[candidates].fillna(0)
            labels = onehot.idxmax(axis=1).astype(str)
        elif len(candidates) == 1:
            labels = df[candidates[0]].astype(str)
        else:
            non_image_cols = [c for c in df.columns if c != img_col]
            if not non_image_cols:
                raise ValueError("В CSV нет ни колонки с метками, ни колонок с one-hot метками.")
            labels = df[non_image_cols[-1]].astype(str)

    image_index = build_image_index()
    rows = []
    missing = 0

    for idx, row in df.iterrows():
        resolved = resolve_image(row[img_col], csv_path.parent, image_index)
        if resolved is None:
            missing += 1
            continue

        label = str(labels.loc[idx]).strip()
        if label.lower() in {"", "nan", "none"}:
            continue

        rows.append({
            "image_path": resolved,
            "label": label
        })

    if missing > 0:
        print(f"Пропущено {missing} изображений, потому что не удалось найти файлы.")

    out = pd.DataFrame(rows, columns=["image_path", "label"])
    out = out.drop_duplicates(subset="image_path").reset_index(drop=True)
    return out


def has_images(folder):
    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMG_EXTS:
            return True
    return False


def collect_from_folders():
    rows = []

    bases = []
    for name in ["train", "Train", "images"]:
        if (RAW_DIR / name).is_dir():
            bases.append(RAW_DIR / name)
    bases.append(RAW_DIR)

    for base in bases:
        if not base.is_dir():
            continue

        class_dirs = []
        for d in base.iterdir():
            if not d.is_dir():
                continue
            if base == RAW_DIR and d.name.lower() in FORBIDDEN_FOLDER_NAMES:
                continue
            if has_images(d):
                class_dirs.append(d)

        if len(class_dirs) >= 2:
            for class_dir in class_dirs:
                label = class_dir.name
                for img_path in class_dir.rglob("*"):
                    if img_path.is_file() and img_path.suffix.lower() in IMG_EXTS:
                        rows.append({
                            "image_path": to_project_relative(img_path),
                            "label": label
                        })
            break

    return pd.DataFrame(rows, columns=["image_path", "label"])


def main():
    params = read_params()
    prep_params = params.get("prepare", {})

    seed = int(prep_params.get("seed", 42))
    test_size = float(prep_params.get("test_size", 0.2))
    max_images_per_class = int(prep_params.get("max_images_per_class", 0))

    PREP_DIR.mkdir(parents=True, exist_ok=True)

    csv_df = pd.DataFrame(columns=["image_path", "label"])
    folder_df = pd.DataFrame(columns=["image_path", "label"])

    csv_path = find_csv()
    if csv_path is not None:
        print(f"Использую CSV: {csv_path}")
        try:
            csv_df = collect_from_csv(csv_path)
        except Exception as e:
            print(f"Не удалось разобрать CSV: {e}")

    folder_df = collect_from_folders()

    if not csv_df.empty and csv_df["label"].nunique() >= 2:
        df = csv_df
        print("Источник разметки: CSV")
    elif not folder_df.empty and folder_df["label"].nunique() >= 2:
        df = folder_df
        print("Источник разметки: папки с классами")
    elif not csv_df.empty:
        df = csv_df
        print("Источник разметки: CSV")
    elif not folder_df.empty:
        df = folder_df
        print("Источник разметки: папки с классами")
    else:
        raise SystemExit("Не нашел данные в data/raw. Проверь структуру датасета.")

    df = df.dropna(subset=["image_path", "label"]).reset_index(drop=True)

    print(f"Всего изображений до ограничения: {len(df)}")
    print("Классы до ограничения:", ", ".join(sorted(df["label"].astype(str).unique())))

    if max_images_per_class > 0:
        parts = []
        for label, group in df.groupby("label"):
            n = min(len(group), max_images_per_class)
            parts.append(group.sample(n=n, random_state=seed))

        if parts:
            df = pd.concat(parts, ignore_index=True)
        else:
            raise SystemExit("После ограничения по классам не осталось данных.")

    print(f"Изображений после ограничения: {len(df)}")
    print("Классы:", ", ".join(sorted(df["label"].astype(str).unique())))

    classes = sorted(df["label"].astype(str).unique().tolist())

    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            stratify=df["label"]
        )
    except ValueError:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=seed,
            shuffle=True
        )

    train_df.to_csv(PREP_DIR / "train.csv", index=False)
    test_df.to_csv(PREP_DIR / "test.csv", index=False)

    with open(PREP_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train_df)}")
    print(f"Test: {len(test_df)}")
    print("Данные сохранены в data/prepared")


if __name__ == "__main__":
    main()