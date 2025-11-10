"""
Dataset preparation script for HybridResNet50V2–RViT brain tumour classification.

This script is designed to be:
- Reproducible
- Transparent
- Aligned with common preprocessing practice reported in:
  - Krishnan et al. (2024), "Enhancing brain tumor detection in MRI with a
    rotation invariant Vision Transformer" (Frontiers in Neuroinformatics).
  - Rasool et al. (2024), "Brain Tumor Classification using Deep Learning:
    A State-of-the-Art Review" (ETASR).
- Consistent with the workflow presented to Dr Koay.

Key design choices and rationale:

1. Dataset:
   - Uses the Kaggle Brain Tumor MRI dataset (Nickparvar).
   - This is the same dataset family used in Krishnan et al. for RViT evaluation
     and is widely adopted in works surveyed by Rasool et al.

2. Image size (224 x 224):
   - 224 x 224 is the standard input resolution for ImageNet-pretrained CNNs
     such as ResNet50V2 and for many ViT/RViT configurations.
   - Resizing at this stage ensures consistent input geometry.

3. Pixel normalisation:
   - I didn't overwrite images with [0, 1]-scaled values here.
   - Instead, I preserve standard 8-bit images and apply normalisation
     (img / 255.0, and optional mean/std) inside the training pipelines.
   - This matches common, well-documented practice in CNN/ViT literature and
     keeps the dataset reusable.

4. Data augmentation:
   - Not applied in this script.
   - Augmentation (small rotations ±10–15°, flips, mild brightness changes)
     will be applied on-the-fly during training.
   - This design is consistent with:
       * Krishnan et al.: rotational robustness handled in the RViT architecture.
       * Rasool et al.: augmentations described as part of model training,
         not permanent dataset mutation.

5. Splitting strategy (70/15/15, stratified):
   - Combines all available images, removes duplicates, and performs a
     stratified split:
       * Train 70%, Val 15%, Test 15%.
   - This matches your slide deck and is consistent with ranges (70/15/15,
     80/10/10, 80/20) reported as standard in the review literature.

6. Verification and reproducibility:
   - SHA1 hashing to detect exact duplicate files.
   - Export of:
       * data/splits/train.csv, val.csv, test.csv
       * results/dataset_summary.csv (class counts per split)
   - These artefacts document the prepared dataset configuration explicitly.
"""

import os
from pathlib import Path
import hashlib

from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm


# -------------------- CONFIGURATION --------------------

# Location of the raw Kaggle dataset
# structure:
# data/raw/brain-tumor-mri-dataset/kaggle_brain_mri_scan_dataset/
#   ├── Training/
#   │   ├── glioma/
#   │   ├── meningioma/
#   │   ├── pituitary/
#   │   └── notumor/
#   └── Testing/
#       ├── glioma/
#       ├── meningioma/
#       ├── pituitary/
#       └── notumor/
RAW_ROOT = Path("data/raw/brain-tumor-mri-dataset/kaggle_brain_mri_scan_dataset")

# Where resized, cleaned data will be stored.
# Structure created by this script:
# data/processed/
#   ├── train/{class}/
#   ├── val/{class}/
#   └── test/{class}/
PROCESSED_ROOT = Path("data/processed")

# CSVs mapping image paths to labels.
SPLITS_ROOT = Path("data/splits")

# Summary CSV with counts per class and split.
SUMMARY_PATH = Path("results/dataset_summary.csv")

# Target input size for ResNet50V2 / ViT / RViT.
IMG_SIZE = (224, 224)

# Canonical class labels based on Kaggle folder names.
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]

# Fixed seed for reproducibility.
RANDOM_STATE = 42

# ------------------------------------------------------


def collect_images() -> pd.DataFrame:
    """
    Traversin' the raw dataset directories and collectin' all image file paths
    with their associated class labels.

    This step:
    - Verifies directory consistency.
    - Encodes the label-space explicitly.
    """
    records = []

    for split in ["Training", "Testing"]:
        for cls in CLASSES:
            folder = RAW_ROOT / split / cls
            if not folder.exists():
                # If any class/split is missing, we skip; this is reported implicitly
                # in the final counts.
                continue

            for p in folder.iterdir():
                if p.is_file():
                    records.append(
                        {
                            "orig_path": str(p),
                            "class": cls,
                            "source_split": split.lower(),  # for traceability only
                        }
                    )

    df = pd.DataFrame(records)
    print(f"[INFO] Total images found (before deduplication): {len(df)}")
    return df


def sha1_of_file(path: str, block_size: int = 65536) -> str:
    """
    Compute SHA1 hash of a file.

    Rationale:
    - Ensurin' we can detect exact byte-level duplicates across Training/Testing
      folders or any accidental copies.
    - Duplication checks are routinely recommended in robust ML pipelines.
    """
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removin' exact duplicate images based on SHA1 hash.

    This:
    - Guards against information leakage across splits.
    - Aligns with good experimental practice emphasised in the literature,
      even if not always explicitly detailed.
    """
    print("[INFO] Computing SHA1 hashes for duplicate detection...")
    df["sha1"] = df["orig_path"].apply(sha1_of_file)

    before = len(df)
    df = df.drop_duplicates(subset="sha1").reset_index(drop=True)
    after = len(df)

    print(f"[INFO] Removed {before - after} duplicate files (if any).")
    return df


def make_splits(df: pd.DataFrame):
    """
    Performin' a stratified 70/15/15 split over the deduplicated dataset.

    Why this is defensible:
    - Rasool et al. highlight 70/15/15 and 80/10/10 as standard.
    - Stratification preserves class balance, which is explicitly recommended
      in both review and experimental works.
    """
    X = df["orig_path"].values
    y = df["class"].values

    # 70% train, 30% temp (stratified)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Splittin' 30% temp into 15% val and 15% test (i.e., 50/50 of temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    def to_df(paths, labels, split_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "orig_path": [str(Path(p)) for p in paths],
                "class": labels,
                "split": split_name,
            }
        )

    df_train = to_df(X_train, y_train, "train")
    df_val = to_df(X_val, y_val, "val")
    df_test = to_df(X_test, y_test, "test")

    print("[INFO] Split sizes (after deduplication):")
    print(f"  Train: {len(df_train)}")
    print(f"  Val:   {len(df_val)}")
    print(f"  Test:  {len(df_test)}")

    return df_train, df_val, df_test


def prepare_processed_dirs():
    """
    Creating the canonical processed directory structure:

    data/processed/train/{class}/
    data/processed/val/{class}/
    data/processed/test/{class}/
    """
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            out_dir = PROCESSED_ROOT / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)


def resize_and_copy(df_split: pd.DataFrame, split_name: str):
    """
    For each image in the given split:
    - Load with PIL.
    - Convert to RGB (ensures 3 channels for all models).
    - Resize to IMG_SIZE (224 x 224).
    - Savin' into data/processed/{split}/{class}/ with original filename.

    Note:
    - Only geometric standardisation is applied here.
    - Intensity normalisation and augmentations are applied later in the
      training/loader pipelines.
    """
    rows = df_split.to_dict(orient="records")

    for row in tqdm(rows, desc=f"[INFO] Processing {split_name}"):
        src = Path(row["orig_path"])
        cls = row["class"]
        dst = PROCESSED_ROOT / split_name / cls / src.name

        if dst.exists():
            # If already processed (e.g., rerun), skip.
            continue

        try:
            img = Image.open(src).convert("RGB")
            img = img.resize(IMG_SIZE)
            img.save(dst)
        except Exception as e:
            # Log and continue; problematic files can be inspected separately.
            print(f"[WARN] Failed to process {src}: {e}")


def save_csv_splits(df_train, df_val, df_test):
    """
    Savin' split definition CSVs using PROCESSED paths.

    These CSVs:
    - Are the authoritative mapping from file path to class label.
    - Are intended to be consumed by all training/experiment scripts.
    """
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)

    def map_to_processed(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        processed_paths = []
        for _, row in df.iterrows():
            orig = Path(row["orig_path"])
            cls = row["class"]
            processed_paths.append(
                str(PROCESSED_ROOT / split_name / cls / orig.name)
            )
        return pd.DataFrame({"image_path": processed_paths, "class": df["class"].values})

    train_csv = map_to_processed(df_train, "train")
    val_csv = map_to_processed(df_val, "val")
    test_csv = map_to_processed(df_test, "test")

    train_csv.to_csv(SPLITS_ROOT / "train.csv", index=False)
    val_csv.to_csv(SPLITS_ROOT / "val.csv", index=False)
    test_csv.to_csv(SPLITS_ROOT / "test.csv", index=False)

    print("[INFO] Saved split CSVs to data/splits/.")


def save_summary(df_train, df_val, df_test):
    """
    Creatin' a concise verification summary with class counts per split.

    This delivers the "Dataset Verification" artefact promised in the slides:
    - Confirms balanced splits.
    - Provides a documented reference for reports/appendices.
    """
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    def counts(df, split_name):
        c = df["class"].value_counts().rename("count").reset_index()
        c = c.rename(columns={"index": "class"})
        c.insert(0, "split", split_name)
        return c

    summary_df = pd.concat(
        [
            counts(df_train, "train"),
            counts(df_val, "val"),
            counts(df_test, "test"),
        ],
        axis=0,
        ignore_index=True,
    )

    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("[INFO] Saved dataset summary to results/dataset_summary.csv")
    print(summary_df)


def main():
    # 1. Collectin' all labelled images from raw dataset.
    df = collect_images()

    # 2. Removin exact duplicates (if any).
    df = drop_duplicates(df)

    # 3. Stratified 70/15/15 split.
    df_train, df_val, df_test = make_splits(df)

    # 4. Ensurin' processed folder structure exists.
    prepare_processed_dirs()

    # 5. Resizin' and copying to processed directories.
    resize_and_copy(df_train, "train")
    resize_and_copy(df_val, "val")
    resize_and_copy(df_test, "test")

    # 6. Saving CSVs with processed paths.
    save_csv_splits(df_train, df_val, df_test)

    # 7. Savin' summary statistics for verification and reporting.
    save_summary(df_train, df_val, df_test)

    print("[INFO] Dataset preparation completed in line with the documented plan.")


if __name__ == "__main__":
    main()
