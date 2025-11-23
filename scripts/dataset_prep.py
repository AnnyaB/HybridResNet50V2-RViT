"""
Dataset preparation script for HybridResNet50V2–RViT brain tumour classification.

This script is designed to be:
- Corresponded with common preprocessing practice reported in:
  - Krishnan et al. (2024), "Enhancing brain tumor detection in MRI with a
    rotation invariant Vision Transformer" (Frontiers in Neuroinformatics).
  - Rasool et al. (2024), "Brain Tumor Classification using Deep Learning:
    A State-of-the-Art Review" (ETASR).


Key design choices and rationale:

1. Dataset:
   - Uses the Kaggle Brain Tumor MRI dataset (Nickparvar, 2024).
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

6. Verification and reproducibility:
   - SHA1 hashing to detect exact duplicate files.
   - Export of:
       * data/splits/train.csv, val.csv, test.csv
       * results/dataset_summary.csv (class counts per split)
   - These artefacts document the prepared dataset configuration explicitly.

Additional narrative (what I am doing and why):

- I wrote this script as the **single source of truth** for dataset preparation
  for my hybrid Modified ResNet50V2–RViT pipeline. I wanted to avoid doing
  ad-hoc manual preprocessing steps in notebooks, because that makes it hard
  to reproduce the exact dataset state later when writing the thesis or
  re-running experiments.

- I kept this script model-agnostic but architecture-awared, meaning:
  * I resized to 224×224 because both the modified ResNet50V2 (Sarada et al.)
    and RViT (Krishnan et al.) expect ImageNet-style input size.
  * I only enforced geometry (size, RGB channels) here and keep pixel scale as
    raw 0–255, so that I can plug in different normalisation schemes for
    ResNet50V2, RViT, ResViT, etc. inside the training code without rewriting
    the dataset.

- I explicitly merged Kaggle “Training” and “Testing” folders into one pool,
  then re-split with stratification. I did this to:
  * avoid leaking Kaggle’s original split design into my experimental design,
  * have full control over my own 70/15/15 splits for fair comparison across
    different models.

- I computed SHA1 hashes of all files and drop duplicates so that:
  * no exact MRI image appears in more than one split,
  * my evaluation metrics for ResNet50V2 and RViT are not artificially inflated
    by information leakage between train/val/test.

- I generated CSV mapping files (train/val/test) and a summary CSV with
  per-class counts, so I can:
  * show a concrete “Dataset Verification” artefact in my report,
  * plug the same splits into multiple training scripts in a clean, traceable
    way.
"""

import os
from pathlib import Path
import hashlib

from PIL import Image
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm


# -------------------------- CONFIGURATION SECTION --------------------------
# In this section I defined all high-level configuration values in one place.
# This makes the script easier to read and ensures that if I need to change
# something (e.g., image size or folder structure), I only change it here.


# Location of the raw Kaggle dataset
# structure:
# data/raw/brain-tumor-mri-dataset/kaggle_brain_mri_scan_dataset/
#   - Training/
#      - glioma/
#      - meningioma/
#      - pituitary/
#      - notumor/
#   - Testing/
#       - glioma/
#       - meningioma/
#       - pituitary/
#       - notumor/
# I fixed this root path so that all downstream code can build on it
# using Path operations instead of string concatenation.
RAW_ROOT = Path("data/raw/brain-tumor-mri-dataset/kaggle_brain_mri_scan_dataset")

# Where resized, cleaned data will be stored.
# Structure created by this script:
# data/processed/
#   - train/{class}/
#   - val/{class}/
#   - test/{class}/
# I separated "raw" and "processed" so that I never accidentally overwrite
# the original Kaggle dataset and can regenerate processed data anytime.
PROCESSED_ROOT = Path("data/processed")

# CSVs mapping image paths to labels.
# These are used by my training scripts to know which image belongs
# to which class and split.
SPLITS_ROOT = Path("data/splits")

# Summary CSV with counts per class and split.
# This acts as my "Dataset Verification" artefact for the report.
SUMMARY_PATH = Path("results/dataset_summary.csv")

# Target input size for ResNet50V2 / ViT / RViT.
# I chose 224 x 224 to:
# - match ImageNet-pretrained ResNet50V2 (Sarada et al.)
# - match the RViT configuration (Krishnan et al., 2024)
IMG_SIZE = (224, 224)

# Canonical class labels based on Kaggle folder names.
# I kept these labels consistent across the entire project so that
# plots, confusion matrices and CSVs use the same naming.
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]

# Fixed seed for reproducibility.
# I used this for all train/val/test splits so my results can be
# reproduced exactly later when writing up experiments.
RANDOM_STATE = 42

# ------------------------------------------------------


def collect_images() -> pd.DataFrame:
    """
    Traversin' the raw dataset directories and collectin' all image file paths
    with their associated class labels.

    This step:
    - Verifies directory consistency.
    - Encodes the label-space explicitly.

    What I did here:
    ----------------
    - I walked through the Kaggle directory structure (both "Training" and "Testing")
      for each of the four classes: glioma, meningioma, pituitary, notumor.
    - For every file I found, I recorded:
        * its absolute/relative path (orig_path),
        * its class label (class),
        * which original Kaggle split it came from (source_split).

    Why I did it this way:
    ---------------------
    - I decided to ignore Kaggle's original "Training/Testing" semantics and
      instead treat the entire dataset as one pool of labelled images.
      Then I created my own 70/15/15 stratified splits.
    - I kept source_split only for traceability in case I need to analyse
      later where certain images originally came from.
    - I used a pandas DataFrame as the central in-memory representation,
      because it is convenient for filtering, deduplication, and exporting.
    """
    records = []

    # Loopin' over the two top-level splits provided by Kaggle.
    for split in ["Training", "Testing"]:
        # Loopin' over the four tumour classes.
        for cls in CLASSES:
            folder = RAW_ROOT / split / cls
            if not folder.exists():
                # If any class/split is missing, we skip; this is reported implicitly
                # in the final counts.
                # I chose to skip instead of failing hard, so the script is robust
                # even if someone downloaded a partial dataset.
                continue

            # Iteratin' over all files in the class folder.
            for p in folder.iterdir():
                if p.is_file():
                    # For each file, I stored a record with:
                    # - original path
                    # - class label
                    # - original Kaggle split (for traceability)
                    records.append(
                        {
                            "orig_path": str(p),
                            "class": cls,
                            "source_split": split.lower(),  # for traceability only
                        }
                    )

    # Convertin' list of dicts into a DataFrame for easier manipulation.
    df = pd.DataFrame(records)
    print(f"[INFO] Total images found (before deduplication): {len(df)}")
    return df


def sha1_of_file(path: str, block_size: int = 65536) -> str:
    """
    Computin' SHA1 hash of a file.

    Rationale:
    - Ensurin' we can detect exact byte-level duplicates across Training/Testing
      folders or any accidental copies.
    - Duplication checks are routinely recommended in robust ML pipelines.

    What I did here:
    ----------------
    - I opened the file in binary mode and read it in chunks (blocks) so that
      even large images will not overflow memory.
    - I updated a SHA1 hash with each chunk until the whole file is read.
    - I returned the final hexadecimal digest as a string.

    Why I chose SHA1 and this pattern:
    ----------------------------------
    - I only needed a collision-resistant identifier for exact duplicates.
      I am not using hashes for security here, only for deduplication.
    - SHA1 is more than sufficient for this purpose and is very fast.
    - Reading in blocks (block_size=65536) is a standard pattern to handle
      potentially large files efficiently.
    """
    h = hashlib.sha1()
    with open(path, "rb") as f:
        # Read in fixed-size blocks and update the hash incrementally.
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removin' exact duplicate images based on SHA1 hash.

    This:
    - Guards against information leakage across splits.
    - Corresponds with good experimental practice emphasised in the literature,
      even if not always explicitly detailed.

    What I did here:
    ----------------
    - For each image path in the DataFrame, I computed a SHA1 hash of its file.
    - I then dropped all rows that share the same SHA1 hash, keeping only one
      instance of each unique file.
    - After deduplication, I reset the index to keep the DataFrame tidy.

    Why I did this:
    --------------
    - The Kaggle dataset merges data from several sources (Figshare, SARTAJ, Br35H),
      and there can be exact duplicates between "Training" and "Testing" or across
      sources.
    - If I did not remove duplicates, the same MRI could end up in train and test:
        -> the model would see the same image during training and evaluation,
          artificially inflating accuracy.
    - By deduplicating before splitting, I made sure each physical image appears
      in exactly one of train/val/test.
    """
    print("[INFO] Computing SHA1 hashes for duplicate detection...")
    # Compute SHA1 hash for each original path.
    df["sha1"] = df["orig_path"].apply(sha1_of_file)

    before = len(df)
    # Drop rows with duplicate hashes (keep first occurrence).
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

    What I did here:
    ----------------
    - I first splitted the full deduplicated dataset into:
        * 70% train
        * 30% temporary pool (temp)
      using stratified sampling so that the class balance is preserved.
    - I then splitted that 30% temporary pool into:
        * 15% validation
        * 15% test
      by dividing it 50/50, again with stratification.

    Why I chose 70/15/15 with stratification:
    -----------------------------------------
    - My project plan promise a 70/15/15 split to:
        * give enough data to train ResNet50V2 and RViT reliably,
        * keep a proper validation set for hyperparameter tuning,
        * keep a proper test set for final evaluation.
    - Stratified splits ensure that each split has roughly the same proportion
      of glioma, meningioma, pituitary, and no tumour images. This is crucial
      for fair evaluation and to avoid trivial skew.

    Output:
    -------
    - Three DataFrames:
        * df_train: paths and labels for training set
        * df_val:   paths and labels for validation set
        * df_test:  paths and labels for test set
    """
    # Features: original image paths.
    X = df["orig_path"].values
    # Labels: tumour class names.
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
        """
        Helper: convert arrays of paths and labels into a tidy DataFrame
        with an explicit 'split' column.
        """
        return pd.DataFrame(
            {
                "orig_path": [str(Path(p)) for p in paths],
                "class": labels,
                "split": split_name,
            }
        )

    # Buildin' dedicated DataFrames for each split.
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

    What I did here:
    ----------------
    - I created (if necessary) all the directories into which I will later
      save the resized 224×224 RGB images.

    Why I did this first:
    --------------------
    - Ensuring the directory structure exists before resizing/copying avoids
      having to check for missing folders for every single file.
    - Having a clean, predictable structure makes it easier for the training
      scripts (and for a human) to navigate the processed dataset:
        * e.g., data/processed/train/glioma contains all glioma train images.
    """
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            out_dir = PROCESSED_ROOT / split / cls
            # parents=True ensures all missing parent directories are also created.
            # exist_ok=True prevents errors if the folder already exists (e.g., rerun).
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

    What I did here:
    ----------------
    - I iterated over the rows for a given split (train, val, or test).
    - For each row:
        * I opened the original image from orig_path.
        * I converted it to RGB to guarantee 3 channels (some MRIs might be grayscale).
        * I resized it to (224, 224) pixels.
        * I saved it into the processed directory under:
              data/processed/{split_name}/{class}/{original_filename}

    Why I kept original filenames:
    ------------------------------
    - This makes it easy to trace back any processed file to the original image.
    - It also keeps CSV mappings intuitive and human-readable.

    Why I did not normalise pixel intensities here:
    ----------------------------------------------
    - I wanted to keep the processed dataset generic and reusable:
        * ResNet50V2 might use one normalisation scheme,
        * RViT might use another.
    - By only doing resizing and channel conversion here, I kept the dataset
      clean and apply normalisation inside the model pipelines.
    """
    # Convertin' DataFrame rows to a list of dictionaries for easy iteration.
    rows = df_split.to_dict(orient="records")

    # tqdm gives a progress bar in the terminal for convenience.
    for row in tqdm(rows, desc=f"[INFO] Processing {split_name}"):
        src = Path(row["orig_path"])
        cls = row["class"]
        dst = PROCESSED_ROOT / split_name / cls / src.name

        if dst.exists():
            # If already processed (e.g., rerun), skip.
            # This makes the script idempotent: I can re-run without duplicating work.
            continue

        try:
            # Openin' original image and enforce RGB (3 channels).
            img = Image.open(src).convert("RGB")
            # Resizin' to 224x224 using PIL's default interpolation.
            img = img.resize(IMG_SIZE)
            # Savn' to the processed folder; keep the same filename.
            img.save(dst)
        except Exception as e:
            # Log and continue; problematic files can be inspected separately.
            # I chose not to crash the whole script on a single corrupted image.
            print(f"[WARN] Failed to process {src}: {e}")


def save_csv_splits(df_train, df_val, df_test):
    """
    Savin' split definition CSVs using PROCESSED paths.

    These CSVs:
    - Are the authoritative mapping from file path to class label.
    - Are intended to be consumed by all training/experiment scripts.

    What I did here:
    ----------------
    - I took the three DataFrames (train, val, test) that point to original
      image paths and map them to the corresponding processed image paths
      in data/processed/{split}/{class}/.
    - I then saved each split as a CSV with columns:
        * data/splits/train.csv
        * data/splits/val.csv
        * data/splits/test.csv

    Why this is important:
    ----------------------
    - Instead of scanning directories at training time, I wanted my training
      scripts to read a simple CSV:
          image_path, class
      which makes experiments deterministic and easy to track.
    - If I ever need to change image size or processing, I can regenerate
      processed images and just point to the correct CSVs.
    """
    # Ensurin' the splits directory exists.
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)

    def map_to_processed(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        """
        Convertin' a DataFrame with orig_path/class into a DataFrame with
        processed image paths and the same class labels.
        """
        processed_paths = []
        for _, row in df.iterrows():
            orig = Path(row["orig_path"])
            cls = row["class"]
            processed_paths.append(
                str(PROCESSED_ROOT / split_name / cls / orig.name)
            )
        # The resulting DataFrame is exactly what training code needs.
        return pd.DataFrame({"image_path": processed_paths, "class": df["class"].values})

    # Map each logical split to the processed folder structure.
    train_csv = map_to_processed(df_train, "train")
    val_csv = map_to_processed(df_val, "val")
    test_csv = map_to_processed(df_test, "test")

    # Save CSV files without index column.
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

    What I did here:
    ----------------
    - For each split (train, val, test), I counted how many images belong
      to each class.
    - I then concatenated these counts into a single DataFrame with columns:
        split, class, count
    - I saved this as results/dataset_summary.csv and also print it.

    Why I did this:
    --------------
    - This summary is extremely useful in the report:
        * I can drop it in the appendix or Dataset section to show
          per-class distribution in each split.
        * It proves that I used stratified sampling and that there is no
          extreme imbalance across splits.
    - It also acts as a quick visual sanity check when I re-run the script.
    """
    # Ensurin' the parent directory for the summary file exists.
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    def counts(df, split_name):
        """
        Countin' number of samples per class for a given split.
        """
        c = df["class"].value_counts().rename("count").reset_index()
        c = c.rename(columns={"index": "class"})
        c.insert(0, "split", split_name)
        return c

    # Buildin' a single DataFrame with counts for train, val, and test.
    summary_df = pd.concat(
        [
            counts(df_train, "train"),
            counts(df_val, "val"),
            counts(df_test, "test"),
        ],
        axis=0,
        ignore_index=True,
    )

    # Savin' the summary to CSV.
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print("[INFO] Saved dataset summary to results/dataset_summary.csv")
    print(summary_df)


def main():
    """
    Orchestrates the full dataset preparation pipeline.

    High-level steps I performed here:
    --------------------------------
    1. Collected all labelled images from the raw Kaggle dataset.
    2. Removed any exact duplicate files using SHA1 hashes.
    3. Created a stratified 70/15/15 train/val/test split.
    4. Created the processed directory structure.
    5. Resized all images to 224×224 RGB and save them into processed folders.
    6. Saved CSVs that map processed image paths to labels for each split.
    7. Saved a summary CSV with per-class counts per split.

    This main() function is what makes the script a **single button**:
    I just ran it once and it prepares the entire dataset exactly as planned
    for the Hybrid Modified ResNet50V2–RViT experiments.
    """
    # 1. Collectin' all labelled images from raw dataset.
    df = collect_images()

    # 2. Removin exact duplicates (if any).
    df = drop_duplicates(df)

    # 3. Stratifyin' 70/15/15 split.
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
  
    # I kept this guard so that:
    # - main() only runs when this file is executed as a script,
    # - but not when it is imported as a module in another Python file.
    main()
