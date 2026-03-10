# THIS IS THE MAIN FILE FOR DATASET PREPROCESSING 
# I'VE DONE HERE SEVERAL STEPS TAKING GUIDANCE FROM MY SUPERVISOR AND THE PAPERS
# IT DOESN'T DO TOO MUCH TO MAKE THE IMAGES POLISHED FOR TRAINING AND TESTING SINCE THE IDEA IS TO KEEP IT QUITE GENERALIZED
# IT DOES CROPPING, RESIZING, CONVERTING TO RGB, SPLITTING, AND DUPLICATE REMOVAL IN A LEAKAGE-SAFE WAY, 
# AND ALSO ANALYZES THE RAW IMAGES TO FLAG ANY POTENTIAL QUALITY ISSUES (FAILED LOADS, TOO DARK/BRIGHT, LOW CONTRAST) 
# WITHOUT DROPPING ANY IMAGES (FLAGS ONLY). NO AUGMENTATION OR INTENSITY NORMALIZATION IS DONE HERE, 
# AS I WANT TO KEEP THIS STEP AS SIMPLE AND GENERAL AS POSSIBLE.
# FIND MORE ABOUT USING LANCZOS RESAMPLING HERE - https://github.com/jeffboody/Lanczos
# FOR SHA1 SEE https://docs.python.org/3/library/hashlib.html

# Libraries I needed 
# (OS/filesystem, hashing, math utilities)
import os
from pathlib import Path
import hashlib
import math


# numpy: numeric ops, image arrays, stats
# PIL: image I/O and EXIF orientation safety and conversions
# sklearn: stratified splitting
# pandas: tabular DataFrames and CSV output
# tqdm: progress bars (makes long loops visible)
import numpy as np
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm


# CONFIGURATION SECTION


# PROJECT_ROOT is set to the repository root by going up one level from this file:
# e.g. project/scripts/dataset_prep.py -> project/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# RAW_ROOT points to my raw Kaggle dataset folder structure.
# This script assumes I placed the Kaggle dataset under:
# data/raw/brain-tumor-mri-dataset/kaggle_brain_mri_scan_dataset/ (WHERE IT IS ACTUALLY)
RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "brain-tumor-mri-dataset"
    / "kaggle_brain_mri_scan_dataset"
)

# Base folders for outputs (processed images, split CSVs, and results)
PROCESSED_BASE = PROJECT_ROOT / "data" / "processed"
SPLITS_BASE = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results"

# Variant name: lets you keep multiple preprocessing versions side-by-side
VARIANT = "tightcrop"

# Concrete output roots for this preprocessing variant
PROCESSED_ROOT = PROCESSED_BASE / VARIANT
SPLITS_ROOT = SPLITS_BASE / VARIANT

# Output artifact paths (all go to results/)
SUMMARY_PATH                 = RESULTS_DIR / "dataset_summary.csv"
RAW_STATS_PATH               = RESULTS_DIR / "raw_image_stats.csv"
RAW_RESOLUTION_SUMMARY_PATH  = RESULTS_DIR / "raw_resolution_summary.csv"
RAW_QUALITY_SUMMARY_PATH     = RESULTS_DIR / "raw_quality_flags_summary.csv"
RAW_CLASS_COUNTS_PATH        = RESULTS_DIR / "raw_class_counts_by_source.csv"
DUPLICATES_PATH              = RESULTS_DIR / "duplicate_files.csv"
DUPLICATE_SUMMARY_PATH       = RESULTS_DIR / "duplicate_summary.csv"

# Target model input size: 224x224 for ResNet/ImageNet conventions and ViT defaults
IMG_SIZE = (224, 224)

# The four dataset class folder names used in the Kaggle dataset
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]

# Reproducibility for the Train/Val split
RANDOM_STATE = 42

# Kaggle-aligned split:
# - Kaggle Training -> Train/Val split (stratified)
# - Kaggle Testing  -> Test set (held-out)
VAL_FRACTION_OF_TRAINING = 0.20   # 80/20 split inside Kaggle Training (Same as Krishnan et al 2024)

# Dedup policy:
# If an exact duplicate exists in BOTH Training and Testing, keep Testing copy, it's the safer choice for leakage prevention
# (Training is more likely to be used in data augmentation or be accidentally included in validation
# if creating a new test split from Training).
PREFER_TESTING_ON_CROSS_SPLIT_DUPLICATES = True

# Tight crop parameters:
# - threshold: pixels darker than this are treated as background
# - margin: extra pixels around the detected brain bounding box
BACKGROUND_INTENSITY_THRESHOLD = 5
CROP_MARGIN = 10

# Only process real image files (prevents .DS_Store, Thumbs.db, etc.)
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _get_resample():
    
    """
    Pillow version compatibility: Image.Resampling exists in newer versions.
    I pick LANCZOS for best downsampling quality.
    """
    # Newer Pillow: Image.Resampling.LANCZOS exists
    try:
        return Image.Resampling.LANCZOS
    # Older Pillow: fallback to Image.LANCZOS
    except AttributeError:
        return Image.LANCZOS


# Chosen resampling method used for all resizing operations
RESAMPLE = _get_resample()


# COLLECTING RAW IMAGE PATHS


def collect_images() -> pd.DataFrame:
    
    """
    Walking through the raw Kaggle folders and building a DataFrame.

    For each file record:
      - orig_path    : full path on disk
      - class        : glioma / meningioma / pituitary / notumor
      - source_split : training or testing (Kaggle's folder)
    """
    # records accumulates one dict per image file found
    records = []

    # Kaggle dataset contains two main split folders: Training/ and Testing/
    for split in ["Training", "Testing"]:
        # Each split has class subfolders (glioma, meningioma, etc.)
        for cls in CLASSES:
            folder = RAW_ROOT / split / cls

            # If folder missing, skip (keeps script robust to partial datasets)
            if not folder.exists():
                continue

            # Iterating through files in the class folder
            for p in folder.iterdir():
                # Skipping directories; keep only files
                if not p.is_file():
                    continue

                # Skipping hidden/system files
                if p.name.startswith("."):
                    continue

                # Skipping non-image extensions
                if p.suffix.lower() not in ALLOWED_EXTS:
                    continue

                # Recording file path, label and source split
                records.append(
                    {
                        "orig_path": str(p),               # absolute/relative string path
                        "class": cls,                      # folder name = class label
                        "source_split": split.lower(),     # training / testing
                    }
                )

    # Converting list of dicts to a DataFrame for easy grouping/filtering later
    df = pd.DataFrame(records)

    # Immediate visibility: how many raw files exist before deduplication
    print(f"Total images found (before deduplication): {len(df)}")
    return df


def save_raw_class_counts(df: pd.DataFrame):
    
    """
    Summarising how many images there are per class in Kaggle's
    original Training vs Testing folders.

    Output: results/raw_class_counts_by_source.csv
    """
    # Ensure results/ exists
    RAW_CLASS_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Defensive: if nothing loaded, skip writing
    if df.empty:
        print("No images found; raw class counts not saved.")
        return

    # Counting rows grouped by (source_split, class)
    counts = (
        df.groupby(["source_split", "class"])
        .size()
        .reset_index(name="count")
        .sort_values(["source_split", "class"])
    )

    # Saving to CSV for my report
    counts.to_csv(RAW_CLASS_COUNTS_PATH, index=False)

    # Logging output for verification
    print(f"Saved raw class counts (by Kaggle split) to {RAW_CLASS_COUNTS_PATH}")
    print(counts)



# EXACT DUPLICATE REMOVAL (SHA1) - LEAKAGE SAFE


def sha1_of_file(path: str, block_size: int = 65536) -> str:
    """
    Compute SHA1 hash of a file.
    If two files have the same SHA1, they are byte-for-byte identical.
    """
    # Creating SHA1 hasher object
    h = hashlib.sha1()

    # Opening file in binary mode (hash must reflect bytes)
    with open(path, "rb") as f:
        # Reading file in blocks (memory-safe for big files)
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)

    # Returning hex digest string
    return h.hexdigest()

def processed_filename_for(orig_path: str) -> str:
    
    """
    Building a unique, deterministic filename for the processed image
    to avoid collisions (same src.name in different folders).

    I used SHA1(file_bytes) and original extension.
    """
    # Converting to Path for suffix handling
    p = Path(orig_path)

    # Filename is SHA1 hash plus normalized extension
    return f"{sha1_of_file(str(p))}{p.suffix.lower()}"


def drop_duplicates_leakage_safe(df: pd.DataFrame):
    
    """
    Removing exact duplicate files based on SHA1, using a leakage-safe policy.

    Key rule:
      - If a duplicate group spans both Kaggle Training and Kaggle Testing,
        I KEEP the Testing copy and DROP the Training copy(s).

    Returns:
      - dedup_df : DataFrame of kept rows only (unique SHA1)
      - dups_df  : DataFrame of all rows that belong to duplicate groups,
                  including which file was kept/dropped
    """
    # If df empty, return empty results safely
    if df.empty:
        return df.copy(), pd.DataFrame()

    print("Computing SHA1 hashes for duplicate detection...")

    # Working on a copy to avoid mutating caller’s DataFrame
    df = df.copy()

    # Adding SHA1 column computed from file bytes for each path
    df["sha1"] = df["orig_path"].apply(sha1_of_file)

    # Tracking count before deduplication
    before = len(df)

    # Marking duplicates: duplicated(..., keep=False) flags all members of a dup group
    dup_mask = df.duplicated(subset="sha1", keep=False)

    # Extracting just the duplicate rows for auditing/reporting
    dups_df = df[dup_mask].copy()

    # If no duplicates detected, return df as-is
    if dups_df.empty:
        print("No duplicate files detected based on SHA1 hashes.")
        dedup_df = df.reset_index(drop=True)
        print(f"Unique files after deduplication: {len(dedup_df)}")
        return dedup_df, dups_df

    # keep_indices stores the single chosen representative row index per SHA1 group
    keep_indices = []

    # Iterating over each SHA1 group deterministically
    for sha1, group in dups_df.groupby("sha1"):
        # Sorting for deterministic behaviour across OS/filesystems
        g = group.sort_values(["source_split", "orig_path"]).copy()

        # Detecting whether this group contains any testing/training copies
        has_testing = (g["source_split"] == "testing").any()
        has_training = (g["source_split"] == "training").any()

        # Leakage-safe rule: if group spans training and testing, keep testing copy
        if PREFER_TESTING_ON_CROSS_SPLIT_DUPLICATES and has_testing and has_training:
            # Keeping ONE testing copy (smallest path ensures determinism)
            keep_row = g[g["source_split"] == "testing"].sort_values("orig_path").iloc[0]
        else:
            # Otherwise, keep ONE copy (smallest path overall)
            keep_row = g.sort_values("orig_path").iloc[0]

        # Storing original DataFrame index of chosen keep_row
        keep_indices.append(int(keep_row.name))

    # non_dup_df: all unique rows that are not part of any duplicate group
    non_dup_df = df[~dup_mask].copy()

    # kept_from_dups_df: one representative row per duplicate group
    kept_from_dups_df = df.loc[keep_indices].copy()

    # Concatenating uniques and chosen dup representatives -> final deduplicated df
    dedup_df = pd.concat([non_dup_df, kept_from_dups_df], axis=0).reset_index(drop=True)

    # Tracking after deduplication
    after = len(dedup_df)

    # Printing deduplication effect
    print(f"Unique files after deduplication: {after}")
    print(f"Removed {before - after} duplicate file entries (if any).")

    # Augmenting dups_df with keep/drop info for auditing
    dups_df = dups_df.copy()
    dups_df["kept"] = False

    # Marking which rows were kept (by original index)
    kept_set = set(keep_indices)
    dups_df.loc[dups_df.index.isin(kept_set), "kept"] = True

    # For each SHA1 group, storing the kept path (helps auditing in my report)
    kept_path_by_sha1 = (
        dups_df[dups_df["kept"]]
        .set_index("sha1")["orig_path"]
        .to_dict()
    )
    dups_df["kept_path_for_group"] = dups_df["sha1"].map(kept_path_by_sha1)

    return dedup_df, dups_df


def save_duplicate_summary(dups_df: pd.DataFrame):
    
    """
    Write detailed information about exact duplicates:

      - duplicate_files.csv   : full listing of all duplicate-group members,
                               including which one was kept
      - duplicate_summary.csv : one row per SHA1 group
    """
    # Ensuring results/ exists
    DUPLICATES_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If no duplicates, nothing to write
    if dups_df is None or dups_df.empty:
        print("No duplicate files to save.")
        return

    # Saving full listing of every file in every duplicate group
    dups_df.to_csv(DUPLICATES_PATH, index=False)
    print(f"Saved full duplicate listing to {DUPLICATES_PATH}")

    # Building a compact summary per SHA1 group (useful in report)
    rows = []
    for sha1, group in dups_df.groupby("sha1"):
        group = group.copy()

        # The kept row should be exactly one (by design)
        kept_rows = group[group["kept"]]
        kept_path = kept_rows["orig_path"].iloc[0] if not kept_rows.empty else ""

        # Flags:
        # - crosses_splits: duplicate appears in both training & testing
        # - crosses_classes: same bytes appear under different class labels (rare but important :!)
        crosses_splits = group["source_split"].nunique() > 1
        crosses_classes = group["class"].nunique() > 1

        # Assemble summary row
        rows.append(
            {
                "sha1": sha1,
                "n_files": len(group),
                "crosses_splits": bool(crosses_splits),
                "crosses_classes": bool(crosses_classes),
                "classes": ";".join(sorted(group["class"].unique())),
                "source_splits": ";".join(sorted(group["source_split"].unique())),
                "kept_path": kept_path,
                # including a small sample of dropped paths (prevents huge CSV cells)
                "dropped_paths_example": "; ".join(group[~group["kept"]]["orig_path"].head(5)),
            }
        )

    # Creating DataFrame sorted by largest duplicate groups first
    summary_df = pd.DataFrame(rows).sort_values("n_files", ascending=False)

    # Saving compact summary
    summary_df.to_csv(DUPLICATE_SUMMARY_PATH, index=False)

    # Printing preview for sanity check
    print(f"Saved duplicate summary to {DUPLICATE_SUMMARY_PATH}")
    print(summary_df.head())


# RAW IMAGE ANALYSIS (GEOMETRY and INTENSITY)


def analyze_raw_images(df: pd.DataFrame) -> pd.DataFrame:
    
    """
    Audit each deduplicated raw image before resizing.

    For each image record:
      - width, height, aspect_ratio
      - grayscale intensity stats: mean, std, min, max
      - a 'failed' flag if PIL couldn't read the file
      - conservative quality flags:
          too_dark, too_bright, low_contrast, suspect

    This function does NOT drop images; it only flags them.
    """
    # records accumulates one audit record per image file
    records = []
    print("Analysing raw image geometry and intensity statistics...")

    # Iterating through images with a progress bar
    for row in tqdm(df.to_dict(orient="records"), desc="Analysing raw images"):
        # Extracting metadata from row
        path = row["orig_path"]
        cls = row["class"]
        source_split = row["source_split"]
        sha1 = row.get("sha1", None)

        # Default values (filled in after reading)
        width = height = None
        aspect_ratio = None
        mean_intensity = std_intensity = None
        min_intensity = max_intensity = None
        failed = False

        try:
            # Opening image
            img = Image.open(path)

            # Fix orientation using EXIF metadata (avoids rotated stats)
            img = ImageOps.exif_transpose(img)

            # Forcing decode now so corrupt images fail inside try/except
            img.load()

            # Geometry
            width, height = img.size
            aspect_ratio = (width / height) if height else np.nan

            # Converting to grayscale and compute intensity stats
            gray = img.convert("L")
            arr = np.array(gray, dtype=np.float32)

            mean_intensity = float(arr.mean())
            std_intensity = float(arr.std())
            min_intensity = float(arr.min())
            max_intensity = float(arr.max())

        except Exception as e:
            # If PIL cannot open/decode image, flag failed
            print(f"Failed to analyse {path}: {e}")
            failed = True

        # Appending audit record regardless of failed or not
        records.append(
            {
                "orig_path": path,
                "class": cls,
                "source_split": source_split,
                "sha1": sha1,
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "mean_intensity": mean_intensity,
                "std_intensity": std_intensity,
                "min_intensity": min_intensity,
                "max_intensity": max_intensity,
                "failed": failed,
            }
        )

    # Converting list of audit dicts to DataFrame
    stats_df = pd.DataFrame(records)

    # Convenience mask: only consider quality flags if image loaded successfully
    not_failed = ~stats_df["failed"]

    # Conservative too dark flag:
    # - low mean AND low max suggests mostly black image
    stats_df["too_dark"] = (
        not_failed
        & (stats_df["mean_intensity"] < 15)
        & (stats_df["max_intensity"] < 60)
    )

    # Conservative too bright flag:
    # - very high mean AND high min suggests washed-out bright image
    stats_df["too_bright"] = (
        not_failed
        & (stats_df["mean_intensity"] > 240)
        & (stats_df["min_intensity"] > 200)
    )

    # Low contrast flag: standard deviation extremely small
    stats_df["low_contrast"] = not_failed & (stats_df["std_intensity"] < 5)

    # Combine quality issues into one suspect flag
    stats_df["suspect"] = (
        stats_df["too_dark"]
        | stats_df["too_bright"]
        | stats_df["low_contrast"]
        | stats_df["failed"]
    )

    return stats_df


def save_raw_analysis(stats_df: pd.DataFrame):
    
    """
    Saving raw-image analysis to:
      - raw_image_stats.csv
      - raw_resolution_summary.csv
      - raw_quality_flags_summary.csv
    """
    # Ensuring results/ exists
    RAW_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Defensive: nothing to save
    if stats_df.empty:
        print("No stats to save; stats_df is empty.")
        return

    # Saving full per-image stats
    stats_df.to_csv(RAW_STATS_PATH, index=False)
    print(f"Saved raw image stats to {RAW_STATS_PATH}")

    # Summarising resolutions (width,height) frequencies
    res_summary = (
        stats_df.groupby(["width", "height"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    res_summary.to_csv(RAW_RESOLUTION_SUMMARY_PATH, index=False)
    print(f"Saved resolution summary to {RAW_RESOLUTION_SUMMARY_PATH}")
    print("Top 5 most common resolutions:")
    print(res_summary.head())

    # Summarising quality flags per class by summing boolean columns
    qual_summary = (
        stats_df.groupby("class")[["too_dark", "too_bright", "low_contrast", "suspect", "failed"]]
        .sum()
        .reset_index()
    )
    qual_summary.to_csv(RAW_QUALITY_SUMMARY_PATH, index=False)
    print(f"Saved quality flags summary to {RAW_QUALITY_SUMMARY_PATH}")
    print(qual_summary)

    # Printing total suspect count for quick inspection
    total_suspect = int(stats_df["suspect"].sum())
    print(f"Total suspect images (any flag or failed): {total_suspect}")



# TIGHT BRAIN CROPPING - SIMILAR TO KAGGLE SPLITS - RESIZED IMAGES


def tight_crop_to_brain(img: Image.Image) -> Image.Image:
    
    """
    Crop away black background around the brain using a simple intensity mask.
    If no foreground pixels are found, return the original image.
    """
    # Converting to grayscale for background detection (intensity-based)
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    # Foreground mask: pixels above the threshold are treated as content
    mask = arr > BACKGROUND_INTENSITY_THRESHOLD

    # If mask has no foreground pixels, cropping would be unsafe -> return original
    if not mask.any():
        return img

    # Finding coordinates of all foreground pixels
    ys, xs = np.where(mask)

    # Building bounding box around foreground, with margin, clamped to image bounds
    y_min = max(int(ys.min()) - CROP_MARGIN, 0)
    y_max = min(int(ys.max()) + 1 + CROP_MARGIN, arr.shape[0])

    x_min = max(int(xs.min()) - CROP_MARGIN, 0)
    x_max = min(int(xs.max()) + 1 + CROP_MARGIN, arr.shape[1])

    # Cropping uses (left, upper, right, lower)
    return img.crop((x_min, y_min, x_max, y_max))


def make_splits_kaggle_aligned(df: pd.DataFrame):
    
    """
    Creating splits:

      - Train/Val are created ONLY from Kaggle Training images (stratified).
      - Test is the Kaggle Testing folder (held-out, no splitting).

    This is the correct evaluation setup for this Kaggle dataset:
    I do not create a new test set from Training because Kaggle already provides one.
    """
    # Can't split empty dataset
    if df.empty:
        raise ValueError("Cannot split an empty dataset.")

    # Separating sources exactly as Kaggle intended
    df_train_source = df[df["source_split"] == "training"].copy()
    df_test_source  = df[df["source_split"] == "testing"].copy()

    # Training must exist; testing may be absent (warn, but allow)
    if df_train_source.empty:
        raise ValueError("No Kaggle Training images found after deduplication.")
    if df_test_source.empty:
        print("WARNING: No Kaggle Testing images found after deduplication. Test set will be empty.")

    # X = paths, y = class labels (used for stratification)
    X = df_train_source["orig_path"].values
    y = df_train_source["class"].values

    # Stratified train/val split inside Kaggle Training only
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=VAL_FRACTION_OF_TRAINING,
            stratify=y,
            random_state=RANDOM_STATE,
        )
    except Exception as e:
        
        # Robust fallback: if stratify fails, still split deterministically
        print(f"WARNING: Stratified split failed ({e}). Falling back to non-stratified split.")
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=VAL_FRACTION_OF_TRAINING,
            random_state=RANDOM_STATE,
        )

    # Helper: build DataFrame for each split with consistent columns
    def to_df(paths, labels, split_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "orig_path": [str(Path(p)) for p in paths],
                "class": labels,
                "split": split_name,
            }
        )

    # Train/Val from Kaggle Training
    df_train = to_df(X_train, y_train, "train")
    df_val   = to_df(X_val, y_val, "val")

    # Test from Kaggle Testing (held-out)
    df_test = pd.DataFrame(
        {
            "orig_path": df_test_source["orig_path"].astype(str).values,
            "class": df_test_source["class"].values,
            "split": "test",
        }
    )

    # Printing split sizes for verification
    print("Split sizes (Kaggle-aligned, after deduplication):")
    print(f"  Train (from Kaggle Training): {len(df_train)}")
    print(f"  Val   (from Kaggle Training): {len(df_val)}")
    print(f"  Test  (Kaggle Testing):       {len(df_test)}")

    return df_train, df_val, df_test


def prepare_processed_dirs():
    
    """
    Ensuring canonical directory structure exists for the cropped variant:

        data/processed/tightcrop/train/{class}/
        data/processed/tightcrop/val/{class}/
        data/processed/tightcrop/test/{class}/
    """
    # Creating all split/class directories ahead of processing
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            out_dir = PROCESSED_ROOT / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)


def resize_and_copy(df_split: pd.DataFrame, split_name: str):
    
    """
    Creating the processed dataset:

      - reading raw image
      - exif_transpose (safety)
      - tight cropping around brain
      - resizing to IMG_SIZE (224x224) using LANCZOS
      - converting to RGB
      - saving into data/processed/tightcrop/{split}/{class}/

    Files are saved using SHA1-based filenames to prevent collisions.
    """
    # Converting split DataFrame to list of dict records for faster iteration
    rows = df_split.to_dict(orient="records")

    # Iterating through every row with progress bar
    for row in tqdm(rows, desc=f"Processing {split_name} [{VARIANT}]"):
        # Source raw image path
        src = Path(row["orig_path"])

        # Class label determines output folder
        cls = row["class"]

        # Deterministic output filename based on SHA1(file_bytes)
        out_name = processed_filename_for(row["orig_path"])

        # Output destination path
        dst = PROCESSED_ROOT / split_name / cls / out_name

        # If already processed, skip (allows re-running pipeline cheaply)
        if dst.exists():
            continue

        try:
            # Load raw image
            img = Image.open(src)

            # Fix EXIF orientation issues
            img = ImageOps.exif_transpose(img)

            # Converting to RGB before crop/resize (ensures consistent 3 channels)
            img = img.convert("RGB")

            # Tight cropping to remove background
            img = tight_crop_to_brain(img)

            # Resizing to model input size using chosen resampler
            img = img.resize(IMG_SIZE, resample=RESAMPLE)

            # Saving processed image
            img.save(dst)
        except Exception as e:
            # Failure is logged, but pipeline continues (robust batch processing)
            print(f"Failed to process {src}: {e}")



def save_csv_splits(df_train, df_val, df_test):
    
    """
    Building CSVs that map to the processed (cropped) paths.

    Each CSV has columns:
      - image_path : path to cropped 224x224 RGB image
      - class      : tumour class label
    """
    # Ensuring data/splits/tightcrop exists
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)

    # Helper: map each raw orig_path to the corresponding processed output path
    def map_to_processed(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        processed_paths = []
        for _, row in df.iterrows():
            cls = row["class"]
            out_name = processed_filename_for(row["orig_path"])
            processed_paths.append(str(PROCESSED_ROOT / split_name / cls / out_name))
        return pd.DataFrame({"image_path": processed_paths, "class": df["class"].values})

    # Building the three CSVs
    train_csv = map_to_processed(df_train, "train")
    val_csv   = map_to_processed(df_val, "val")
    test_csv  = map_to_processed(df_test, "test")

    # Writing them to disk
    train_csv.to_csv(SPLITS_ROOT / "train.csv", index=False)
    val_csv.to_csv(SPLITS_ROOT / "val.csv", index=False)
    test_csv.to_csv(SPLITS_ROOT / "test.csv", index=False)

    print(f"Saved split CSVs to {SPLITS_ROOT}.")


def save_summary(df_train, df_val, df_test):
    
    """
    Summarising class counts per split.

    Output:
      - results/dataset_summary.csv
    """
    # Ensuring results/ exists
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Helper: class counts for one split
    def counts(df, split_name):
        c = df["class"].value_counts().rename("count").reset_index()
        c = c.rename(columns={"index": "class"})
        c.insert(0, "split", split_name)
        return c

    # Concatenating train, val and test counts into a single summary table
    summary_df = pd.concat(
        [counts(df_train, "train"), counts(df_val, "val"), counts(df_test, "test")],
        axis=0,
        ignore_index=True,
    )

    # Save summary
    summary_df.to_csv(SUMMARY_PATH, index=False)
    print("Saved dataset summary to results/dataset_summary.csv")
    print(summary_df)

# MAIN ORCHESTRATION

def main():
    
    """
    entire dataset preparation pipeline:

      1. Collecting raw images from Kaggle folders
      2. Saving raw class counts per Kaggle split
      3. Deduplicating based on SHA1 with leakage-safe policy (prefer keep Testing on cross-split dups)
      4. Analyzing raw images (geometry + intensity + quality flags)
      5. Saving analysis CSVs
      6. Creating Kaggle-aligned splits:
           - Train/Val from Kaggle Training (stratified)
           - Test from Kaggle Testing (held-out)
      7. Writing cropped 224x224 RGB images into data/processed/tightcrop/
      8. Saving train/val/test CSVs and overall summary table
    """
    # print for run logs
    print(f"Running dataset preparation for VARIANT = '{VARIANT}' (cropped-only pipeline)")

    # Step 1: collecting raw image paths
    df_raw = collect_images()

    # Step 2: raw class counts per Kaggle split
    save_raw_class_counts(df_raw)

    # Step 3: exact duplicates (SHA1) with leakage-safe policy
    df_dedup, dups_df = drop_duplicates_leakage_safe(df_raw)
    save_duplicate_summary(dups_df)

    # Step 4: analysing raw images
    stats_df = analyze_raw_images(df_dedup)

    # Step 5: saving analysis artefacts
    save_raw_analysis(stats_df)

    # Step 6: splits (paper/evaluation correct)
    df_train, df_val, df_test = make_splits_kaggle_aligned(df_dedup)

    # Step 7: preparing dirs and writing processed images
    prepare_processed_dirs()
    resize_and_copy(df_train, "train")
    resize_and_copy(df_val, "val")
    resize_and_copy(df_test, "test")

    # Step 8: saving split CSVs and summary
    save_csv_splits(df_train, df_val, df_test)
    save_summary(df_train, df_val, df_test)

    # Finished
    print("Dataset preparation and audit completed (cropped 224x224 images only).")



# allows importing functions without running pipeline automatically
if __name__ == "__main__":
    main()
    
### THE COMMENTS CAN BE UNORGANIZED AT PLACES BECAUSE IT THE IMPLEMENTATION TOOK QUITE SOME TIME
### SO PLEASE BARE WITH ME