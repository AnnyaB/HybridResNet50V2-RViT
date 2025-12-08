# I started working on dataset preparation from 12th november 2025

# -------------------------------------------------------------------
# dataset preparation pipeline (what I already did)
#
# In this script I had:
#   - walked the original Kaggle brain tumour MRI folders and recorded
#     every file path together with its class and Kaggle split
#     (Training / Testing),
#   - computed SHA1 hashes to detect and remove exact byte-for-byte
#     duplicate files, while also saving a detailed duplicate summary,
#   - audited every deduplicated raw image for geometry and intensity:
#       – width, height, aspect ratio,
#       – mean / std / min / max grayscale intensity,
#       – conservative flags for too dark, too bright, low contrast,
#         and a combined "suspect" quality flag,
#   - produced CSV summaries for raw resolutions and quality flags so
#     that I could describe the original dataset objectively in the
#     report,
#   - optionally run a perceptual near-duplicate audit using pHash:
#       - computed a pHash for each non-failed image,
#       - searched for near-duplicate pairs based on Hamming distance,
#       - grouped them into clusters,
#       - wrote CSVs and example montages (for qualitative inspection),
#   - created a stratified 70/15/15 train/val/test split after
#     exact-duplicate removal, stratifying by tumour class,
#   - defined a tight brain-cropping function that removed black
#     background based on an intensity threshold plus a small margin,
#   - applied this crop to every image in the split, resized to
#     224x224 (matching ImageNet / ResNet50V2 / RViT defaults),
#     converted to RGB, and saved them into a clean
#     data/processed/tightcrop/{split}/{class}/ structure,
#  -  saved train/val/test CSVs that point to these cropped 224×224
#     images (for model training and evaluation),
#   - saved an overall split summary CSV with per-class counts per
#     split, which I later used to generate figures in dataset_plots.py.
#
# This file therefore gave me a single, reproducible, auditable
# pipeline from raw Kaggle folders -> cleaned, cropped 224x224 dataset
# and analysis artefacts for the project's methodology section.
# -------------------------------------------------------------------

"""
Dataset preparation script for HybridResNet50V2–RViT brain tumour classification.

In this file I built one reproducible pipeline that:

  - walks the raw Kaggle folders and records exactly what is there,
  - removes byte-for-byte duplicate files using SHA1,
  - audits the raw images for geometry and intensity problems
    (so I can talk about “suspect” images with evidence),
  - optionally runs a perceptual near-duplicate audit using pHash,
  - creates a stratified 70/15/15 split AFTER deduplication,
  - and finally writes *cropped* 224×224 RGB images and CSVs for model training.

Important design decisions:

  - All model training uses only the cropped 224x224 images.
  - Cropping is done first (tight crop around brain region), then resized to 224x224.
  - The audit (intensity, duplicates, near-duplicates) is always done on the raw
    uncropped images so that I can describe the original dataset properly.
"""

# I imported all standard and third-party libraries that I needed along the pipeline.

import os                               # I kept this for any simple filesystem helpers if needed.
from pathlib import Path                # I used Path for clean, OS-independent path handling.
import hashlib                          # I used hashlib.SHA1 for exact duplicate detection.
from collections import defaultdict, deque  # I used these to build adjacency lists and BFS for pHash clusters.
import math                             # I used math.ceil when laying out montage grids.

import numpy as np                      # I used NumPy for numerical operations on image arrays.
from PIL import Image                   # I used Pillow to open, crop and resize images.
from sklearn.model_selection import train_test_split  # I used this to create stratified 70/15/15 splits.
import pandas as pd                     # I used pandas for tabular data manipulation and CSV I/O.
from tqdm import tqdm                   # I used tqdm for progress bars over large image loops.
import imagehash                        # I used imagehash.phash for perceptual near-duplicate auditing.


# -------------------------------------------------------------------
# CONFIGURATION SECTION
# -------------------------------------------------------------------
# Here I fixed all paths, constants and hyperparameters so that any
# change in layout or image size only required edits in one place.

# I set the project root as the folder above scripts/ using Path logic.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# I defined the root of the raw Kaggle dataset following my folder layout:
# brain_tumor_project/data/raw/brain-tumor-mri-dataset/kaggle_brain_mri_scan_dataset/
RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "brain-tumor-mri-dataset"
    / "kaggle_brain_mri_scan_dataset"
)

# I pointed PROCESSED_BASE to where I stored cropped and resized images.
PROCESSED_BASE = PROJECT_ROOT / "data" / "processed"

# I pointed SPLITS_BASE to where I stored train/val/test CSVs.
SPLITS_BASE = PROJECT_ROOT / "data" / "splits"

# I chose a label for this preprocessing run. I kept tightcrop to
# emphasise that all images had tight brain cropping.

VARIANT = "tightcrop"  # final training dataset: cropped brain and 224×224.

# I defined the final root directory where this variant wrote its images.
PROCESSED_ROOT = PROCESSED_BASE / VARIANT
# I defined the final root directory where this variant wrote its CSV splits.
SPLITS_ROOT = SPLITS_BASE / VARIANT

# I pointed RESULTS_DIR to the folder that would hold all analysis artefacts.
RESULTS_DIR = PROJECT_ROOT / "results"

# I defined CSV output paths for dataset summaries and raw analysis.
SUMMARY_PATH                 = RESULTS_DIR / "dataset_summary.csv"
RAW_STATS_PATH               = RESULTS_DIR / "raw_image_stats.csv"
RAW_RESOLUTION_SUMMARY_PATH  = RESULTS_DIR / "raw_resolution_summary.csv"
RAW_QUALITY_SUMMARY_PATH     = RESULTS_DIR / "raw_quality_flags_summary.csv"
RAW_CLASS_COUNTS_PATH        = RESULTS_DIR / "raw_class_counts_by_source.csv"
DUPLICATES_PATH              = RESULTS_DIR / "duplicate_files.csv"
DUPLICATE_SUMMARY_PATH       = RESULTS_DIR / "duplicate_summary.csv"

# I defined the paths for perceptual near-duplicate (pHash) outputs.
NEAR_DUP_PAIRS_PATH          = RESULTS_DIR / "near_duplicate_pairs.csv"
NEAR_DUP_CLUSTERS_PATH       = RESULTS_DIR / "near_duplicate_clusters_summary.csv"
NEAR_DUP_EXAMPLES_DIR        = RESULTS_DIR / "near_duplicate_examples"

# I fixed the target input size for both ResNet50V2 and RViT models.
# I used 224×224 because it matches ImageNet pretrained weights and standard ViT configs.
IMG_SIZE = (224, 224)  # Every processed image was resized to this H×W.

# I listed the canonical class labels exactly as in the Kaggle folder names.
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]

# I fixed a random seed to make my stratified splits reproducible.
RANDOM_STATE = 42

# I configured pHash near-duplicate auditing: threshold, number of montages, and on/off flag.
PHASH_THRESHOLD    = 5           # maximum Hamming distance to treat as near-duplicate.
TOP_K_MONTAGES     = 10          # number of clusters for which I produced montage images.
RUN_NEAR_DUP_AUDIT = True        # I left this True so the expensive audit still ran.

# I set cropping settings for the tight brain cropping step.
# Any grayscale pixel intensity ≤ BACKGROUND_INTENSITY_THRESHOLD was treated as background.
BACKGROUND_INTENSITY_THRESHOLD = 5   # intensity threshold (0..255) to detect background.
CROP_MARGIN = 10                     # extra pixels I left around the foreground bounding box.


# -------------------------------------------------------------------
# COLLECTING RAW IMAGE PATHS
# -------------------------------------------------------------------

def collect_images() -> pd.DataFrame:
    """
    Walk through the raw Kaggle folders and build a DataFrame
    with one row per file.

    For each file I record:
      - orig_path    : full path on disk,
      - class        : glioma / meningioma / pituitary / notumor,
      - source_split : "training" or "testing" (Kaggle's folder).
    """
    # I started with an empty list that I later transformed into a DataFrame.
    records = []

    # I looped over Kaggle's two top-level splits: Training and Testing.
    for split in ["Training", "Testing"]:
        # For each split, I looped over the four tumour classes.
        for cls in CLASSES:
            # I built the folder path for that split + class.
            folder = RAW_ROOT / split / cls
            # If the folder did not exist (for robustness), I simply skipped it.
            if not folder.exists():
                continue

            # I iterated over all entries inside that folder.
            for p in folder.iterdir():
                # I only cared about regular files (not subdirectories).
                if p.is_file():
                    # I appended a record with original path, class and the Kaggle split in lowercase.
                    records.append(
                        {
                            "orig_path": str(p),
                            "class": cls,
                            "source_split": split.lower(),
                        }
                    )

    # I converted the list of records into a pandas DataFrame.
    df = pd.DataFrame(records)
    # I printed how many images I had discovered before any deduplication.
    print(f"Total images found (before deduplication): {len(df)}")
    # I returned this raw listing for the next steps.
    return df


def save_raw_class_counts(df: pd.DataFrame):
    
    """
    Summarise how many images there are per class in Kaggle's
    original Training vs Testing folders.

    Output: results/raw_class_counts_by_source.csv
    """
    # I ensured the parent directory existed before saving.
    RAW_CLASS_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If the DataFrame was empty, I skipped saving and logged a message.
    if df.empty:
        print("No images found; raw class counts not saved.")
        return

    # I grouped by source_split and class and counted how many images each combination had.
    counts = (
        df.groupby(["source_split", "class"])
        .size()
        .reset_index(name="count")
        .sort_values(["source_split", "class"])
    )

    # I wrote the summary table to CSV for reuse in figures / report.
    counts.to_csv(RAW_CLASS_COUNTS_PATH, index=False)
    print(
        "Saved raw class counts (by Kaggle split) to "
        f"{RAW_CLASS_COUNTS_PATH}"
    )
    # I printed the counts for a quick sanity check in the console.
    print(counts)


# -------------------------------------------------------------------
# EXACT DUPLICATE REMOVAL (SHA1)
# -------------------------------------------------------------------

def sha1_of_file(path: str, block_size: int = 65536) -> str:
    """
    Compute a SHA1 hash of the file at 'path'.

    SHA1 is used purely as a fingerprint: if two files have the same
    SHA1, they are byte-for-byte identical.
    """
    # I initialised a new SHA1 hash object.
    h = hashlib.sha1()
    # I opened the file in binary mode so I could hash its raw bytes.
    with open(path, "rb") as f:
        # I read the file in chunks to avoid loading very large files fully into memory.
        for chunk in iter(lambda: f.read(block_size), b""):
            # I updated the hash with each chunk.
            h.update(chunk)
    # I returned the hexadecimal representation of the final SHA1 digest.
    return h.hexdigest()


def drop_duplicates(df: pd.DataFrame):
    
    """
    Remove exact duplicate files based on SHA1.

    Returns:
      - dedup_df : DataFrame where each SHA1 appears only once.
      - dups_df  : all rows that were part of a duplicate group.
    """
    # I told the user that SHA1 computation for duplicate detection had started. ( the user could be those who run the script for instance the supervisor or myself)
    print("Computing SHA1 hashes for duplicate detection...")
    # I computed a SHA1 hash for every original file path.
    df["sha1"] = df["orig_path"].apply(sha1_of_file)

    # I recorded how many rows I had before deduplication.
    before = len(df)
    # I extracted all rows that belonged to any duplicated SHA1 group.
    dups_df = df[df.duplicated(subset="sha1", keep=False)].copy()
    # I dropped duplicates so that each SHA1 appeared only once (keeping the first occurrence).
    dedup_df = df.drop_duplicates(subset="sha1").reset_index(drop=True)
    # I recorded how many unique rows remained after deduplication.
    after = len(dedup_df)

    # I logged the number of unique files after deduplication.
    print(f"Unique files after deduplication: {after}")
    # I logged how many duplicate entries were removed.
    print(f"Removed {before - after} duplicate file entries (if any).")

    # I returned the deduplicated DataFrame and the duplicates listing.
    return dedup_df, dups_df


def save_duplicate_summary(dups_df: pd.DataFrame):
    
    """
    Write detailed information about exact duplicates:

      - duplicate_files.csv   : full listing of all duplicate files.
      - duplicate_summary.csv : one row per hash group.
    """
    # I ensured the results directory existed.
    DUPLICATES_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If no duplicates were found, I simply logged this and returned.
    if dups_df.empty:
        print("No duplicate files detected based on SHA1 hashes.")
        return

    # I saved the full duplicate listing to CSV.
    dups_df.to_csv(DUPLICATES_PATH, index=False)
    print(f"Saved full duplicate listing to {DUPLICATES_PATH}")

    # I built a per-hash summary table.
    rows = []
    # I iterated over each group of files sharing the same SHA1.
    for sha1, group in dups_df.groupby("sha1"):
        # For each group, I recorded the number of files and some metadata.
        rows.append(
            {
                "sha1": sha1,
                "n_files": len(group),
                "classes": ";".join(sorted(group["class"].unique())),
                "source_splits": ";".join(sorted(group["source_split"].unique())),
                "example_paths": "; ".join(group["orig_path"].head(3)),
            }
        )
    # I built a DataFrame from all summarized groups and sorted by cluster size.
    summary_df = pd.DataFrame(rows).sort_values("n_files", ascending=False)
    # I saved the summary to CSV.
    summary_df.to_csv(DUPLICATE_SUMMARY_PATH, index=False)
    print(f"Saved duplicate summary to {DUPLICATE_SUMMARY_PATH}")
    # I printed the head of the summary for a console sanity check.
    print(summary_df.head())


# -------------------------------------------------------------------
# RAW IMAGE ANALYSIS (GEOMETRY and INTENSITY)
# -------------------------------------------------------------------

def analyze_raw_images(df: pd.DataFrame) -> pd.DataFrame:
    """
    Audit each deduplicated raw image before resizing.

    For each image I computed and recorded:
      - width, height, aspect_ratio
      - grayscale intensity stats: mean, std, min, max
      - a 'failed' flag if PIL couldn't read the file
      - conservative quality flags:
          too_dark, too_bright, low_contrast, suspect

    I didn't drop these images here; I only flag them.
    """
    # I collected one record per image in this list.
    records = []

    # I logged that I was starting the analysis.
    print("Analysing raw image geometry and intensity statistics...")
    # I iterated over the deduplicated image metadata using tqdm for a progress bar.
    for row in tqdm(
        df.to_dict(orient="records"), desc="[INFO] Analysing raw images"
    ):
        # I unpacked fields from the row for convenience.
        path = row["orig_path"]
        cls = row["class"]
        source_split = row["source_split"]
        sha1 = row["sha1"]

        # I initialised geometry and intensity fields to None (or NaN later).
        width = height = None
        aspect_ratio = None
        mean_intensity = std_intensity = None
        min_intensity = max_intensity = None
        failed = False

        try:
            # I tried to open the image with Pillow.
            img = Image.open(path)

            # I extracted width and height from the PIL image.
            width, height = img.size
            # I computed aspect ratio if height was valid and non-zero.
            if height is not None and height != 0:
                aspect_ratio = width / height
            else:
                # If height was invalid, I stored NaN for aspect_ratio.
                aspect_ratio = np.nan

            # I converted the image to grayscale for intensity statistics.
            gray = img.convert("L")
            # I converted the grayscale image into a float32 NumPy array.
            arr = np.array(gray, dtype=np.float32)

            # I computed mean intensity of the whole image.
            mean_intensity = float(arr.mean())
            # I computed the standard deviation of intensities (contrast proxy).
            std_intensity = float(arr.std())
            # I recorded the minimum pixel intensity.
            min_intensity = float(arr.min())
            # I recorded the maximum pixel intensity.
            max_intensity = float(arr.max())

        except Exception as e:
            # If anything went wrong (e.g. corrupt image), I logged and marked it as failed.
            print(f"Failed to analyse {path}: {e}")
            failed = True

        # I appended a dictionary with all computed fields for this image.
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

    # I converted all collected records into a DataFrame.
    stats_df = pd.DataFrame(records)

    # I built a boolean mask for images that did not fail to load.
    not_failed = ~stats_df["failed"]

    # I defined a "too_dark" condition based on conservative thresholds.
    stats_df["too_dark"] = (
        not_failed
        & (stats_df["mean_intensity"] < 15)
        & (stats_df["max_intensity"] < 60)
    )

    # I defined a "too_bright" condition for images that were almost saturated.
    stats_df["too_bright"] = (
        not_failed
        & (stats_df["mean_intensity"] > 240)
        & (stats_df["min_intensity"] > 200)
    )

    # I flagged low-contrast images based on very low intensity standard deviation.
    stats_df["low_contrast"] = not_failed & (stats_df["std_intensity"] < 5)

    # I combined all quality issues and the failure flag into a single "suspect" flag.
    stats_df["suspect"] = (
        stats_df["too_dark"]
        | stats_df["too_bright"]
        | stats_df["low_contrast"]
        | stats_df["failed"]
    )

    # I returned the stats DataFrame so later steps and plots could use it.
    return stats_df


def save_raw_analysis(stats_df: pd.DataFrame):
    """
    Save the raw-image analysis to several CSVs:

      - raw_image_stats.csv
      - raw_resolution_summary.csv
      - raw_quality_flags_summary.csv
    """
    # I ensured the results directory existed before saving.
    RAW_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    # If there was nothing to save, I logged and returned early.
    if stats_df.empty:
        print("No stats to save; stats_df is empty.")
        return

    # I wrote the full per-image statistics to CSV.
    stats_df.to_csv(RAW_STATS_PATH, index=False)
    print(f"Saved raw image stats to {RAW_STATS_PATH}")

    # I summarised the distribution of raw resolutions (width × height).
    res_summary = (
        stats_df.groupby(["width", "height"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    # I saved the resolution summary to CSV.
    res_summary.to_csv(RAW_RESOLUTION_SUMMARY_PATH, index=False)
    print(f"Saved resolution summary to {RAW_RESOLUTION_SUMMARY_PATH}")
    print("Top 5 most common resolutions:")
    # I printed the top 5 resolutions for quick inspection.
    print(res_summary.head())

    # I summarised how many images per class had each quality flag.
    qual_summary = (
        stats_df.groupby("class")[
            ["too_dark", "too_bright", "low_contrast", "suspect", "failed"]
        ]
        .sum()
        .reset_index()
    )
    # I saved the quality flags summary to CSV.
    qual_summary.to_csv(RAW_QUALITY_SUMMARY_PATH, index=False)
    print(f"Saved quality flags summary to {RAW_QUALITY_SUMMARY_PATH}")
    # I printed the summary so I could see per-class distributions.
    print(qual_summary)

    # I computed the total number of suspect images overall.
    total_suspect = int(stats_df["suspect"].sum())
    print(f"Total suspect images (any flag or failed): {total_suspect}")


# -------------------------------------------------------------------
# PERCEPTUAL NEAR-DUPLICATE AUDIT (pHash)
# -------------------------------------------------------------------

def compute_phash(path: str):
    """
    Compute a perceptual hash (pHash) for a given image.

    Used only for auditing near-duplicates; does not change training data.
    """
    try:
        # I opened the image and converted it to grayscale for pHash.
        img = Image.open(path).convert("L")
        # I computed the perceptual hash using imagehash's phash implementation.
        h = imagehash.phash(img)
        # I returned the hash object; it supported Hamming distance operations.
        return h
    except Exception as e:
        # If pHash computation failed, I logged a warning and returned None.
        print(f"Failed to compute pHash for {path}: {e}")
        return None


def build_clusters(pairs_df: pd.DataFrame):
    """
    Given a DataFrame of near-duplicate pairs (i, j), build connected
    components (clusters) using a simple BFS.
    """
    # I created an adjacency list mapping each index to its neighbours.
    adj = defaultdict(set)

    # I populated the adjacency list from the near-duplicate pairs.
    for _, row in pairs_df.iterrows():
        i = int(row["idx1"])
        j = int(row["idx2"])
        adj[i].add(j)
        adj[j].add(i)

    # I tracked which indices I had already visited.
    visited = set()
    # I stored each connected component (cluster) in a list.
    clusters = []

    # I iterated over all nodes that had at least one neighbour.
    for i in adj.keys():
        # I skipped nodes that were already part of a discovered cluster.
        if i in visited:
            continue
        # I started a new component list.
        comp = []
        # I initialised a BFS queue from this node.
        q = deque([i])
        # I marked the starting node as visited.
        visited.add(i)
        # I performed BFS to collect the full connected component.
        while q:
            u = q.popleft()
            comp.append(u)
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    q.append(v)
        # I added the sorted list of indices for this cluster.
        clusters.append(sorted(comp))

    # I returned all discovered clusters.
    return clusters


def save_cluster_montage(df: pd.DataFrame, cluster_indices, cluster_id: int):
    """
    For each near-duplicate cluster, build a montage and save it
    to results/near_duplicate_examples/cluster_XXX.png.
    """
    # I made sure the montage output directory existed.
    NEAR_DUP_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # I collected resized PIL images for this cluster.
    imgs = []
    for idx in cluster_indices:
        # I looked up the original path of this cluster member.
        path = df.at[idx, "orig_path"]
        try:
            # I opened the image and converted it to RGB.
            img = Image.open(path).convert("RGB")
            # I resized it to a standard thumbnail size.
            img = img.resize((224, 224))
            # I added it to the list of thumbnails.
            imgs.append(img)
        except Exception as e:
            # If a file failed to load, I logged a warning and skipped it.
            print(f"Failed to load image for montage {path}: {e}")

    # If no images were successfully loaded, I aborted montage creation.
    if not imgs:
        return None

    # I computed how many thumbnails I had in this cluster.
    n = len(imgs)
    # I limited the number of columns to at most 5 for readability.
    cols = min(n, 5)
    # I computed how many rows I needed given that many columns.
    rows = math.ceil(n / cols)

    # I fetched the size of an individual thumbnail.
    w, h = imgs[0].size
    # I created a blank canvas large enough for all thumbnails in a grid.
    canvas = Image.new("RGB", (cols * w, rows * h))

    # I pasted each thumbnail into its grid position.
    for k, img in enumerate(imgs):
        r = k // cols
        c = k % cols
        canvas.paste(img, (c * w, r * h))

    # I built the output path for this cluster's montage.
    out_path = NEAR_DUP_EXAMPLES_DIR / f"cluster_{cluster_id:03d}.png"
    # I saved the composed montage image.
    canvas.save(out_path)
    # I returned the path as a string so it could be written into the summary CSV.
    return str(out_path)


def run_near_duplicate_audit(stats_df: pd.DataFrame):
    """
    Perform a pHash-based near-duplicate audit on the raw images.

    This does NOT remove any images; it just produces CSVs and
    montages for the FYP dataset section.
    """
    # I worked on a copy of the stats DataFrame to avoid modifying the original.
    df = stats_df.copy()

    # If a failed column was present, I removed failed images from the pHash audit.
    if "failed" in df.columns:
        df = df[~df["failed"]].copy()

    # If there were no images left, I logged and exited.
    if df.empty:
        print("No images available for near-duplicate audit.")
        return

    # I logged how many images I was about to pHash.
    print(f"Computing pHash for {len(df)} images...")
    # I computed and stored a pHash for every remaining image, with a progress bar.
    df["phash"] = [
        compute_phash(p) for p in tqdm(df["orig_path"], desc="[INFO] pHash")
    ]

    # I recorded how many rows I had before filtering out failed pHashes.
    before = len(df)
    # I removed any rows where pHash computation returned None.
    df = df[df["phash"].notnull()].reset_index(drop=True)
    # I computed how many images were dropped due to pHash failure.
    dropped = before - len(df)
    print(f"[INFO] Dropped {dropped} images with failed pHash computation.")

    # I stored how many images remained.
    n = len(df)
    # If none remained, I logged and skipped the rest of the audit.
    if n == 0:
        print(
            "No images left after pHash filtering; "
            "skipping near-duplicate audit."
        )
        return

    # I logged that I was going to search for near-duplicate pairs.
    print(
        f"Searching for near-duplicate pairs "
        f"(threshold={PHASH_THRESHOLD}) among {n} images..."
    )

    # I collected all near-duplicate pairs into this list of dictionaries.
    pairs = []

    # I used a double loop over indices i < j to compute pairwise Hamming distances.
    for i in tqdm(range(n), desc="near-duplicate search"):
        hi = df.at[i, "phash"]
        for j in range(i + 1, n):
            hj = df.at[j, "phash"]
            # The pHash objects supported Hamming distance via subtraction.
            dist = hi - hj
            # If the distance was below or equal to my threshold, I recorded the pair.
            if dist <= PHASH_THRESHOLD:
                pairs.append(
                    {
                        "idx1": i,
                        "idx2": j,
                        "hamming_dist": int(dist),
                        "orig_path_1": df.at[i, "orig_path"],
                        "orig_path_2": df.at[j, "orig_path"],
                        "class_1": df.at[i, "class"],
                        "class_2": df.at[j, "class"],
                        "source_split_1": df.at[i, "source_split"],
                        "source_split_2": df.at[j, "source_split"],
                    }
                )

    # If I found no near-duplicate pairs at this threshold, I logged and stopped.
    if not pairs:
        print(
            f"No near-duplicate pairs found "
            f"(threshold={PHASH_THRESHOLD})."
        )
        return

    # I converted the list of pairs to a DataFrame.
    pairs_df = pd.DataFrame(pairs)
    # I ensured the directory existed before saving the pairs.
    NEAR_DUP_PAIRS_PATH.parent.mkdir(parents=True, exist_ok=True)
    # I saved all near-duplicate pairs to CSV.
    pairs_df.to_csv(NEAR_DUP_PAIRS_PATH, index=False)
    print(f"Saved near-duplicate pairs to {NEAR_DUP_PAIRS_PATH}")
    # I printed the first few pairs for quick inspection.
    print(pairs_df.head())

    # I built connected components (clusters) from the pair graph.
    clusters = build_clusters(pairs_df)
    print(f"Built {len(clusters)} near-duplicate clusters.")

    # I prepared a list of summary rows describing each cluster.
    summary_rows = []
    # I iterated over all clusters with a 1-based cluster ID.
    for cid, comp in enumerate(clusters, start=1):
        # took the subset of df belonging to this cluster.
        subset = df.loc[comp]
        # recorded which classes appeared in this cluster.
        classes = ";".join(sorted(subset["class"].unique()))
        # I recorded which Kaggle splits appeared.
        splits = ";".join(sorted(subset["source_split"].unique()))
        # used the first path as a simple example reference.
        example_path = subset["orig_path"].iloc[0]

        # I appended one summary row for this cluster.
        summary_rows.append(
            {
                "cluster_id": cid,
                "cluster_size": len(comp),
                "classes": classes,
                "source_splits": splits,
                "example_path": example_path,
            }
        )

    # I built a DataFrame from the cluster summaries and sorted by size.
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "cluster_size", ascending=False
    )

    # For the top K largest clusters, I created and linked montage images.
    for _, row in summary_df.head(TOP_K_MONTAGES).iterrows():
        cid = int(row["cluster_id"])
        comp = clusters[cid - 1]
        montage_path = save_cluster_montage(df, comp, cid)
        if montage_path is not None:
            # I wrote the montage path into the summary DataFrame.
            summary_df.loc[
                summary_df["cluster_id"] == cid, "example_montage"
            ] = montage_path

    # I saved the final cluster summary to CSV.
    summary_df.to_csv(NEAR_DUP_CLUSTERS_PATH, index=False)
    print(
        f"Saved near-duplicate cluster summary to "
        f"{NEAR_DUP_CLUSTERS_PATH}"
    )
    # I printed the first few cluster summaries to the console.
    print(summary_df.head(10))


# -------------------------------------------------------------------
# TIGHT BRAIN CROPPING - STRATIFIED SPLITS - RESIZED IMAGES
# -------------------------------------------------------------------

def tight_crop_to_brain(img: Image.Image) -> Image.Image:
    
    """
    Crop away the black background around the brain.

    Steps:
      - convert the image to grayscale,
      - build a mask of foreground pixels with intensity > BACKGROUND_INTENSITY_THRESHOLD,
      - if there is any foreground, compute the tight bounding box,
      - expand that box by CROP_MARGIN pixels on each side (clamped),
      - crop the original RGB image to that box.

    If the image is completely empty (no foreground), return the original image.
    """
    # I converted the input image to grayscale for background detection.
    gray = img.convert("L")
    # I converted the grayscale image to a uint8 NumPy array.
    arr = np.array(gray, dtype=np.uint8)

    # I built a mask where True indicates potential brain pixels (above threshold).
    mask = arr > BACKGROUND_INTENSITY_THRESHOLD

    # If no foreground pixels existed, I returned the original image without cropping.
    if not mask.any():
        return img

    # I got the y and x coordinates of all foreground pixels.
    ys, xs = np.where(mask)

    # I computed the minimum and maximum y coordinate (and applied the margin).
    y_min = max(int(ys.min()) - CROP_MARGIN, 0)
    y_max = min(int(ys.max()) + 1 + CROP_MARGIN, arr.shape[0])
    # I computed the minimum and maximum x coordinate (and applied the margin).
    x_min = max(int(xs.min()) - CROP_MARGIN, 0)
    x_max = min(int(xs.max()) + 1 + CROP_MARGIN, arr.shape[1])

    # I cropped the original RGB image using the computed bounding box.
    cropped = img.crop((x_min, y_min, x_max, y_max))
    # I returned the cropped image.
    return cropped


def make_splits(df: pd.DataFrame):
    """
    Create a stratified 70/15/15 split after SHA1 deduplication.

    Splits are stratified by class, independent of Kaggle's
    original Training/Testing folders.
    """
    # I extracted the feature array as the original paths.
    X = df["orig_path"].values
    # I extracted the label array as the tumour classes.
    y = df["class"].values

    # I first split the data into 70% train and 30% temporary set.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # I then split the temporary set equally into validation and test (15% each).
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    # I defined a small helper that converted arrays into a DataFrame with a split label.
    def to_df(paths, labels, split_name: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "orig_path": [str(Path(p)) for p in paths],
                "class": labels,
                "split": split_name,
            }
        )

    # I created the DataFrames for train, val and test splits.
    df_train = to_df(X_train, y_train, "train")
    df_val = to_df(X_val, y_val, "val")
    df_test = to_df(X_test, y_test, "test")

    # I printed the sizes of each split so I could confirm the counts.
    print("Split sizes (after deduplication):")
    print(f"  Train: {len(df_train)}")
    print(f"  Val:   {len(df_val)}")
    print(f"  Test:  {len(df_test)}")

    # returned the three split DataFrames.
    return df_train, df_val, df_test


def prepare_processed_dirs():
    
    """
    Ensure the canonical directory structure exists for the cropped variant:

        data/processed/tightcrop/train/{class}/
        data/processed/tightcrop/val/{class}/
        data/processed/tightcrop/test/{class}/
    """
    # I looped over the three logical splits.
    for split in ["train", "val", "test"]:
        # For each split, I ensured all four class subfolders existed.
        for cls in CLASSES:
            out_dir = PROCESSED_ROOT / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)


def resize_and_copy(df_split: pd.DataFrame, split_name: str):
    """
    Create the processed dataset:

      - read the raw image,
      - apply tight cropping around the brain,
      - resize to IMG_SIZE (224x224),
      - convert to RGB,
      - save into data/processed/tightcrop/{split}/{class}/.

    All training/validation/testing is done on these cropped 224x224 RGB images.
    """
    # I converted the split DataFrame into a list of row dictionaries.
    rows = df_split.to_dict(orient="records")

    # I looped over all rows with a tqdm progress bar.
    for row in tqdm(rows, desc=f"Processing {split_name} [{VARIANT}]"):
        # I extracted the source path as a Path object.
        src = Path(row["orig_path"])
        # I read the class label.
        cls = row["class"]
        # I built the destination path under the processed directory.
        dst = PROCESSED_ROOT / split_name / cls / src.name

        # If the destination file already existed, I skipped reprocessing it.
        if dst.exists():
            continue

        try:
            # I opened the source image and converted it to RGB.
            img = Image.open(src).convert("RGB")

            # I always cropped around the brain before resizing.
            img = tight_crop_to_brain(img)
            # I resized the cropped image to the standard IMG_SIZE.
            img = img.resize(IMG_SIZE)

            # I saved the final processed image to the destination path.
            img.save(dst)
        except Exception as e:
            # If anything failed during processing, I logged the issue but continued.
            print(f"Failed to process {src}: {e}")


def save_csv_splits(df_train, df_val, df_test):
    
    """
    Build CSVs that map to the *processed* (cropped) paths.

    Each CSV has columns:
      - image_path : path to cropped 224x224 RGB image.
      - class      : tumour class label.
    """
    # I ensured the split directory for this variant existed.
    SPLITS_ROOT.mkdir(parents=True, exist_ok=True)

    # I defined a helper that remapped original paths to processed paths for one split.
    def map_to_processed(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
        processed_paths = []
        # I iterated over all rows in the DataFrame.
        for _, row in df.iterrows():
            orig = Path(row["orig_path"])
            cls = row["class"]
            # I built the corresponding processed image path.
            processed_paths.append(
                str(PROCESSED_ROOT / split_name / cls / orig.name)
            )
        # I returned a DataFrame with final image paths and class labels.
        return pd.DataFrame(
            {"image_path": processed_paths, "class": df["class"].values}
        )

    # I built the remapped DataFrames for each split.
    train_csv = map_to_processed(df_train, "train")
    val_csv = map_to_processed(df_val, "val")
    test_csv = map_to_processed(df_test, "test")

    # I saved them to CSV files.
    train_csv.to_csv(SPLITS_ROOT / "train.csv", index=False)
    val_csv.to_csv(SPLITS_ROOT / "val.csv", index=False)
    test_csv.to_csv(SPLITS_ROOT / "test.csv", index=False)

    # I logged where the CSVs had been saved.
    print(f"Saved split CSVs to {SPLITS_ROOT}.")


def save_summary(df_train, df_val, df_test):
    
    """
    Summarise class counts per split, so that I can show
    a simple table in the report and double-check that
    stratification worked as expected.

    Output:
      - results/dataset_summary.csv
    """
    # I ensured the results directory existed.
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # I defined a small helper that counted class frequencies for a given split.
    def counts(df, split_name):
        c = df["class"].value_counts().rename("count").reset_index()
        c = c.rename(columns={"index": "class"})
        c.insert(0, "split", split_name)
        return c

    # I concatenated the per-split class counts into a single summary DataFrame.
    summary_df = pd.concat(
        [
            counts(df_train, "train"),
            counts(df_val, "val"),
            counts(df_test, "test"),
        ],
        axis=0,
        ignore_index=True,
    )

    # I saved the summary table to CSV.
    summary_df.to_csv(SUMMARY_PATH, index=False)

    # I logged that the summary was written and printed it for inspection.
    print("Saved dataset summary to results/dataset_summary.csv")
    print(summary_df)


# -------------------------------------------------------------------
# MAIN ORCHESTRATION
# -------------------------------------------------------------------

def main():
    """
    Orchestrate the entire dataset preparation pipeline:

      1. Collected raw images from Kaggle folders.
      2. Saved raw class counts per Kaggle split.
      3. Deduplicated based on SHA1 and record duplicates.
      4. Analysed raw images (geometry - intensity - quality flags).
      5. Saved analysis CSVs (used later for figures).
      6. Optionally ran a pHash-based near-duplicate audit.
      7. Created a stratified 70/15/15 split on the deduplicated set.
      8. Wrote cropped 224x224 RGB images into data/processed/tightcrop/.
      9. Savex train/val/test CSVs under data/splits/tightcrop/ and
         an overall summary table (shared for all model runs).
    """
    # I printed a header message indicating which variant I was preparing.
    print(f"Running dataset preparation for VARIANT = '{VARIANT}' (cropped-only pipeline)")

    # Step 1: I collected all raw image paths from the Kaggle folders.
    df_raw = collect_images()
    # Step 2: I saved class counts based on Kaggle's original split.
    save_raw_class_counts(df_raw)

    # Step 3: I removed exact duplicates and saved duplicate metadata.
    df_dedup, dups_df = drop_duplicates(df_raw)
    save_duplicate_summary(dups_df)

    # Step 4: I analysed the raw images for geometry and intensity.
    stats_df = analyze_raw_images(df_dedup)
    # Step 5: I saved analysis artefacts (stats and summaries) to CSV.
    save_raw_analysis(stats_df)

    # Step 6: I optionally ran a pHash near-duplicate audit for qualitative analysis.
    if RUN_NEAR_DUP_AUDIT:
        run_near_duplicate_audit(stats_df)

    # Step 7: I created stratified train/val/test splits based on the deduplicated set.
    df_train, df_val, df_test = make_splits(df_dedup)

    # Step 8: I prepared directories and wrote cropped, resized images for each split.
    prepare_processed_dirs()
    resize_and_copy(df_train, "train")
    resize_and_copy(df_val, "val")
    resize_and_copy(df_test, "test")

    # Step 9: I saved split CSVs and the final dataset summary.
    save_csv_splits(df_train, df_val, df_test)
    save_summary(df_train, df_val, df_test)

    # I logged that the whole pipeline had completed successfully.
    print("Dataset preparation and audit completed (cropped 224x224 images only).")


# I invoked main() only when running this file as a script.
if __name__ == "__main__":
    main()
