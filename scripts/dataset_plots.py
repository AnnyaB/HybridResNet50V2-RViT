# dataset_plots.py

"""
Publication-quality figures for the dataset/methodology section, generated
from the CSV artefacts created by dataset_prep.py.

Figures produced (saved under results/):
1) kaggle_training_testing_pies.png
2) splits_class_pies.png
3) class_distribution_overall_pct.png
4) quality_flags_pct.png
5) good_vs_suspect_overall_pie.png
6) resolution_distribution_topk.png
7) resolution_distribution_all_pie.png
8) duplicates_effect_bar.png
9) split_class_heatmap.png
10) examples_per_class.png
11) examples_good_vs_weird.png

Notes:
- This script assumes dataset_prep.py has already been run.
- Titles/wording align with the leakage-safe Kaggle-aligned split:
    * Train/Val are created from Kaggle Training (e.g., 80/20 stratified).
    * Test is Kaggle Testing (held-out).
"""

# Standard library: path handling
from pathlib import Path

# Third-party: numeric ops, CSV loading, plotting, and image loading
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ---------------------------------------------------------------------
# Global Matplotlib style (paper-like)
# ---------------------------------------------------------------------

# Configure Matplotlib defaults once so all figures are consistent:
# - high DPI outputs for reports
# - clean fonts, readable sizes
# - minimalist axes styling + subtle grid
plt.rcParams.update(
    {
        "figure.dpi": 200,        # interactive/back-end dpi (preview)
        "savefig.dpi": 600,       # file dpi (crisp for thesis PDF)
        "savefig.facecolor": "white",

        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.25,
        "axes.axisbelow": True,

        "axes.linewidth": 0.9,
        "lines.linewidth": 1.6,
    }
)

# ---------------------------------------------------------------------
# Paths (match dataset_prep.py conventions)
# ---------------------------------------------------------------------

# Resolve project root relative to this file (repo/scripts/..)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# All figures go to results/
RESULTS_DIR = PROJECT_ROOT / "results"

# CSV artefacts from dataset_prep.py (these MUST exist for plots)
DATASET_SUMMARY_PATH = RESULTS_DIR / "dataset_summary.csv"
RAW_STATS_PATH = RESULTS_DIR / "raw_image_stats.csv"
RAW_RESOLUTION_SUMMARY_PATH = RESULTS_DIR / "raw_resolution_summary.csv"
RAW_CLASS_COUNTS_PATH = RESULTS_DIR / "raw_class_counts_by_source.csv"
DUPLICATE_SUMMARY_PATH = RESULTS_DIR / "duplicate_summary.csv"

# Optional split CSVs for showing processed examples (cropped/resized model inputs)
SPLITS_CSV_DIR = PROJECT_ROOT / "data" / "splits" / "tightcrop"
TRAIN_SPLIT_CSV = SPLITS_CSV_DIR / "train.csv"
VAL_SPLIT_CSV = SPLITS_CSV_DIR / "val.csv"
TEST_SPLIT_CSV = SPLITS_CSV_DIR / "test.csv"

# Consistent class order across all plots (important for comparison)
CLASS_ORDER = ["glioma", "meningioma", "pituitary", "notumor"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _pretty_class_name(c: str) -> str:
    # Convert internal class key to report-friendly label
    if c == "notumor":
        return "No tumour"
    return str(c).replace("_", " ").capitalize()


def _prettify_classes(classes):
    # Vectorized helper: apply _pretty_class_name to each class
    return [_pretty_class_name(c) for c in classes]


def _save_fig(fig: plt.Figure, stem_name: str):
    """
    Save both PNG (high DPI) and PDF (vector) for report use.
    """
    # Ensure results/ exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Output filenames are consistent across formats
    png_path = RESULTS_DIR / f"{stem_name}.png"
    pdf_path = RESULTS_DIR / f"{stem_name}.pdf"

    # Save with tight bounding box (avoids cropped labels)
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")

    # Close figure to avoid memory leaks in long runs
    plt.close(fig)

    # Console logs help confirm outputs
    print(f"Saved figure: {png_path}")
    print(f"Saved figure: {pdf_path}")


def _legend_labels_with_counts_and_pct(labels, counts):
    # Builds legend entries like: "Glioma (n=123, 12.3%)"
    total = float(np.sum(counts)) if np.sum(counts) > 0 else 1.0
    out = []
    for lab, n in zip(labels, counts):
        pct = 100.0 * (float(n) / total)
        out.append(f"{lab}  (n={int(n)}, {pct:.1f}%)")
    return out


def _donut(ax, counts, labels, title, colors=None):
    """
    Donut chart with a clean legend showing both counts and percentages.
    """
    # Ensure numeric
    counts = np.asarray(counts, dtype=float)

    # Pie chart wedges (labels hidden; legend used instead)
    wedges, _ = ax.pie(
        counts,
        startangle=90,
        counterclock=False,
        labels=None,
        colors=colors,
        wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
    )

    # Donut hole (white circle drawn on top)
    centre = plt.Circle((0, 0), 0.62, fc="white")
    ax.add_artist(centre)

    # Titles and equal aspect so circle stays circular
    ax.set_title(title)
    ax.axis("equal")

    # Legend shows label + count + percent
    legend_labels = _legend_labels_with_counts_and_pct(labels, counts)
    ax.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )


def _barh_counts(ax, labels, counts, title, xlabel):
    """
    Horizontal bar chart (paper-friendly for long labels) with value annotations.
    """
    # y positions for each label
    y = np.arange(len(labels))

    # Draw bars
    ax.barh(y, counts)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)

    # Invert so highest count appears at top (more readable)
    ax.invert_yaxis()

    ax.set_xlabel(xlabel)
    ax.set_title(title)

    # Pad for text annotations
    maxv = float(np.max(counts)) if len(counts) else 1.0
    pad = maxv * 0.01

    # Annotate each bar with the integer count
    for i, v in enumerate(counts):
        ax.text(float(v) + pad, i, f"{int(v)}", va="center")


def _barh_percent_with_counts(ax, labels, counts, title, xlabel="Percentage of images (%)"):
    """
    Horizontal bar chart that shows percentages with count annotations.
    """
    # Convert to numeric
    counts = np.asarray(counts, dtype=float)

    # Total for percent conversion
    total = float(np.sum(counts)) if np.sum(counts) > 0 else 1.0

    # Percent values
    perc = (counts / total) * 100.0

    # y positions
    y = np.arange(len(labels))

    # Draw bars in percentage space
    ax.barh(y, perc)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel(xlabel)
    ax.set_title(title)

    # Padding for annotation text
    maxp = float(np.max(perc)) if len(perc) else 1.0
    pad = maxp * 0.01

    # Annotate with "x.x% (n=...)"
    for i, (p, n) in enumerate(zip(perc, counts)):
        ax.text(float(p) + pad, i, f"{p:.1f}%  (n={int(n)})", va="center")


# ---------------------------------------------------------------------
# 1) Kaggle Training vs Testing donuts
# ---------------------------------------------------------------------

def plot_kaggle_training_testing_pies():
    # This plot describes the original Kaggle folders (before our Train/Val split).
    # Data source: results/raw_class_counts_by_source.csv
    if not RAW_CLASS_COUNTS_PATH.exists():
        print("raw_class_counts_by_source.csv not found; skipping Kaggle donuts.")
        return

    # Load counts per (source_split, class)
    df = pd.read_csv(RAW_CLASS_COUNTS_PATH)

    # Ensure consistent class order
    df = df[df["class"].isin(CLASS_ORDER)].copy()

    # Two subplots: Training and Testing
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Use a categorical colormap and lock colors by class order
    cmap = plt.get_cmap("Set2")
    colors = cmap(np.linspace(0, 1, len(CLASS_ORDER)))
    labels = _prettify_classes(CLASS_ORDER)

    # Build a donut per split
    for ax, split in zip(axes, ["training", "testing"]):
        sub = df[df["source_split"] == split].set_index("class").reindex(CLASS_ORDER)
        counts = sub["count"].fillna(0).to_numpy()
        _donut(
            ax,
            counts=counts,
            labels=labels,
            title=f"Kaggle {split.capitalize()} (N={int(np.sum(counts))})",
            colors=colors,
        )

    # Save in results/ as PNG+PDF
    _save_fig(fig, "kaggle_training_testing_pies")


# ---------------------------------------------------------------------
# 2) Our Train/Val/Test donuts (from dataset_summary.csv)
# ---------------------------------------------------------------------

def plot_our_split_pies():
    # This plot describes our final split after deduplication + Train/Val creation.
    # Data source: results/dataset_summary.csv
    if not DATASET_SUMMARY_PATH.exists():
        print("dataset_summary.csv not found; skipping split donuts.")
        return

    df = pd.read_csv(DATASET_SUMMARY_PATH)
    df = df[df["class"].isin(CLASS_ORDER)].copy()

    # Three donuts: train, val, test
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

    cmap = plt.get_cmap("Set2")
    colors = cmap(np.linspace(0, 1, len(CLASS_ORDER)))
    labels = _prettify_classes(CLASS_ORDER)

    for ax, split in zip(axes, ["train", "val", "test"]):
        sub = df[df["split"] == split].set_index("class").reindex(CLASS_ORDER)
        counts = sub["count"].fillna(0).to_numpy()

        # Titles deliberately remind the reader what “test” means here
        if split == "test":
            subtitle = "Held-out Kaggle Testing"
        else:
            subtitle = "From Kaggle Training"

        _donut(
            ax,
            counts=counts,
            labels=labels,
            title=f"{split.upper()} ({subtitle})\nN={int(np.sum(counts))}",
            colors=colors,
        )

    _save_fig(fig, "splits_class_pies")


# ---------------------------------------------------------------------
# 3) Overall class distribution (percent + counts)
# ---------------------------------------------------------------------

def plot_overall_class_distribution_pct():
    # Shows overall label balance after deduplication + final split.
    if not DATASET_SUMMARY_PATH.exists():
        print("dataset_summary.csv not found; skipping overall class distribution.")
        return

    df = pd.read_csv(DATASET_SUMMARY_PATH)

    # Sum counts across splits to get overall per-class counts
    overall = df.groupby("class")["count"].sum().reindex(CLASS_ORDER).fillna(0)

    labels = _prettify_classes(overall.index.tolist())
    counts = overall.to_numpy()
    total = int(np.sum(counts))

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    _barh_percent_with_counts(
        ax,
        labels=labels,
        counts=counts,
        title=f"Overall class distribution (after deduplication; N={total})",
        xlabel="Percentage of images (%)",
    )
    _save_fig(fig, "class_distribution_overall_pct")


# ---------------------------------------------------------------------
# 4) Quality flags per class (stacked percentage bars)
# ---------------------------------------------------------------------

def plot_quality_flags_pct():
    # Shows how many images were flagged too_dark / too_bright / low_contrast / failed.
    # Data source: results/raw_image_stats.csv
    if not RAW_STATS_PATH.exists():
        print("raw_image_stats.csv not found; skipping quality flags plot.")
        return

    stats = pd.read_csv(RAW_STATS_PATH)
    stats = stats[stats["class"].isin(CLASS_ORDER)].copy()
    if stats.empty:
        print("raw_image_stats.csv empty; skipping quality flags plot.")
        return

    # If columns missing (in case script evolves), create them to avoid crashes
    for col in ["too_dark", "too_bright", "low_contrast", "failed", "suspect"]:
        if col not in stats.columns:
            stats[col] = False

    # Aggregate counts of each flag per class
    agg = (
        stats.groupby("class")[["too_dark", "too_bright", "low_contrast", "failed"]]
        .sum()
        .reindex(CLASS_ORDER)
        .fillna(0)
        .astype(int)
    )

    # Total images per class (needed for percent conversion)
    total_per_class = stats.groupby("class").size().reindex(CLASS_ORDER).fillna(0).astype(int)

    # Convert each flag count to a percent of that class
    pct = agg.div(total_per_class.replace(0, np.nan), axis=0).fillna(0.0) * 100.0

    labels = _prettify_classes(CLASS_ORDER)
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)

    left = np.zeros(len(labels), dtype=float)
    parts = [
        ("too_dark", "Too dark"),
        ("too_bright", "Too bright"),
        ("low_contrast", "Low contrast"),
        ("failed", "Failed to load"),
    ]

    # Stacked horizontal bars: each part adds on the left
    for key, lab in parts:
        values = pct[key].to_numpy()
        ax.barh(y, values, left=left, label=lab)
        left += values

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Percentage of images (%)")
    ax.set_title("Quality flags per class (percent of images)")

    # Legend outside plot area (paper-friendly)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    # Annotate suspect percentage (derived in dataset_prep.py) for quick interpretation
    suspect_pct = (stats.groupby("class")["suspect"].mean().reindex(CLASS_ORDER).fillna(0.0) * 100.0).to_numpy()
    for i, sp in enumerate(suspect_pct):
        ax.text(min(left[i] + 1.0, 99.0), i, f"Suspect: {sp:.1f}%", va="center")

    _save_fig(fig, "quality_flags_pct")


# ---------------------------------------------------------------------
# 5) Overall good vs suspect (donut)
# ---------------------------------------------------------------------

def plot_good_vs_suspect_overall_pie():
    # Donut chart: how many images are typical vs suspect in the raw audit.
    if not RAW_STATS_PATH.exists():
        print("raw_image_stats.csv not found; skipping good vs suspect donut.")
        return

    stats = pd.read_csv(RAW_STATS_PATH)
    if "suspect" not in stats.columns:
        print("No 'suspect' column found; skipping good vs suspect donut.")
        return

    n_suspect = int(stats["suspect"].sum())
    n_good = int((~stats["suspect"]).sum())
    total = n_good + n_suspect

    fig, ax = plt.subplots(figsize=(8.2, 4.5), constrained_layout=True)
    _donut(
        ax,
        counts=[n_good, n_suspect],
        labels=["Typical / OK", "Suspect / low quality"],
        title=f"Typical vs suspect images (raw audit; N={total})",
        colors=None,
    )
    _save_fig(fig, "good_vs_suspect_overall_pie")


# ---------------------------------------------------------------------
# 6) Resolution distribution (top-K + Other)
# ---------------------------------------------------------------------

def plot_resolution_distribution_topk(max_bins: int = 8):
    # Bar chart for the most common raw resolutions, plus "Other".
    if not RAW_RESOLUTION_SUMMARY_PATH.exists():
        print("raw_resolution_summary.csv not found; skipping resolution plot.")
        return

    res = pd.read_csv(RAW_RESOLUTION_SUMMARY_PATH)
    if res.empty:
        print("raw_resolution_summary.csv is empty; skipping resolution plot.")
        return

    res = res.sort_values("count", ascending=False).reset_index(drop=True)
    total = int(res["count"].sum())

    # If too many unique resolutions, collapse tail into "Other"
    if len(res) > max_bins:
        top = res.head(max_bins - 1).copy()
        other_count = int(res["count"].iloc[max_bins - 1 :].sum())
        top = pd.concat(
            [top, pd.DataFrame([{"width": -1, "height": -1, "count": other_count}])],
            ignore_index=True,
        )
        labels = [f"{int(w)}×{int(h)}" for w, h in zip(top["width"][:-1], top["height"][:-1])] + ["Other"]
        counts = top["count"].to_numpy()
    else:
        labels = [f"{int(w)}×{int(h)}" for w, h in zip(res["width"], res["height"])]
        counts = res["count"].to_numpy()

    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    _barh_counts(
        ax,
        labels=labels,
        counts=counts,
        title=f"Most common raw resolutions (top {max_bins} incl. Other; N={total})",
        xlabel="Number of images",
    )
    _save_fig(fig, "resolution_distribution_topk")


# ---------------------------------------------------------------------
# 7) Resolution distribution donut (top-N + Other)
# ---------------------------------------------------------------------

def plot_resolution_distribution_all_pie(top_n: int = 10):
    # Donut chart version of resolution distribution.
    if not RAW_RESOLUTION_SUMMARY_PATH.exists():
        print("raw_resolution_summary.csv not found; skipping resolution donut.")
        return

    res = pd.read_csv(RAW_RESOLUTION_SUMMARY_PATH)
    if res.empty:
        print("raw_resolution_summary.csv is empty; skipping resolution donut.")
        return

    res = res.sort_values("count", ascending=False).reset_index(drop=True)
    total = int(res["count"].sum())

    if len(res) > top_n:
        top = res.head(top_n).copy()
        other_count = int(res["count"].iloc[top_n:].sum())
        labels = [f"{int(w)}×{int(h)}" for w, h in zip(top["width"], top["height"])] + ["Other"]
        counts = top["count"].to_numpy().tolist() + [other_count]
    else:
        labels = [f"{int(w)}×{int(h)}" for w, h in zip(res["width"], res["height"])]
        counts = res["count"].to_numpy().tolist()

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    _donut(
        ax,
        counts=counts,
        labels=labels,
        title=f"Raw resolution distribution (top {min(top_n, len(res))} + Other; N={total})",
        colors=None,
    )
    _save_fig(fig, "resolution_distribution_all_pie")


# ---------------------------------------------------------------------
# 8) Duplicate removal effect
# ---------------------------------------------------------------------

def plot_duplicates_effect_bar():
    # Visualizes how many duplicate entries were removed by SHA1 deduplication.
    if not DUPLICATE_SUMMARY_PATH.exists() or not DATASET_SUMMARY_PATH.exists():
        print("duplicate_summary.csv or dataset_summary.csv missing; skipping duplicates plot.")
        return

    dup_summary = pd.read_csv(DUPLICATE_SUMMARY_PATH)

    # If a group had n_files, dedup removes (n_files - 1) entries
    duplicates_removed = int((dup_summary["n_files"] - 1).clip(lower=0).sum())

    summary = pd.read_csv(DATASET_SUMMARY_PATH)
    n_after = int(summary["count"].sum())
    n_before = n_after + duplicates_removed

    labels = ["Unique images kept", "Exact duplicates removed"]
    counts = [n_after, duplicates_removed]

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    _barh_counts(
        ax,
        labels=labels,
        counts=counts,
        title=f"Effect of SHA1-based deduplication (before ≈ {n_before}, after = {n_after})",
        xlabel="Number of images",
    )
    _save_fig(fig, "duplicates_effect_bar")


# ---------------------------------------------------------------------
# 9) Split x class heatmap
# ---------------------------------------------------------------------
def plot_split_class_heatmap():
    """
    Publication-style heatmap of counts per (class × split) from dataset_summary.csv.
    (Self-contained: does not depend on _prettify_categories)
    """
    # Read final summary counts
    df = pd.read_csv(DATASET_SUMMARY_PATH)
    df = df[df["class"].isin(CLASS_ORDER)].copy()

    # Pivot to a matrix: rows=class, cols=split, values=count
    pivot = df.pivot(index="class", columns="split", values="count")
    pivot = pivot.reindex(index=CLASS_ORDER)
    pivot = pivot.reindex(columns=["train", "val", "test"])
    pivot = pivot.fillna(0).astype(int)

    # Pretty labels for plot axes
    label_map = {
        "glioma": "Glioma",
        "meningioma": "Meningioma",
        "pituitary": "Pituitary",
        "notumor": "No tumour",
    }
    classes_pretty = [label_map.get(c, str(c).capitalize()) for c in pivot.index.tolist()]
    splits_pretty = ["Train", "Val", "Test"]

    data = pivot.values
    vmin = int(data.min()) if data.size else 0
    vmax = int(data.max()) if data.size else 1
    thresh = vmin + (vmax - vmin) * 0.55  # used for auto-contrast text color

    fig, ax = plt.subplots(figsize=(7.2, 4.8))

    # Heatmap image
    im = ax.imshow(
        data,
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
        vmin=vmin,
        vmax=vmax,
    )

    # Colorbar explains numeric scale
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of images", rotation=90)
    cbar.ax.tick_params(labelsize=9)

    # Axis labels and tick labels
    ax.set_xticks(np.arange(len(splits_pretty)))
    ax.set_xticklabels(splits_pretty, fontsize=11)
    ax.set_yticks(np.arange(len(classes_pretty)))
    ax.set_yticklabels(classes_pretty, fontsize=11)

    ax.set_xlabel("Split", fontsize=12)
    ax.set_ylabel("Class", fontsize=12)
    ax.set_title("Number of images per class and split", fontsize=14, pad=10)

    # White gridlines between cells (confusion-matrix style readability)
    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.0)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Annotate each cell with its integer count, using auto-contrast color
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = int(data[i, j])
            text_color = "black" if val >= thresh else "white"
            ax.text(
                j, i, f"{val}",
                ha="center", va="center",
                fontsize=12, fontweight="bold",
                color=text_color,
            )

    # Save PNG (this function saves only PNG; your other helper saves PNG+PDF)
    fig.tight_layout()
    out_path = RESULTS_DIR / "split_class_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")

# ---------------------------------------------------------------------
# 10) Example grid: one example per class
# ---------------------------------------------------------------------

def _load_any_image(path: str, size=(256, 256)) -> Image.Image:
    # Load and resize any image to a consistent thumbnail size for grids
    img = Image.open(path).convert("RGB")
    return img.resize(size)


def create_example_per_class_grid(out_stem="examples_per_class", seed=42):
    """
    1×4 grid: one representative image per class.

    Prefers processed/cropped images (model inputs) via split CSVs if present.
    Falls back to raw images from raw_image_stats.csv.
    """
    # Deterministic sampling
    np.random.seed(seed)

    picked = []

    # Prefer processed split CSVs: these show exactly what the model sees
    split_csvs = [TRAIN_SPLIT_CSV, VAL_SPLIT_CSV, TEST_SPLIT_CSV]
    processed_rows = []
    for p in split_csvs:
        if p.exists():
            processed_rows.append(pd.read_csv(p))

    if processed_rows:
        processed = pd.concat(processed_rows, axis=0, ignore_index=True)
        processed = processed[processed["class"].isin(CLASS_ORDER)].copy()
        for cls in CLASS_ORDER:
            sub = processed[processed["class"] == cls]
            if sub.empty:
                continue
            path = sub.sample(1, random_state=seed)["image_path"].iloc[0]
            picked.append((cls, path))
    else:
        # Fallback: use raw stats CSV if processed split CSVs aren't available
        if not RAW_STATS_PATH.exists():
            print("No split CSVs and raw_image_stats.csv missing; cannot build examples grid.")
            return

        stats = pd.read_csv(RAW_STATS_PATH)
        stats = stats[stats["class"].isin(CLASS_ORDER)].copy()

        # Avoid failed-to-load images
        if "failed" in stats.columns:
            stats = stats[~stats["failed"]].copy()

        for cls in CLASS_ORDER:
            sub = stats[stats["class"] == cls]
            if sub.empty:
                continue
            # Prefer good (non-suspect) if possible
            if "suspect" in sub.columns:
                good = sub[~sub["suspect"]]
                if not good.empty:
                    sub = good
            path = sub.sample(1, random_state=seed)["orig_path"].iloc[0]
            picked.append((cls, path))

    if not picked:
        print("No images found to build per-class grid.")
        return

    labels = [_pretty_class_name(c) for c, _ in picked]
    paths = [p for _, p in picked]

    thumb_size = (256, 256)
    n = len(paths)

    fig, axes = plt.subplots(1, n, figsize=(3.0 * n, 3.4), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, lab, path in zip(axes, labels, paths):
        try:
            ax.imshow(_load_any_image(path, size=thumb_size))
            ax.set_title(lab)
            ax.axis("off")
        except Exception as e:
            ax.axis("off")
            ax.set_title(lab)
            ax.text(0.5, 0.5, f"Failed to load\n{Path(path).name}", ha="center", va="center")
            print(f"Failed to load example image {path}: {e}")

    _save_fig(fig, out_stem)


# ---------------------------------------------------------------------
# 11) Example grid: typical vs atypical/suspect (raw audit view)
# ---------------------------------------------------------------------
def create_example_good_vs_weird_grid(
    n_good=4,
    n_weird=4,
    out_stem="examples_good_vs_weird",
    seed=42,
):
    """
    Two-row grid with an OUTSIDE label column (paper-style).

    Uses raw_image_stats.csv because suspect flags + raw resolution come from the raw audit.
    """
    if not RAW_STATS_PATH.exists():
        print("raw_image_stats.csv not found; skipping good vs weird grid.")
        return

    np.random.seed(seed)
    stats = pd.read_csv(RAW_STATS_PATH).copy()

    # Remove failed loads
    if "failed" in stats.columns:
        stats = stats[~stats["failed"]].copy()

    if not {"width", "height"}.issubset(stats.columns):
        print("width/height not found in raw stats; skipping good vs weird grid.")
        return

    # Dominant resolution = most frequent (width,height)
    res_counts = stats.groupby(["width", "height"]).size().sort_values(ascending=False)
    dom_w, dom_h = res_counts.index[0]
    dominant_mask = (stats["width"] == dom_w) & (stats["height"] == dom_h)

    # Define good vs weird candidates based on dominant resolution and suspect flags
    if "suspect" in stats.columns:
        good_candidates = stats[dominant_mask & (~stats["suspect"])].copy()
        weird_candidates = stats[(~dominant_mask) | (stats["suspect"])].copy()
    else:
        good_candidates = stats[dominant_mask].copy()
        weird_candidates = stats[~dominant_mask].copy()

    # Fallbacks if categories are too small
    if len(good_candidates) < n_good:
        good_candidates = stats.copy()
    if len(weird_candidates) < n_weird:
        weird_candidates = stats.copy()

    # Sample paths deterministically
    good_rows = good_candidates.sample(min(n_good, len(good_candidates)), random_state=seed)
    weird_rows = weird_candidates.sample(min(n_weird, len(weird_candidates)), random_state=seed + 1)

    good_paths = good_rows["orig_path"].tolist()
    weird_paths = weird_rows["orig_path"].tolist()

    thumb_size = (224, 224)
    ncols = max(len(good_paths), len(weird_paths))
    if ncols == 0:
        print("No images available for good vs weird grid.")
        return

    # Layout: 2 rows × (1 label column + N image columns)
    fig_w = 2.35 * ncols + 4.0
    fig_h = 6.0
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = fig.add_gridspec(
        nrows=2,
        ncols=ncols + 1,
        width_ratios=[1.7] + [1.0] * ncols,
        wspace=0.06,
        hspace=0.08,
    )

    # Left label panels
    ax_lab_top = fig.add_subplot(gs[0, 0])
    ax_lab_bot = fig.add_subplot(gs[1, 0])
    for ax in (ax_lab_top, ax_lab_bot):
        ax.axis("off")

    # Text box styling
    box = dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="black", linewidth=0.8)

    # Top label (typical)
    ax_lab_top.text(
        0.02, 0.5,
        f"Typical / high-quality\n"
        f"• Dominant resolution: {int(dom_w)}×{int(dom_h)}\n"
        f"• Not flagged suspect (if available)",
        ha="left", va="center", fontsize=11, bbox=box
    )

    # Bottom label (atypical)
    ax_lab_bot.text(
        0.02, 0.5,
        "Atypical / suspect\n"
        "• Non-dominant resolution OR\n"
        "• Flagged too dark / too bright /\n"
        "  low contrast / failed-to-load, etc.",
        ha="left", va="center", fontsize=11, bbox=box
    )

    # Safe image loading helper
    def _show_image(ax, path):
        ax.axis("off")
        try:
            img = Image.open(path).convert("RGB").resize(thumb_size)
            ax.imshow(img)
        except Exception as e:
            ax.text(0.5, 0.5, "Failed to load", ha="center", va="center")
            print(f"Failed to load {path}: {e}")

    # Top row images (typical)
    for col in range(ncols):
        ax = fig.add_subplot(gs[0, col + 1])
        if col < len(good_paths):
            _show_image(ax, good_paths[col])
        else:
            ax.axis("off")

    # Bottom row images (suspect/atypical)
    for col in range(ncols):
        ax = fig.add_subplot(gs[1, col + 1])
        if col < len(weird_paths):
            _show_image(ax, weird_paths[col])
        else:
            ax.axis("off")

    fig.suptitle(
        "Qualitative raw-data audit: typical vs atypical/suspect examples",
        fontsize=14, y=0.98
    )

    _save_fig(fig, out_stem)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    """
    Orchestrates generation of all dataset figures.
    """
    # Hard requirements: prep must have been run
    if not DATASET_SUMMARY_PATH.exists():
        raise FileNotFoundError(f"{DATASET_SUMMARY_PATH} not found. Run dataset_prep.py first.")
    if not RAW_STATS_PATH.exists():
        raise FileNotFoundError(f"{RAW_STATS_PATH} not found. Run dataset_prep.py first.")

    # Print a small summary so logs show what dataset size we're plotting
    summary = pd.read_csv(DATASET_SUMMARY_PATH)
    total_images = int(summary["count"].sum())
    per_split = summary.groupby("split")["count"].sum()

    print(f"Total images after deduplication + final split: {total_images}")
    for split, n in per_split.items():
        print(f"  {split}: {int(n)}")

    # Kaggle Training vs Testing donuts
    plot_kaggle_training_testing_pies()

    # Our final Train/Val/Test donuts (Kaggle-aligned)
    plot_our_split_pies()

    # Class distribution and quality
    plot_overall_class_distribution_pct()
    plot_quality_flags_pct()
    plot_good_vs_suspect_overall_pie()

    # Resolution + duplicates
    plot_resolution_distribution_topk()
    plot_resolution_distribution_all_pie()
    plot_duplicates_effect_bar()

    # Heatmap
    plot_split_class_heatmap()

    # Qualitative examples
    create_example_per_class_grid()
    create_example_good_vs_weird_grid()


if __name__ == "__main__":
    main()
