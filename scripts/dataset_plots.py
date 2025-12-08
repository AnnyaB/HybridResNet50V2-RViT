# started working on it from 18th november 2025 
"""
dataset_plots.py

figures for the dataset section of the project method, generated
from the CSV artefacts created by dataset_prep.py.


Figures produced (all saved under results/):

1) kaggle_training_testing_pies.png
2) splits_class_pies.png
3) class_distribution_overall_pct.png
4) quality_flags_pct.png
5) good_vs_suspect_overall_pie.png
6) resolution_distribution_topk.png
7) resolution_distribution_all_pie.png
8) duplicates_effect_bar.png
9) near_duplicate_cluster_sizes.png
10) split_class_heatmap.png
11) examples_per_class.png
12) examples_good_vs_weird.png
"""

# these are the standard library  imports
from pathlib import Path          # clean, OS-independent path handling

import numpy as np               # numerical work (arrays, simple maths)
import pandas as pd              # reading the CSV artefacts and grouping
import matplotlib.pyplot as plt  # all plots for the figures
from PIL import Image            # opening, resizing and composing example MRIs

# ---------------------------------------------------------------------
# Global Matplotlib style settings
# ---------------------------------------------------------------------

# I updated matplotlib's global rcParams once so that all figures
# in this file use consistent fonts, DPI and axis style. This helps
# the plots look like publication-quality figures in the thesis.
plt.rcParams.update(
    {
        # Higher default DPI so everything is crisp in the PDF
        "figure.dpi": 300,
        "savefig.dpi": 300,
        # Fonts: slightly larger than matplotlib defaults
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        # Turn off top/right spines by default for a cleaner look
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ---------------------------------------------------------------------
# Paths (same conventions as dataset_prep.py)
# ---------------------------------------------------------------------

# Project root = folder above scripts/. This matches dataset_prep.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# All CSV artefacts and figures are stored under results/
RESULTS_DIR = PROJECT_ROOT / "results"

# Paths to the CSVs produced by dataset_prep.py
DATASET_SUMMARY_PATH        = RESULTS_DIR / "dataset_summary.csv"
RAW_STATS_PATH              = RESULTS_DIR / "raw_image_stats.csv"
RAW_RESOLUTION_SUMMARY_PATH = RESULTS_DIR / "raw_resolution_summary.csv"
RAW_QUALITY_SUMMARY_PATH    = RESULTS_DIR / "raw_quality_flags_summary.csv"
RAW_CLASS_COUNTS_PATH       = RESULTS_DIR / "raw_class_counts_by_source.csv"
DUPLICATE_SUMMARY_PATH      = RESULTS_DIR / "duplicate_summary.csv"
NEAR_DUP_CLUSTERS_PATH      = RESULTS_DIR / "near_duplicate_clusters_summary.csv"

# Fixed order of tumour classes so colours and legend order stay consistent
# across all plots (important when comparing different figures).
CLASS_ORDER = ["glioma", "meningioma", "pituitary", "notumor"]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _prettify_categories(categories):
    
    """
    Helper to convert internal labels (e.g. 'notumor') into nicer axis
    labels for plots (e.g. 'No tumour').
    """
    pretty = []
    for c in categories:
        if c == "notumor":
            
            # Special case: nicer name for the negative class
            pretty.append("No tumour")
        else:
            
            # Replaced underscores with spaces and capitalise the first letter
            pretty.append(str(c).replace("_", " ").capitalize())
    return pretty


def _autopct_with_counts(counts, decimals=1):
    
    """
    Formatter factory for pie charts. It returns a function that
    matplotlib can call for each slice.

    The formatter prints both:
      - the percentage with 'decimals' decimal places, and
      - the absolute count on a new line.

    Example: "23.4%\n57"
    """
    total = float(sum(counts)) if sum(counts) != 0 else 1.0

    def _fmt(pct):
        
        # Converted percentage value (0–100) back into an approximate count
        count = int(round(pct * total / 100.0))
        return f"{pct:.{decimals}f}%\n{count}"

    return _fmt


def bar_plot_percent(categories, percents, title, ylabel, output_path):
    
    """
    Shared helper for nicer percentage bar plots.

    Responsibilities:
      - applied consistent pretty labels to categories
      - added gridlines on the y-axis
      - annotated each bar with its value (1 decimal place)
      - hide top/right spines via global rcParams
      - saved the figure to the requested path
    """
    # Converted internal category labels (like 'notumor') to nicer labels
    display_categories = _prettify_categories(categories)

    # Created a single figure + axis
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(display_categories))

    # Drew bars
    bars = ax.bar(x, percents, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(display_categories, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Robust y-limit handling even if all percents are 0
    ymax = max(percents) if len(percents) > 0 else 1.0
    if ymax <= 0:
        ymax = 1.0

    ax.set_ylim(0, ymax * 1.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)  # gridlines behind bars

    # Added percentage labels above each bar
    for b, p in zip(bars, percents):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + ymax * 0.03,
            f"{p:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # Tight layout and save
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


def bar_plot_counts(categories, counts, title, ylabel, output_path):
    
    """
    Simple bar-plot helper for *absolute* counts (not percentages).

    Same idea as bar_plot_percent but labels show integer counts.
    """
    display_categories = _prettify_categories(categories)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(display_categories))

    bars = ax.bar(x, counts, width=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(display_categories, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ymax = max(counts) if len(counts) > 0 else 1.0
    if ymax <= 0:
        ymax = 1.0

    ax.set_ylim(0, ymax * 1.2)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    # Annotated each bar with the raw count
    for b, c in zip(bars, counts):
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            b.get_height() + ymax * 0.03,
            f"{int(c)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output_path}")


# ---------------------------------------------------------------------
# Kaggle Training vs Testing pies 
# ---------------------------------------------------------------------

def plot_kaggle_training_testing_pies():
    
    """
    Two pies: Kaggle Training vs Testing class distribution.

    This uses raw_class_counts_by_source.csv to reproduce the type of
    training/testing pie chart shown in the original Kaggle paper.
    """
    # Read per-class counts for Kaggle Training / Testing folders
    df = pd.read_csv(RAW_CLASS_COUNTS_PATH)

    # Kept only our four canonical classes, in a fixed order
    df = df[df["class"].isin(CLASS_ORDER)].copy()

    splits = ["training", "testing"]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Colour palette shared across both pies (stable mapping over the thesis)
    cmap = plt.get_cmap("Set2")
    colours = cmap(np.linspace(0, 1, len(CLASS_ORDER)))
    labels = _prettify_categories(CLASS_ORDER)

    # Will hold wedges from the last pie; used for a single shared legend
    wedges_for_legend = None

    # Drew one pie per split (Training / Testing)
    for ax, split in zip(axes, splits):
        sub = df[df["source_split"] == split]
        # Reindex ensures classes appear in CLASS_ORDER even if some are missing
        sub = sub.set_index("class").reindex(CLASS_ORDER)
        counts = sub["count"].fillna(0).to_numpy()
        total = int(counts.sum())

        wedges, _texts, _autotexts = ax.pie(
            counts,
            labels=None,  # legend below instead
            autopct=_autopct_with_counts(counts, decimals=1),
            startangle=90,
            colors=colours,
            textprops={"fontsize": 8},
            wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        )
        wedges_for_legend = wedges
        ax.set_title(f"{split.capitalize()} data (N={total})", fontsize=11)
        ax.axis("equal")  # kept pie circular

    # Single shared legend beneath the pies
    fig.legend(
        handles=wedges_for_legend,
        labels=labels,
        loc="lower center",
        ncol=len(CLASS_ORDER),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout()
    out_path = RESULTS_DIR / "kaggle_training_testing_pies.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------
# Our own 70/15/15 Train/Val/Test pies
# ---------------------------------------------------------------------

def plot_our_split_pies():
    
    """
    Three pies: Train / Val / Test class distribution (70/15/15).

    This figure shows what the class balance looks like after my
    own stratified split, independent of the Kaggle folders.
    """
    df = pd.read_csv(DATASET_SUMMARY_PATH)
    df = df[df["class"].isin(CLASS_ORDER)].copy()

    splits = ["train", "val", "test"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))

    cmap = plt.get_cmap("Set2")
    colours = cmap(np.linspace(0, 1, len(CLASS_ORDER)))
    labels = _prettify_categories(CLASS_ORDER)

    wedges_for_legend = None

    for ax, split in zip(axes, splits):
        sub = df[df["split"] == split]
        sub = sub.set_index("class").reindex(CLASS_ORDER)
        counts = sub["count"].fillna(0).to_numpy()
        total = int(counts.sum())

        wedges, _texts, _autotexts = ax.pie(
            counts,
            labels=None,
            autopct=_autopct_with_counts(counts, decimals=1),
            startangle=90,
            colors=colours,
            textprops={"fontsize": 7},
            wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
        )
        wedges_for_legend = wedges
        ax.set_title(f"{split.capitalize()} split (N={total})", fontsize=11)
        ax.axis("equal")

    # Shared legend for all three pies
    fig.legend(
        handles=wedges_for_legend,
        labels=labels,
        loc="lower center",
        ncol=len(CLASS_ORDER),
        fontsize=8,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout()
    out_path = RESULTS_DIR / "splits_class_pies.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------
# Overall class distribution (bar)
# ---------------------------------------------------------------------

def plot_overall_class_distribution_pct():
    
    """
    Bar plot of the overall class distribution in percentage, based on
    the stratified split (train+val+test). This is used in the report
    to show how imbalanced the dataset is after deduplication.
    """
    df = pd.read_csv(DATASET_SUMMARY_PATH)

    # Sum counts over all splits for each class
    overall = df.groupby("class")["count"].sum().reset_index()
    overall = overall[overall["class"].isin(CLASS_ORDER)]
    total = int(overall["count"].sum())
    overall["percent"] = overall["count"] / total * 100.0

    categories = overall["class"].tolist()
    percents = overall["percent"].tolist()

    title = (
        f"Overall class distribution "
        f"(after deduplication and 70/15/15 split, N={total})"
    )
    out_path = RESULTS_DIR / "class_distribution_overall_pct.png"
    bar_plot_percent(
        categories=categories,
        percents=percents,
        title=title,
        ylabel="Percentage of images (%)",
        output_path=out_path,
    )


# ---------------------------------------------------------------------
# Suspect / low-quality images per class (bar) and overall pie
# ---------------------------------------------------------------------

def plot_quality_flags_pct():
    
    """
    Bar plot showing the percentage of images flagged as suspect /
    low-quality for each class.

    'suspect' was computed in dataset_prep.py from intensity and
    contrast heuristics (too dark / too bright / low contrast / failed).
    """
    stats = pd.read_csv(RAW_STATS_PATH)

    # Total images per class
    total_per_class = stats.groupby("class").size().rename("total")
    # Number of suspect images per class
    suspect_per_class = stats.groupby("class")["suspect"].sum()

    # Combined into a single DataFrame
    df = pd.concat([total_per_class, suspect_per_class], axis=1).reset_index()
    df = df[df["class"].isin(CLASS_ORDER)]
    # Converted counts to percentages, guarding against division by zero
    df["percent_suspect"] = (
        df["suspect"] / df["total"].replace(0, np.nan) * 100.0
    ).fillna(0.0)

    categories = df["class"].tolist()
    percents = df["percent_suspect"].tolist()

    out_path = RESULTS_DIR / "quality_flags_pct.png"
    bar_plot_percent(
        categories=categories,
        percents=percents,
        title="Percentage of suspect / low-quality images per class",
        ylabel="Percentage of images (%)",
        output_path=out_path,
    )


def plot_good_vs_suspect_overall_pie():
    
    """
    Overall good vs suspect pie chart for the entire dataset.

    Good = not flagged as suspect.
    Suspect = too dark / too bright / low contrast / failed to load.
    """
    stats = pd.read_csv(RAW_STATS_PATH)
    n_suspect = int(stats["suspect"].sum())
    n_good = int((~stats["suspect"]).sum())
    total = n_good + n_suspect

    counts = [n_good, n_suspect]
    labels = ["Typical / OK", "Suspect / low quality"]

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    wedges, _texts, _autotexts = ax.pie(
        counts,
        labels=None,
        autopct=_autopct_with_counts(counts, decimals=1),
        startangle=90,
        textprops={"fontsize": 9},
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    ax.axis("equal")
    ax.set_title(f"Typical vs suspect images (N={total})")

    # Legend beneath the pie
    fig.legend(
        handles=wedges,
        labels=labels,
        loc="lower center",
        ncol=2,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout()
    out_path = RESULTS_DIR / "good_vs_suspect_overall_pie.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------
# Resolution distribution plots
# ---------------------------------------------------------------------

def plot_resolution_distribution_topk(max_bins: int = 6):
    
    """
    Bar plot of the most common raw resolutions.

    If there are more than max_bins distinct resolutions, the last bar
    is "Other" (all remaining resolutions). This keeps the figure
    readable while still being honest about the tail of resolutions.
    """
    if not RAW_RESOLUTION_SUMMARY_PATH.exists():
        print("raw_resolution_summary.csv not found; skipping resolution plot.")
        return

    res = pd.read_csv(RAW_RESOLUTION_SUMMARY_PATH)
    if res.empty:
        print("No resolution summary to plot.")
        return

    # Sorted by descending count so the most common sizes appear first
    res = res.sort_values("count", ascending=False)
    total = int(res["count"].sum())

    if len(res) > max_bins:
        
        # Took the top max_bins - 1 resolutions explicitly
        top = res.head(max_bins - 1).copy()
        # Merged the rest into an "Other" category
        other_count = int(res["count"].iloc[max_bins - 1 :].sum())

        # Dummy row representing Other
        other_row = pd.DataFrame(
            [{"width": -1, "height": -1, "count": other_count}]
        )
        top = pd.concat([top, other_row], ignore_index=True)

        labels = [
            f"{int(w)}×{int(h)}"
            for w, h in zip(top["width"][:-1], top["height"][:-1])
        ] + ["Other"]
        counts = top["count"].tolist()
    else:
        # If the number of resolutions is small, show all of them
        labels = [
            f"{int(w)}×{int(h)}"
            for w, h in zip(res["width"], res["height"])
        ]
        counts = res["count"].tolist()

    title = f"Most common raw resolutions (top {max_bins}, N={total})"
    out_path = RESULTS_DIR / "resolution_distribution_topk.png"
    bar_plot_counts(
        categories=labels,
        counts=counts,
        title=title,
        ylabel="Number of images",
        output_path=out_path,
    )


def plot_resolution_distribution_all_pie():
    
    """
    Pie chart of ALL distinct raw resolutions.

    Each (width x height) gets its own slice. Labels are shown
    in a legend next to the pie, so everything remains readable
    even if there are many distinct resolutions.
    """
    if not RAW_RESOLUTION_SUMMARY_PATH.exists():
        print("raw_resolution_summary.csv not found; skipping full resolution pie.")
        return

    res = pd.read_csv(RAW_RESOLUTION_SUMMARY_PATH)
    if res.empty:
        print("raw_resolution_summary.csv is empty; skipping full resolution pie.")
        return

    # Sort by count so legend is ordered from most to least common
    res = res.sort_values("count", ascending=False)
    total = int(res["count"].sum())

    labels = [f"{int(w)}×{int(h)}" for w, h in zip(res["width"], res["height"])]
    counts = res["count"].tolist()

    fig, ax = plt.subplots(figsize=(8,8))

    wedges, _texts, _autotexts = ax.pie(
        counts,
        labels=None,  # use legend instead of on-slice labels
        autopct=lambda pct: f"{pct:.1f}%",
        startangle=90,
        textprops={"fontsize": 8},
        wedgeprops={"linewidth": 0.8, "edgecolor": "white"},
    )
    ax.axis("equal")
    ax.set_title(f"Raw resolution distribution (all sizes, N={total})", fontsize=12)

    # Legend with one entry per resolution
    ax.legend(
        wedges,
        labels,
        title="Resolution (W×H)",
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=8,
        title_fontsize=9,
    )

    fig.tight_layout()
    out_path = RESULTS_DIR / "resolution_distribution_all_pie.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------
# Duplicate removal effect
# ---------------------------------------------------------------------

def plot_duplicates_effect_bar():
    """
    Bar plot: unique images kept vs exact duplicates removed (SHA1-based).

    Uses duplicate_summary.csv (one row per SHA1 group) plus the final
    dataset_summary.csv to approximate how many images existed before
    vs after deduplication.
    """
    if not DUPLICATE_SUMMARY_PATH.exists():
        print("duplicate_summary.csv not found; skipping duplicates plot.")
        return

    dup_summary = pd.read_csv(DUPLICATE_SUMMARY_PATH)
    # Number of duplicate *entries* removed = sum(max(n_files - 1, 0))
    duplicates_removed = int((dup_summary["n_files"] - 1).clip(lower=0).sum())

    summary = pd.read_csv(DATASET_SUMMARY_PATH)
    n_after = int(summary["count"].sum())        # after dedup
    n_before = n_after + duplicates_removed      # approximate number before dedup

    labels = ["Unique images kept", "Exact duplicates removed"]
    counts = [n_after, duplicates_removed]

    title = (
        f"Effect of SHA1-based deduplication "
        f"(before ≈ {n_before}, after = {n_after})"
    )
    out_path = RESULTS_DIR / "duplicates_effect_bar.png"
    bar_plot_counts(
        categories=labels,
        counts=counts,
        title=title,
        ylabel="Number of images",
        output_path=out_path,
    )


# ---------------------------------------------------------------------
# Near-duplicate cluster sizes (pHash)
# ---------------------------------------------------------------------

def plot_near_duplicate_cluster_sizes():
    
    """
    Bar plot showing how many pHash near-duplicate clusters of each size exist.

    Here, a cluster is a connected component in the pHash similarity
    graph created in dataset_prep.py. The figure shows whether
    near-duplicates mostly occur in pairs, triplets, etc.
    """
    if not NEAR_DUP_CLUSTERS_PATH.exists():
        print("near_duplicate_clusters_summary.csv not found; skipping near-duplicate plot.")
        return

    clusters = pd.read_csv(NEAR_DUP_CLUSTERS_PATH)
    if clusters.empty:
        print("No near-duplicate clusters to plot.")
        return

    # size_counts[index = cluster_size, value = how many clusters of that size]
    size_counts = clusters["cluster_size"].value_counts().sort_index()
    sizes = size_counts.index.tolist()
    counts = size_counts.values.tolist()

    # Computed how many images are involved in any near-duplicate cluster
    sizes_arr = size_counts.index.to_numpy()
    counts_arr = size_counts.values
    n_images_in_clusters = int((sizes_arr * counts_arr).sum())

    stats = pd.read_csv(RAW_STATS_PATH)
    n_total = int(len(stats))

    title = (
        "pHash near-duplicate cluster sizes\n"
        f"(images in clusters = {n_images_in_clusters} / {n_total})"
    )
    out_path = RESULTS_DIR / "near_duplicate_cluster_sizes.png"
    bar_plot_counts(
        categories=[str(s) for s in sizes],
        counts=counts,
        title=title,
        ylabel="Number of clusters",
        output_path=out_path,
    )


# ---------------------------------------------------------------------
# Split x class heatmap from dataset_summary.csv
# ---------------------------------------------------------------------

def plot_split_class_heatmap():
    
    """
    Heatmap of counts per (class × split) from dataset_summary.csv.

    Cleaner, publication-style version:
      * fixed ordering of classes and splits
      * labelled colour bar ("Number of images")
      * integer annotations in each cell with automatic text colour
    """
    df = pd.read_csv(DATASET_SUMMARY_PATH)
    df = df[df["class"].isin(CLASS_ORDER)].copy()

    # Pivot to a matrix with rows=class, columns=split, values=counts
    pivot = df.pivot(index="class", columns="split", values="count")
    # Enforced consistent row/column orders
    pivot = pivot.reindex(index=CLASS_ORDER)
    pivot = pivot.reindex(columns=["train", "val", "test"])
    pivot = pivot.fillna(0)

    classes_pretty = _prettify_categories(pivot.index.tolist())
    splits = pivot.columns.tolist()

    fig, ax = plt.subplots(figsize=(5.5, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")

    # Coloured bar with explicit label
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Number of images")

    ax.set_xticks(np.arange(len(splits)))
    ax.set_xticklabels([s.capitalize() for s in splits])
    ax.set_yticks(np.arange(len(classes_pretty)))
    ax.set_yticklabels(classes_pretty)

    ax.set_xlabel("Split")
    ax.set_ylabel("Class")
    ax.set_title("Number of images per class and split")

    vmax = pivot.values.max() if pivot.values.size > 0 else 1

    # Annotated each cell with its integer count
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = int(pivot.values[i, j])
            # White text on dark cells, black text on light cells
            text_color = "white" if value > vmax * 0.5 else "black"
            ax.text(
                j,
                i,
                f"{value}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    fig.tight_layout()
    out_path = RESULTS_DIR / "split_class_heatmap.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ---------------------------------------------------------------------
# Example image grids
# ---------------------------------------------------------------------

def create_example_per_class_grid(out_path=None, seed=42):
    
    """
  
    The function:
      - samples one non-suspect image per class (if available),
      - resizes them to 256×256,
      - concatenates them into a single horizontal strip,
      - overlays class labels above each patch.
    """
    np.random.seed(seed)
    stats = pd.read_csv(RAW_STATS_PATH)

    images = []
    titles = []
    for cls in CLASS_ORDER:
        subset = stats[stats["class"] == cls]
        if subset.empty:
            continue
        # Preferred non-suspect images if available for this class
        good = subset[~subset["suspect"]] if "suspect" in subset.columns else subset
        if good.empty:
            # If all images are suspect, fall back to any available image
            good = subset
        # Sample exactly one path for this class (seeded for reproducibility)
        path = good.sample(1, random_state=seed)["orig_path"].iloc[0]
        images.append(path)
        titles.append("No tumour" if cls == "notumor" else cls.capitalize())

    if not images:
        print("No images found to build per-class grid.")
        return

    n = len(images)
    thumb_size = (256, 256)
    grid_width = n * thumb_size[0]
    grid_height = thumb_size[1]

    # New blank canvas for the horizontal strip
    grid = Image.new("RGB", (grid_width, grid_height), color=(0, 0, 0))

    # Pasted each image next to each other
    for i, path in enumerate(images):
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize(thumb_size)
            grid.paste(img, (i * thumb_size[0], 0))
        except Exception as e:
            print(f"Failed to load {path} for per-class grid: {e}")

    if out_path is None:
        out_path = RESULTS_DIR / "examples_per_class.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrapped the composite image in matplotlib so I can add text labels
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(grid)
    ax.axis("off")

    # Wrote class names above each panel
    for i, title in enumerate(titles):
        x = (i + 0.5) * thumb_size[0]
        y = 18  # small offset from the top border
        ax.text(
            x,
            y,
            title,
            color="white",
            ha="center",
            va="top",
            fontsize=10,
            bbox=dict(facecolor="black", alpha=0.6, pad=3),
        )

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved per-class example grid: {out_path}")


def create_example_good_vs_weird_grid(
    n_good=4,
    n_weird=4,
    out_path=None,
    seed=42,
):
    """
    Two-row grid: typical high-quality vs atypical / suspect images.

    Row 1 (top): "good" images
       - resolution exactly 512x512 (the dominant size in the dataset)
       - NOT flagged as suspect in raw_image_stats.csv
         (i.e. not too dark / too bright / low contrast / failed)

    Row 2 (bottom): "weird" or atypical images
       - either non-512x512 resolution OR
       - flagged as suspect in raw_image_stats.csv

    Important point: visually some 'weird' images may still look OK to a
    human. They appear in the bottom row simply because they violate at
    least one of the *automatic* heuristics above (size or intensity),
    not because they are obviously bad to the eye.

    The text boxes live outside the image area, with arrows pointing
    into each row (similar to the example figure in the brief).
    """
    np.random.seed(seed)
    stats = pd.read_csv(RAW_STATS_PATH)

    # Typical images: exactly 512x512 and not suspect
    good_mask = (
        (stats["width"] == 512)
        & (stats["height"] == 512)
        & (~stats["suspect"])
    )
    good_candidates = stats[good_mask]

    # Atypical: non-512x512 OR suspect for intensity reasons
    weird_mask = (
        (stats["width"] != 512)
        | (stats["height"] != 512)
        | (stats["suspect"])
    )
    weird_candidates = stats[weird_mask]

    # Fallbacks if there are not enough candidates in either group
    if len(good_candidates) < n_good:
        # Loosen the condition: any non-suspect image counts as "good"
        good_candidates = stats[~stats["suspect"]]

    if len(weird_candidates) < n_weird:
        # If still not enough, use only suspect images; if none exist,
        # fall back to all images.
        weird_candidates = stats[stats["suspect"]] if stats["suspect"].any() else stats

    # Sample the final sets of image paths
    good_rows = (
        good_candidates
        .sample(min(n_good, len(good_candidates)), random_state=seed)
        .reset_index(drop=True)
    )
    weird_rows = (
        weird_candidates
        .sample(min(n_weird, len(weird_candidates)), random_state=seed + 1)
        .reset_index(drop=True)
    )

    good_paths = good_rows["orig_path"].tolist()
    weird_paths = weird_rows["orig_path"].tolist()

    max_cols = max(len(good_paths), len(weird_paths))
    thumb_size = (224, 224)

    grid_width = max_cols * thumb_size[0]
    grid_height = 2 * thumb_size[1]

    # Blank canvas: first row = good, second row = weird
    grid = Image.new("RGB", (grid_width, grid_height), color=(0, 0, 0))

    # Small helper to paste one row of thumbnails into the grid
    def paste_row(img_paths, row_idx):
        for col_idx, p in enumerate(img_paths):
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize(thumb_size)
                grid.paste(img, (col_idx * thumb_size[0], row_idx * thumb_size[1]))
            except Exception as e:
                print(f"Failed to load {p} for grid: {e}")

    # Top row = typical high-quality images
    paste_row(good_paths, row_idx=0)
    # Bottom row = atypical / suspect images
    paste_row(weird_paths, row_idx=1)

    if out_path is None:
        out_path = RESULTS_DIR / "examples_good_vs_weird.png"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Wrapped into matplotlib so I can add explanatory text and arrows
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(grid)
    ax.axis("off")

      # Arrow and text for top row (typical images)
    ax.annotate(
        "Typical high-quality images\n"
        "• Resolution 512×512 (dominant in dataset)\n"
        "• Not flagged as too dark / too bright / low contrast",
        xy=(0.03, 0.75),              
        xycoords="axes fraction",
        xytext=(-0.05, 0.75),         
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            color="white",
            lw=1.5,
        ),
        ha="right",
        va="center",
        color="white",
        bbox=dict(facecolor="black", alpha=0.7, pad=4),
        clip_on=False,
    )
 # Arrow and text for bottom row (suspect images)
    ax.annotate(
        "Atypical / suspect images\n"
        "• Either non-512×512 resolution OR\n"
        "• Flagged as too dark / too bright / low contrast / failed to load",
        xy=(0.03, 0.25),            
        xycoords="axes fraction",
        xytext=(-0.05, 0.25),         
        textcoords="axes fraction",
        arrowprops=dict(
            arrowstyle="->",
            color="white",
            lw=1.5,
        ),
        ha="right",
        va="center",
        color="white",
        bbox=dict(facecolor="black", alpha=0.7, pad=4),
        clip_on=False,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved example good vs weird grid: {out_path}")

# ---------------------------------------------------------------------
# main
# ---------------------------------------------------------------------

def main():
    
    """
    Orchestrates generation of all dataset figures.

    Assumes that dataset_prep.py has already been run and that all
    required CSV artefacts are present in results/.
    """
    if not DATASET_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"{DATASET_SUMMARY_PATH} not found. Run dataset_prep.py first."
        )
    if not RAW_STATS_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_STATS_PATH} not found. Run dataset_prep.py first."
        )

    # Print total images after preprocessing so it’s visible in the console
    summary = pd.read_csv(DATASET_SUMMARY_PATH)
    total_images = int(summary["count"].sum())
    per_split = summary.groupby("split")["count"].sum()
    print(
        f"Total images after deduplication and 70/15/15 split: "
        f"{total_images}"
    )
    for split, n in per_split.items():
        print(f"       {split}: {int(n)}")

    # original Kaggle Training/Testing folders
    if RAW_CLASS_COUNTS_PATH.exists():
        plot_kaggle_training_testing_pies()

    # my own 70/15/15 splits
    plot_our_split_pies()

    # Distributions and quality
    plot_overall_class_distribution_pct()
    plot_quality_flags_pct()
    plot_good_vs_suspect_overall_pie()

    # Resolution and deduplication and near-duplicates (plots)
    if RAW_RESOLUTION_SUMMARY_PATH.exists():
        plot_resolution_distribution_topk()
        plot_resolution_distribution_all_pie()
    if DUPLICATE_SUMMARY_PATH.exists():
        plot_duplicates_effect_bar()
    if NEAR_DUP_CLUSTERS_PATH.exists():
        plot_near_duplicate_cluster_sizes()

    # Split x class heatmap (matrix visual from dataset_summary.csv)
    plot_split_class_heatmap()

    # Example image grids for qualitative illustration
    create_example_per_class_grid()
    create_example_good_vs_weird_grid()


if __name__ == "__main__":
    main()
