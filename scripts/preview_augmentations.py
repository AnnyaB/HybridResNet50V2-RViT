"""
preview_augmentations.py
------------------------
Purpose:
    Produces a visual preview of the augmentation techniques (rotation,
    flip, brightness adjustment) that will later be applied dynamically
    during model training.

Rationale:
    Data augmentation improves model generalisation by simulating natural
    variations that can occur in real MRI scans (slight rotations, mirrored
    orientations, or varying illumination). In this project, augmentation
    is performed only during training to preserve dataset integrity while
    still providing robustness. This script visually demonstrates these
    transformations.

Dependencies:
    - Pillow (PIL)
    - matplotlib
"""

from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
from pathlib import Path

# Path to one representative training image (example from glioma class)
# Choosin' any class sample; this image is used only for visual demonstration.
SAMPLE_IMAGE_PATH = Path("data/processed/train/glioma")

def select_sample_image():
    """
    Selects the first available image from the specified folder.
    """
    for img_file in SAMPLE_IMAGE_PATH.glob("*.jpg"):
        return img_file
    raise FileNotFoundError("No sample images found in the specified folder.")


def main():
    # Step 1: Loadin' one sample image for visual augmentation demonstration
    img_path = select_sample_image()
    img = Image.open(img_path).convert("RGB")

    # Step 2: Applyin' representative augmentations
    rotated = img.rotate(12)  # rotation by +12°, typical range ±10–15°
    flipped = img.transpose(Image.FLIP_LEFT_RIGHT)  # horizontal flip
    bright = ImageEnhance.Brightness(img).enhance(1.2)  # brightness ×1.2

    # Step 3: Plottin' all variations side by side
    fig, axes = plt.subplots(1, 4, figsize=(12, 4))
    variations = [
        ("Original", img),
        ("Rotated (+12°)", rotated),
        ("Flipped", flipped),
        ("Bright ×1.2", bright),
    ]

    for ax, (title, image) in zip(axes, variations):
        ax.imshow(image)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    plt.tight_layout()

    # Step 4: Savin' preview for documentation
    output_path = Path("results/augmentation_preview.png")
    plt.savefig(output_path, dpi=300)
    plt.show()

    print(f"[INFO] Augmentation preview saved to {output_path.resolve()}")

if __name__ == "__main__":
    main()
