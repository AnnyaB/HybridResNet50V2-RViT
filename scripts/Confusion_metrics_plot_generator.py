import numpy as np
import matplotlib.pyplot as plt

# Confusion matrix
cm = np.array([
    [295, 3,   1,   0],
    [0,   303, 1,   0],
    [0,   2,   298, 0],
    [2,   1,   0,   378]
])

classes = ["Glioma", "Meningioma", "Pituitary", "No Tumor"]

fig, ax = plt.subplots(figsize=(7.5, 7.5))

# Use the same clean blue style as the paper
im = ax.imshow(cm, cmap=plt.cm.Blues)

# Colorbar (simple, unobtrusive)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=11)

# Axis ticks & labels
ax.set_xticks(np.arange(len(classes)))
ax.set_yticks(np.arange(len(classes)))
ax.set_xticklabels(classes, fontsize=12)
ax.set_yticklabels(classes, fontsize=12)

ax.set_xlabel("Predicted label", fontsize=13)
ax.set_ylabel("True label", fontsize=13)
ax.set_title("Confusion Matrix", fontsize=14, pad=10)

# Square cells
ax.set_aspect("equal")

# Annotate cells (white on dark, black on light)
thresh = cm.max() / 2.0
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i, cm[i, j],
            ha="center", va="center",
            fontsize=13,
            fontweight="bold",
            color="white" if cm[i, j] > thresh else "black"
        )

# Remove spines for cleaner look
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()



# Save figure (paper-quality)
plt.savefig(
    "Confusion_Matrix__Without_PFDB-GSTEB.png",
    dpi=300,
    bbox_inches="tight"
)

# Optional: also save as PDF (recommended for journals)
plt.savefig(
    "Confusion_Matrix__Without_PFDB-GSTEB.pdf",
    bbox_inches="tight"
)



plt.show()
