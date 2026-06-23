"""
Generate preprocessing strip figures for the report.

Produces two figures:
  1. preprocessing_strip.png    — 4-panel row: Original | Crop | CLAHE | CLAHE+Gaussian
  2. preprocessing_closeup.png  — 2-panel close-up of the bone region: Crop | CLAHE

Uses the exact same pipeline as ultrasound_bone_segmentation_cli.py:
  equalize_adapthist(clip_limit=0.01) → GaussianBlur((7,7))
"""

import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import equalize_adapthist

# --- Pipeline constants (must match ultrasound_bone_segmentation_cli.py) ---
CROP_Y_MIN, CROP_Y_MAX = 100, 700
CROP_X_MIN, CROP_X_MAX = 200, 800
CLAHE_CLIP_LIMIT = 0.01
GAUSSIAN_KERNEL = (7, 7)

# Close-up ROI within the cropped image (y, x) — where cortical bone typically appears
CLOSEUP_Y = (50, 250)
CLOSEUP_X = (50, 500)

DEFAULT_FRAME = os.path.join(
    os.path.dirname(__file__),
    "..", "Dataset", "Patient1", "IMG_frames",
    "image_283536217682_f062.png",
)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "report_figures")


def preprocess(img):
    """Replicate the pipeline preprocessing exactly."""
    cropped = img[CROP_Y_MIN:CROP_Y_MAX, CROP_X_MIN:CROP_X_MAX]
    clahe = equalize_adapthist(cropped / 255.0, clip_limit=CLAHE_CLIP_LIMIT)
    clahe_8 = (clahe * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(clahe_8, GAUSSIAN_KERNEL, 0)
    return cropped, clahe_8, blurred


def save_strip(img, cropped, clahe_8, blurred, out_path):
    # Draw crop box on a copy of the original
    orig_annotated = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.rectangle(
        orig_annotated,
        (CROP_X_MIN, CROP_Y_MIN), (CROP_X_MAX, CROP_Y_MAX),
        color=(255, 50, 50), thickness=4,
    )

    titles = ["Original frame\n(crop region in red)", "After crop", "After CLAHE", "After CLAHE + Gaussian"]
    images = [orig_annotated, cropped, clahe_8, blurred]
    cmaps = [None, "gray", "gray", "gray"]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, image, title, cmap in zip(axes, images, titles, cmaps):
        ax.imshow(image, cmap=cmap, vmin=0 if cmap else None, vmax=255 if cmap else None)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def save_closeup(cropped, clahe_8, out_path, closeup_y=CLOSEUP_Y, closeup_x=CLOSEUP_X):
    y0, y1 = closeup_y
    x0, x1 = closeup_x
    roi_orig = cropped[y0:y1, x0:x1]
    roi_clahe = clahe_8[y0:y1, x0:x1]

    titles = ["Cropped (no enhancement)", "After CLAHE"]
    images = [roi_orig, roi_clahe]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", default=DEFAULT_FRAME,
                        help="Path to input frame PNG")
    parser.add_argument("--out_dir", default=OUT_DIR,
                        help="Output directory for figures")
    parser.add_argument("--closeup_y", nargs=2, type=int, default=list(CLOSEUP_Y),
                        metavar=("Y0", "Y1"), help="Close-up ROI y range within crop")
    parser.add_argument("--closeup_x", nargs=2, type=int, default=list(CLOSEUP_X),
                        metavar=("X0", "X1"), help="Close-up ROI x range within crop")
    args = parser.parse_args()

    img = cv2.imread(args.frame, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {args.frame}")

    os.makedirs(args.out_dir, exist_ok=True)
    cropped, clahe_8, blurred = preprocess(img)

    closeup_y = tuple(args.closeup_y)
    closeup_x = tuple(args.closeup_x)

    save_strip(img, cropped, clahe_8, blurred,
               os.path.join(args.out_dir, "preprocessing_strip.png"))
    save_closeup(cropped, clahe_8,
                 os.path.join(args.out_dir, "preprocessing_closeup.png"),
                 closeup_y, closeup_x)


if __name__ == "__main__":
    main()
