"""
visualize_preprocessing.py — Side-by-side preview of all preprocessing methods.

Shows the original cropped frame alongside each preprocessed version so you
can visually assess what each method does before running segmentation.

Usage
-----
  # All methods on one frame (interactive window)
  python visualize_preprocessing.py --image_path "Dataset/Patient1/IMG_frames/image_383229031802_f294.png"

  # Selected methods only
  python visualize_preprocessing.py --image_path "..." --methods baseline wavelet_bayes_soft fft_lowpass svd_global

  # Save figure instead of displaying
  python visualize_preprocessing.py --image_path "..." --save_path "preview.png"
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]

if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from preprocessing_api import preprocess, list_methods

# Default crop matches the segmentation CLI hard-coded box
CROP_BOX = (200, 100, 800, 700)   # x_min, y_min, x_max, y_max


def load_and_crop(image_path: Path, crop_box: tuple = CROP_BOX) -> np.ndarray:
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    x_min, y_min, x_max, y_max = crop_box
    return img[y_min:y_max, x_min:x_max]


def make_grid(img_cropped: np.ndarray,
              methods: list,
              cols: int = 4) -> plt.Figure:
    """
    Build a grid figure: first panel is the raw cropped original,
    then one panel per preprocessing method.
    """
    panels   = [("original (raw)", img_cropped)]
    for method in methods:
        out = preprocess(img_cropped, method)
        panels.append((method, out))

    n      = len(panels)
    rows   = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3.8, rows * 3.8))
    axes = np.array(axes).ravel()

    for ax, (title, img) in zip(axes, panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=8, pad=4)
        ax.axis("off")

    # Hide any unused axes
    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(
        f"Preprocessing comparison  —  cropped {img_cropped.shape[1]}×{img_cropped.shape[0]} px",
        fontsize=10, y=1.01,
    )
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Visual comparison of preprocessing methods on a single frame."
    )
    parser.add_argument(
        "--image_path",
        type=Path,
        required=True,
        help="Path to the input ultrasound frame (PNG).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help=(
            "Preprocessing methods to show. "
            f"Default: all. Available: {list_methods()}"
        ),
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in the grid. Default: 4.",
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=None,
        help="Save figure to this path instead of displaying it.",
    )
    parser.add_argument(
        "--crop_box",
        nargs=4,
        type=int,
        default=list(CROP_BOX),
        metavar=("X_MIN", "Y_MIN", "X_MAX", "Y_MAX"),
        help=f"Crop box in original-image coordinates. Default: {CROP_BOX}.",
    )
    args = parser.parse_args()

    # Validate methods
    all_methods = list_methods()
    methods = args.methods if args.methods else all_methods
    bad = [m for m in methods if m not in all_methods]
    if bad:
        raise ValueError(f"Unknown methods: {bad}. Available: {all_methods}")

    crop_box = tuple(args.crop_box)
    img_cropped = load_and_crop(args.image_path.resolve(), crop_box)
    print(f"Image loaded and cropped: {img_cropped.shape}  ({len(methods)} methods + original)")

    fig = make_grid(img_cropped, methods, cols=args.cols)

    if args.save_path:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=130, bbox_inches="tight")
        print(f"Saved to {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
