import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import flood, active_contour
from skimage.filters import gaussian, sobel
from skimage.draw import polygon
from skimage.morphology import opening, disk
from skimage.exposure import equalize_adapthist
from skimage.measure import find_contours


# ==========================
# Global parameters to tune
# ==========================

# Crop coordinates: img[y_min:y_max, x_min:x_max]
CROP_Y_MIN, CROP_Y_MAX = 100, 700
CROP_X_MIN, CROP_X_MAX = 200, 800

# Pre-processing
CLAHE_CLIP_LIMIT = 0.01     # for equalize_adapthist
GAUSSIAN_KERNEL = (7, 7)

# Region growing
DEFAULT_TOLERANCE = 15      # gray-level tolerance for flood
MAX_SEEDS_DEFAULT = 5

# Active contour (snake) parameters
SNAKE_ALPHA = 0.0005
SNAKE_BETA = 2.0
SNAKE_GAMMA = 0.01
SNAKE_W_LINE = 0.5
SNAKE_W_EDGE = 0.0


# ==========================
# Utility functions
# ==========================

def load_image(path: str) -> np.ndarray:
    """Load grayscale ultrasound image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def crop_ultrasound_region(img: np.ndarray) -> np.ndarray:
    """Crop approximate facial/jaw region (manually tuned box)."""
    return img[CROP_Y_MIN:CROP_Y_MAX, CROP_X_MIN:CROP_X_MAX]


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    CLAHE (adaptive histogram equalization) + Gaussian blur.
    Input and output are uint8.
    """
    img_norm = img / 255.0
    clahe = equalize_adapthist(img_norm, clip_limit=CLAHE_CLIP_LIMIT)
    clahe_8bit = (clahe * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(clahe_8bit, GAUSSIAN_KERNEL, 0)
    return blurred


def get_multiple_seeds_from_click(image: np.ndarray, max_seeds: int = MAX_SEEDS_DEFAULT):
    """
    Let the user click multiple seed points on the image.
    Returns a list of (row, col) tuples.
    """
    seeds = []

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        if len(seeds) >= max_seeds:
            return

        y, x = int(event.ydata), int(event.xdata)
        print(f"Seed added: ({y}, {x})")
        seeds.append((y, x))

        # Visual feedback
        event.inaxes.plot(x, y, "ro")
        event.canvas.draw()

        if len(seeds) >= max_seeds:
            plt.close(event.canvas.figure)

    fig, ax = plt.subplots()
    ax.imshow(image, cmap="gray")
    ax.set_title(f"Click up to {max_seeds} seed points\n(close window to finish)")
    cid = fig.canvas.mpl_connect("button_press_event", onclick)
    plt.show()
    fig.canvas.mpl_disconnect(cid)

    return seeds


def region_growing(img: np.ndarray, seed, tolerance: int):
    """
    Region growing using skimage.segmentation.flood.
    seed = (row, col)
    Returns a boolean mask.
    """
    mask = flood(img, seed_point=seed, tolerance=tolerance)
    return mask


def clean_mask(mask, seeds, y_band=40, min_area=200):
    """
    Clean the combined region-growing mask.

    - Restrict to a vertical band around the average seed depth.
    - Morphological opening with a small disk.
    - Keep only connected components that:
        * are larger than min_area, AND
        * contain at least one of the seed points.
    """
    # Binary 0/1
    mask_bin = (mask > 0).astype(np.uint8)

    # 1) Restrict to vertical band around seeds
    if len(seeds) > 0:
        mean_y = int(np.mean([s[0] for s in seeds]))    # seeds are (row, col)
        y_min = max(0, mean_y - y_band)
        y_max = min(mask_bin.shape[0], mean_y + y_band)
        band = np.zeros_like(mask_bin, dtype=bool)
        band[y_min:y_max, :] = True
        mask_bin = (mask_bin.astype(bool) & band).astype(np.uint8)

    # 2) Light morphological opening to remove tiny speckles
    opened = opening(mask_bin, disk(1)).astype(np.uint8)  # disk(1) = gentler than disk(3)

    # 3) Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)

    keep = np.zeros_like(mask_bin, dtype=bool)

    # Precompute label at each seed
    seed_labels = set()
    for (y, x) in seeds:
        if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
            seed_labels.add(labels[y, x])

    for lbl in range(1, num_labels):  # skip background label 0
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        if lbl in seed_labels:
            keep |= (labels == lbl)

    return keep.astype(np.uint8) * 255



def active_contour_refinement_from_mask(img: np.ndarray, mask: np.ndarray):
    """
    Extract the largest contour from the mask and refine it with an active contour.
    Returns (initial_contour, refined_snake), both as Nx2 arrays (row, col).
    """
    smoothed = gaussian(img, sigma=1)

    # Find contours
    contours = find_contours(mask.astype(float), level=0.5)
    if not contours:
        raise ValueError("No contour found in the mask.")

    # Use the longest contour as initialization
    init = max(contours, key=len)
    edges = sobel(smoothed)

    snake = active_contour(
        edges,
        init,
        alpha=SNAKE_ALPHA,
        beta=SNAKE_BETA,
        gamma=SNAKE_GAMMA,
        w_line=SNAKE_W_LINE,
        w_edge=SNAKE_W_EDGE,
    )
    return init, snake


def create_mask_from_contour(img_shape, contour) -> np.ndarray:
    """
    Rasterize a contour (row, col) into a filled binary mask.
    """
    mask = np.zeros(img_shape, dtype=np.uint8)
    rr, cc = polygon(contour[:, 0], contour[:, 1], img_shape)
    mask[rr, cc] = 255
    return mask


# ==========================
# Main script
# ==========================

def main():
    parser = argparse.ArgumentParser(
        description="Classical jaw-bone segmentation in a single ultrasound B-scan."
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input ultrasound frame (grayscale PNG/DICOM-converted PNG).",
    )
    parser.add_argument(
        "--output_mask_path",
        type=str,
        required=True,
        help="Path to save the binary bone mask PNG.",
    )
    parser.add_argument(
        "--max_seeds",
        type=int,
        default=MAX_SEEDS_DEFAULT,
        help=f"Maximum number of seed points to click (default: {MAX_SEEDS_DEFAULT}).",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE,
        help=f"Region-growing tolerance (default: {DEFAULT_TOLERANCE}).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="If set, show intermediate figures (preprocessed image, masks, contour).",
    )

    args = parser.parse_args()

    # 1) Load and crop
    img_full = load_image(args.image_path)
    img = crop_ultrasound_region(img_full)

    # 2) Preprocess
    pre = preprocess_image(img)

    if args.show:
        plt.figure()
        plt.imshow(pre, cmap="gray")
        plt.title("Preprocessed (cropped) image")
        plt.axis("off")
        plt.show()

    # 3) Interactive seed selection
    seeds = get_multiple_seeds_from_click(pre, max_seeds=args.max_seeds)
    if not seeds:
        raise RuntimeError("No seeds selected. Aborting segmentation.")

    # 4) Region growing from each seed and combine
    combined_mask = np.zeros_like(pre, dtype=bool)
    for seed in seeds:
        mask = region_growing(pre, seed, tolerance=args.tolerance)
        combined_mask |= mask  # logical OR

    if args.show:
        plt.figure()
        plt.imshow(pre, cmap="gray")
        plt.imshow(combined_mask, alpha=0.4)
        plt.title("Region-growing mask (before cleaning)")
        plt.axis("off")
        plt.show()
    # 5) Clean mask (opening + largest component)
    # mask_clean = clean_mask(combined_mask)
    mask_clean = clean_mask(combined_mask, seeds)
    #mask_clean = (mask_clean.astype(np.uint8) * 255)
    if args.show:
        plt.figure()
        plt.imshow(mask_clean, cmap="gray")
        plt.title("Region growing – cleaned mask")
        plt.axis("off")
        plt.show()

    # 6) Active contour refinement
    init, snake = active_contour_refinement_from_mask(pre, mask_clean)

    if args.show:
        plt.figure()
        plt.imshow(pre, cmap="gray")
        plt.plot(init[:, 1], init[:, 0], "--r", label="Initial contour")
        plt.plot(snake[:, 1], snake[:, 0], "-b", label="Refined snake")
        plt.legend()
        plt.title("Active contour refinement")
        plt.axis("off")
        plt.show()

    # 7) Create final mask from refined contour
    final_mask = create_mask_from_contour(pre.shape, snake)
    final_mask = (final_mask > 0).astype(np.uint8) * 255

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_mask_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(args.output_mask_path, final_mask)
    print(f"Saved segmentation mask to: {args.output_mask_path}")


if __name__ == "__main__":
    main()
