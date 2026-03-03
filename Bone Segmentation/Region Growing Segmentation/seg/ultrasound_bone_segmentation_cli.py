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
from skimage.morphology import dilation, disk  # make sure this import is at the top
from skimage.morphology import opening, closing, dilation, disk, footprint_rectangle


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
DEFAULT_TOLERANCE = 20      # gray-level tolerance for flood
MAX_SEEDS_DEFAULT = 5

# Mask cleaning defaults
CLEAN_Y_BAND = 35
CLEAN_MIN_AREA = 200
OPEN_DISK_RADIUS = 1
CLOSE_RECT_WIDTH = 25

# Seed snap
SNAP_WINDOW = 7

# Final mask thickness (after snake)
SNAKE_DILATE_RADIUS = 3

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
        # ignore clicks outside the axes or without data coords
        if event.xdata is None or event.ydata is None:
            return
        if len(seeds) >= max_seeds:
            return

        y, x = int(event.ydata), int(event.xdata)
        print(f"Seed added: ({y}, {x})")
        seeds.append((y, x))

        # Visual feedback: smaller hollow red circle so you can see the pixel
        ax = event.inaxes
        if ax is not None:
            ax.plot(
                x,
                y,
                "ro",
                markersize=3,          # <<< smaller marker
                markerfacecolor="none",
                markeredgewidth=0.8,
            )
            event.canvas.draw_idle()

        if len(seeds) >= max_seeds:
            plt.close(event.canvas.figure)

    # slightly larger figure for better precision
    fig, ax = plt.subplots(figsize=(6, 6))
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




def clean_mask(mask, seeds, y_band=CLEAN_Y_BAND, min_area=CLEAN_MIN_AREA,
               open_r=OPEN_DISK_RADIUS, close_w=CLOSE_RECT_WIDTH):
    """
    Clean the combined region-growing mask in a seed-robust way.

    - Restrict to vertical band around mean seed depth (y_band)
    - Opening (disk(open_r)) to remove speckles
    - Horizontal closing (1 x close_w) to bridge gaps along x
    - Keep largest connected component with area >= min_area
    """
    mask_bin = (mask > 0).astype(np.uint8)

    h, w = mask_bin.shape
    mean_y = int(np.mean([s[0] for s in seeds])) if len(seeds) > 0 else h // 2

    y_min = max(0, mean_y - y_band)
    y_max = min(h, mean_y + y_band)

    band = np.zeros_like(mask_bin, dtype=bool)
    band[y_min:y_max, :] = True
    mask_band = (mask_bin.astype(bool) & band).astype(np.uint8)

    opened = opening(mask_band, disk(open_r)).astype(np.uint8)

    # bridge gaps along x
    closed = closing(opened, footprint_rectangle((1, close_w))).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask_bin, dtype=np.uint8)

    # collect CC labels under seed points
    seed_labels = set()
    for (r, c) in seeds:
        if 0 <= r < h and 0 <= c < w:
            lbl = labels[r, c]
            if lbl != 0:
                seed_labels.add(lbl)

    if not seed_labels:
        return np.zeros_like(mask_bin, dtype=np.uint8)

    # keep only seed-connected components that are big enough
    keep = np.zeros_like(mask_bin, dtype=bool)
    for lbl in seed_labels:
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            keep |= (labels == lbl)

    return (keep.astype(np.uint8) * 255)




def active_contour_refinement_from_mask(img, mask):
    """
    Extracts the longest contour from the mask and refines it with an
    edge-based active contour (snake).
    """
    # Smooth the image a bit
    smoothed = gaussian(img, sigma=1.0)

    # Find all contours in the binary mask
    contours = find_contours(mask.astype(float), level=0.5)
    if not contours:
        raise ValueError("No contour found in the mask.")

    # Use the longest contour as initialization
    init = max(contours, key=len)

    # Edge map for the snake to follow
    edges = sobel(smoothed)

    # Edge-based snake: follow gradients, shrink less
    snake = active_contour(
        edges,
        init,
        alpha=0.0015,   # internal tension (lower -> less shrinking)
        beta=0.3,      # smoothness (lower -> can bend more)
        gamma=0.01,    # time step
        w_line=0.0,    # ignore absolute intensity
        w_edge=1.0,    # follow edges instead
        # max_px_move=1.0,
        # max_num_iter=500,
        # convergence=0.1,
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

def snap_seeds_to_bright_line(img, seeds, window=5):
    """
    For each seed (y, x), search vertically in [y-window, y+window]
    for the brightest pixel and move the seed there.
    img: preprocessed image (uint8 or float) used for region growing.
    """
    snapped = []
    h, w = img.shape
    for (y, x) in seeds:
        y0 = max(0, y - window)
        y1 = min(h - 1, y + window)
        col = img[y0:y1+1, x]

        if col.size == 0:
            snapped.append((y, x))
            continue

        # index of max in this vertical segment
        dy = int(np.argmax(col))
        new_y = y0 + dy
        snapped.append((new_y, x))
    return snapped

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
    parser.add_argument("--y_band", type=int, default=CLEAN_Y_BAND, help="Vertical +/- band (pixels) around mean seed depth for cleaning.")
    parser.add_argument("--min_area", type=int, default=CLEAN_MIN_AREA, help="Minimum component area to keep after cleaning.")
    parser.add_argument("--open_r", type=int, default=OPEN_DISK_RADIUS, help="Opening disk radius (speckle removal).")
    parser.add_argument("--close_w", type=int, default=CLOSE_RECT_WIDTH, help="Closing rectangle width (bridge gaps along x).")
    parser.add_argument("--snap_window", type=int, default=SNAP_WINDOW, help="Vertical window for snapping seeds to bright ridge.")
    parser.add_argument("--snake_dilate", type=int, default=SNAKE_DILATE_RADIUS, help="Final dilation radius for snake mask thickness.")

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

    # NEW: snap seeds to local bright ridge
    seeds = snap_seeds_to_bright_line(pre, seeds, window=args.snap_window)
    print("Snapped seeds:", seeds)
    
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
    mask_clean = clean_mask(combined_mask,seeds,y_band=args.y_band,min_area=args.min_area,open_r=args.open_r,close_w=args.close_w)

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

   # 7) Build final mask:
    #    take snake region, thicken it a bit, and clamp it inside the cleaned mask.

    # Mask from snake contour (0/255)
    snake_mask = create_mask_from_contour(pre.shape, snake)

    # Convert both to {0,1}
    clean_bin = (mask_clean > 0).astype(np.uint8)
    snake_bin = (snake_mask > 0).astype(np.uint8)

    # Slightly thicken the snake region so it covers seeds & realistic bone thickness
    # radius=3 is a good starting point; you can tune to 2–4
    snake_thick = dilation(snake_bin, disk(args.snake_dilate)).astype(np.uint8)

    # Clamp to region-growing prior: only keep pixels that are in BOTH
    final_bin = (snake_thick & clean_bin).astype(np.uint8)

    # Back to 0/255 for saving
    final_mask = final_bin * 255

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output_mask_path)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(args.output_mask_path, final_mask)
    print(f"Saved segmentation mask to: {args.output_mask_path}")


if __name__ == "__main__":
    main()
