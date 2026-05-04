import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

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

# Active contour parameters currently used by the implementation below.
# These mirror the existing literals so behavior stays unchanged.
ACTIVE_CONTOUR_SIGMA = 1.0
ACTIVE_CONTOUR_ALPHA = 0.0015
ACTIVE_CONTOUR_BETA = 0.3
ACTIVE_CONTOUR_GAMMA = 0.01
ACTIVE_CONTOUR_W_LINE = 0.0
ACTIVE_CONTOUR_W_EDGE = 1.0


# ==========================
# Utility functions
# ==========================

def load_image(path: str) -> np.ndarray:
    """Load grayscale ultrasound image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def crop_ultrasound_region(img: np.ndarray, crop_box) -> np.ndarray:
    """Crop approximate facial/jaw region (manually tuned box)."""
    x_min, y_min, x_max, y_max = crop_box
    return img[y_min:y_max, x_min:x_max]


def get_crop_box(args=None):
    """Return the crop box in original-image coordinates as [x_min, y_min, x_max, y_max]."""
    if args is None:
        return [CROP_X_MIN, CROP_Y_MIN, CROP_X_MAX, CROP_Y_MAX]
    return [args.crop_x_min, args.crop_y_min, args.crop_x_max, args.crop_y_max]


def preprocess_image(img: np.ndarray, clahe_clip_limit: float, gaussian_kernel) -> np.ndarray:
    """
    CLAHE (adaptive histogram equalization) + Gaussian blur.
    Input and output are uint8.
    """
    img_norm = img / 255.0
    clahe = equalize_adapthist(img_norm, clip_limit=clahe_clip_limit)
    clahe_8bit = (clahe * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(clahe_8bit, gaussian_kernel, 0)
    return blurred


def parse_patient_scan_frame_from_path(path: str):
    """
    Infer patient_id, scan_id, and frame_id from a path when possible.
    Missing values are returned as None.
    """
    path_obj = Path(path)
    patient_id = None
    scan_id = None
    frame_id = None

    for part in path_obj.parts:
        if patient_id is None and re.fullmatch(r"patient\d+", part, flags=re.IGNORECASE):
            patient_id = part
        if scan_id is None and re.fullmatch(r"scan\d+", part, flags=re.IGNORECASE):
            scan_id = part

    frame_match = re.search(r"frame_(\d+)", path_obj.name, flags=re.IGNORECASE)
    if frame_match:
        frame_id = int(frame_match.group(1))
    else:
        alt_match = re.search(r"_f(\d+)(?:_mask)?\.", path_obj.name, flags=re.IGNORECASE)
        if alt_match:
            frame_id = int(alt_match.group(1))

    return patient_id, scan_id, frame_id


def convert_seed_coords_to_original(seeds, crop_box):
    """Convert working-image seeds [(row, col), ...] to original-image coordinates."""
    if not seeds:
        return []

    x_min, y_min, _, _ = crop_box
    return [[int(col + x_min), int(row + y_min)] for row, col in seeds]


def convert_seed_coords_to_working(seeds, crop_box):
    """Convert original-image seeds [[x, y], ...] back to working-image [row, col]."""
    if not seeds:
        return []

    x_min, y_min, _, _ = crop_box
    working = []
    for point in seeds:
        if len(point) != 2:
            continue
        x, y = point
        working.append([int(y - y_min), int(x - x_min)])
    return working


def to_serializable(value):
    """Recursively convert NumPy and tuple values into JSON-serializable Python types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tuple):
        return [to_serializable(item) for item in value]
    if isinstance(value, list):
        return [to_serializable(item) for item in value]
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    return value


def maybe_make_relative_path(path: Path):
    """Return a relative path when practical, otherwise a normalized absolute path."""
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def build_metadata_dict(
    image_path: Path,
    mask_path: Path,
    overlay_path: Path | None,
    crop_box,
    original_image_shape,
    working_image_shape,
    clicked_seeds_original,
    used_seeds_working,
    snapped_from_working,
    tolerance,
    preprocessing,
    refinement,
    accepted_mask,
):
    """Build the per-frame metadata payload saved beside the mask."""
    patient_id, scan_id, frame_id = parse_patient_scan_frame_from_path(str(image_path))

    if clicked_seeds_original:
        clicked_original = [[int(x), int(y)] for x, y in clicked_seeds_original]
    else:
        clicked_original = []

    used_working = [[int(row), int(col)] for row, col in used_seeds_working] if used_seeds_working else []
    pre_snap_working = [[int(row), int(col)] for row, col in snapped_from_working] if snapped_from_working else []
    snapped = len(pre_snap_working) > 0 and pre_snap_working != used_working

    # We intentionally preserve two coordinate conventions:
    # original-image clicks are stored as [x, y] for visualization on the full frame,
    # while the segmenter-facing working seeds are stored as [row, col] to match NumPy indexing.
    metadata = {
        "patient_id": patient_id,
        "scan_id": scan_id,
        "frame_id": frame_id,
        "image_path": maybe_make_relative_path(image_path),
        "mask_path": maybe_make_relative_path(mask_path),
        "overlay_path": maybe_make_relative_path(overlay_path) if overlay_path is not None else None,
        "image_filename": image_path.name,
        "mask_filename": mask_path.name,
        "num_seeds": len(used_working),
        "clicked_seeds_original": clicked_original,
        "clicked_seeds_original_format": "xy",
        "used_seeds_working": used_working,
        "used_seeds_working_format": "row_col",
        "seed_order": list(range(len(used_working))),
        "prompt_semantics": ["bone_seed"] * len(used_working),
        "seed_source": "manual_click_plus_snap" if snapped else "manual_click",
        "crop_box": crop_box,
        "working_image_shape": list(working_image_shape),
        "original_image_shape": list(original_image_shape),
        "tolerance": int(tolerance) if tolerance is not None else None,
        "notes": "",
        "coordinate_note": (
            "clicked_seeds_original uses [x, y] in the full original image; "
            "used_seeds_working and snapped_from_working use [row, col] in the cropped working image."
        ),
        "preprocessing": preprocessing,
        "refinement": refinement,
        "snapped_from_working": pre_snap_working,
        "snapped_from_working_format": "row_col",
        "accepted_mask": accepted_mask,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "script_version": Path(__file__).name,
    }
    return to_serializable(metadata)


def save_metadata_json(mask_path: str, metadata: dict):
    """Save metadata JSON beside the output mask."""
    mask_path_obj = Path(mask_path)
    meta_path = mask_path_obj.with_name(mask_path_obj.stem.replace("_mask", "") + "_meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return meta_path


def build_default_overlay_path(mask_path: str):
    """Return the default overlay output path beside the saved mask."""
    mask_path_obj = Path(mask_path)
    return mask_path_obj.with_name(mask_path_obj.stem.replace("_mask", "") + "_overlay.png")


def save_overlay_image(base_image: np.ndarray, final_mask: np.ndarray, overlay_path: str, alpha: float = 0.35):
    """Save an RGB overlay showing the final mask boundary on top of the grayscale scan."""
    base_bgr = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    overlay = base_bgr.copy()
    mask_bin = (final_mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, contourIdx=-1, color=(0, 0, 255), thickness=2)
    blended = cv2.addWeighted(overlay, alpha, base_bgr, 1.0 - alpha, 0.0)

    overlay_path_obj = Path(overlay_path)
    overlay_path_obj.parent.mkdir(parents=True, exist_ok=True)
    save_ok = cv2.imwrite(str(overlay_path_obj), blended)
    if not save_ok:
        raise RuntimeError(f"Failed to save overlay image to: {overlay_path_obj}")
    return overlay_path_obj


def smooth_mask_boundary(mask_bin: np.ndarray, sigma: float) -> np.ndarray:
    """
    Light boundary smoothing applied after final mask creation.
    This is intentionally optional and post-hoc so the current segmentation logic stays unchanged.
    """
    if sigma <= 0:
        return mask_bin.astype(np.uint8)

    mask_8bit = (mask_bin.astype(np.uint8) * 255)
    blurred = cv2.GaussianBlur(mask_8bit, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return (blurred >= 127).astype(np.uint8)


def load_metadata_json(meta_path: str):
    """Load a previously saved metadata JSON file."""
    with Path(meta_path).open("r", encoding="utf-8") as meta_file:
        return json.load(meta_file)


def load_reusable_seeds(meta_path: str, expected_seed_count: int):
    """
    Load reusable seeds from a previous metadata JSON file.
    Returns (clicked_seeds_original, used_seeds_working, seed_source).
    """
    metadata = load_metadata_json(meta_path)
    saved_num_seeds = metadata.get("num_seeds")
    used_seeds = metadata.get("used_seeds_working") or []
    clicked_original = metadata.get("clicked_seeds_original") or []

    if saved_num_seeds is None:
        raise ValueError(f"Metadata JSON is missing num_seeds: {meta_path}")
    if int(saved_num_seeds) != int(expected_seed_count):
        raise ValueError(
            f"Saved seed count ({saved_num_seeds}) does not match current max_seeds ({expected_seed_count})."
        )
    if len(used_seeds) != int(expected_seed_count):
        raise ValueError(
            f"Metadata JSON has {len(used_seeds)} working seeds, expected {expected_seed_count}: {meta_path}"
        )

    clicked_original_xy = []
    for point in clicked_original:
        if not isinstance(point, list) or len(point) != 2:
            continue
        clicked_original_xy.append((int(point[0]), int(point[1])))

    used_working = []
    for point in used_seeds:
        if not isinstance(point, list) or len(point) != 2:
            continue
        used_working.append((int(point[0]), int(point[1])))

    if len(used_working) != int(expected_seed_count):
        raise ValueError(f"Could not recover the expected number of saved seeds from: {meta_path}")

    return clicked_original_xy, used_working, "reused_saved_working_seeds"


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




def active_contour_refinement_from_mask(img, mask, sigma, alpha, beta, gamma, w_line, w_edge):
    """
    Extract all contours from the mask and refine each one with an
    edge-based active contour (snake).
    Returns (list_of_initial_contours, list_of_refined_snakes).
    """
    smoothed = gaussian(img, sigma=sigma)

    # Find all contours in the binary mask
    contours = find_contours(mask.astype(float), level=0.5)
    if not contours:
        raise ValueError("No contour found in the mask.")

    edges = sobel(smoothed)

    init_list = []
    snake_list = []

    for init in contours:
        # Optionally skip extremely short contours
        if len(init) < 5:
            continue

        snake = active_contour(
            edges,
            init,
            alpha=alpha,  # internal tension (lower -> less shrinking)
            beta=beta,    # smoothness (lower -> can bend more)
            gamma=gamma,  # time step
            w_line=w_line,
            w_edge=w_edge,
        )

        init_list.append(init)
        snake_list.append(snake)

    if not snake_list:
        raise ValueError("No valid contours for active contour refinement.")

    return init_list, snake_list



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
    parser.add_argument("--crop_y_min", type=int, default=CROP_Y_MIN, help="Top crop boundary in original-image pixels.")
    parser.add_argument("--crop_y_max", type=int, default=CROP_Y_MAX, help="Bottom crop boundary in original-image pixels.")
    parser.add_argument("--crop_x_min", type=int, default=CROP_X_MIN, help="Left crop boundary in original-image pixels.")
    parser.add_argument("--crop_x_max", type=int, default=CROP_X_MAX, help="Right crop boundary in original-image pixels.")
    parser.add_argument("--clahe_clip_limit", type=float, default=CLAHE_CLIP_LIMIT, help="CLAHE clip limit used in preprocessing.")
    parser.add_argument("--gaussian_kernel_y", type=int, default=GAUSSIAN_KERNEL[0], help="Gaussian blur kernel height.")
    parser.add_argument("--gaussian_kernel_x", type=int, default=GAUSSIAN_KERNEL[1], help="Gaussian blur kernel width.")
    parser.add_argument("--snake_sigma", type=float, default=ACTIVE_CONTOUR_SIGMA, help="Gaussian sigma before snake edge extraction.")
    parser.add_argument("--snake_alpha", type=float, default=ACTIVE_CONTOUR_ALPHA, help="Active contour alpha parameter.")
    parser.add_argument("--snake_beta", type=float, default=ACTIVE_CONTOUR_BETA, help="Active contour beta parameter.")
    parser.add_argument("--snake_gamma", type=float, default=ACTIVE_CONTOUR_GAMMA, help="Active contour gamma parameter.")
    parser.add_argument("--snake_w_line", type=float, default=ACTIVE_CONTOUR_W_LINE, help="Active contour line attraction weight.")
    parser.add_argument("--snake_w_edge", type=float, default=ACTIVE_CONTOUR_W_EDGE, help="Active contour edge attraction weight.")
    parser.add_argument(
        "--final_mask_mode",
        type=str,
        default="union",
        choices=["union", "snake_only"],
        help="How to build the final mask: union keeps region-growing plus snake, snake_only keeps only the snake-based mask.",
    )
    parser.add_argument(
        "--final_open_r",
        type=int,
        default=0,
        help="Optional opening radius applied to the final mask for light bump smoothing. Default keeps the current behavior.",
    )
    parser.add_argument(
        "--boundary_smooth",
        action="store_true",
        help="Optionally smooth the final mask boundary after mask construction.",
    )
    parser.add_argument(
        "--boundary_smooth_sigma",
        type=float,
        default=1.0,
        help="Gaussian sigma used for optional final boundary smoothing.",
    )
    parser.add_argument(
        "--reuse_meta_path",
        type=str,
        default="",
        help="Optional metadata JSON path from a previous run to reuse the saved seed set.",
    )
    parser.add_argument(
        "--output_overlay_path",
        type=str,
        default="",
        help="Optional path to save a scan-plus-mask overlay PNG. Defaults beside the mask.",
    )

    args = parser.parse_args()
    crop_box = get_crop_box(args)
    gaussian_kernel = (args.gaussian_kernel_y, args.gaussian_kernel_x)

    if args.crop_x_min >= args.crop_x_max or args.crop_y_min >= args.crop_y_max:
        raise ValueError("Crop bounds must satisfy x_min < x_max and y_min < y_max.")
    if args.gaussian_kernel_x <= 0 or args.gaussian_kernel_y <= 0:
        raise ValueError("Gaussian kernel sizes must be positive integers.")
    if args.gaussian_kernel_x % 2 == 0 or args.gaussian_kernel_y % 2 == 0:
        raise ValueError("Gaussian kernel sizes must be odd integers.")
    
    # 1) Load and crop
    image_path = Path(args.image_path).resolve()
    output_mask_path = Path(args.output_mask_path).resolve()
    output_overlay_path = (
        Path(args.output_overlay_path).resolve()
        if args.output_overlay_path
        else build_default_overlay_path(str(output_mask_path)).resolve()
    )
    img_full = load_image(str(image_path))
    img = crop_ultrasound_region(img_full, crop_box)

    # 2) Preprocess
    pre = preprocess_image(img, args.clahe_clip_limit, gaussian_kernel)

    if args.show:
        plt.figure()
        plt.imshow(pre, cmap="gray")
        plt.title("Preprocessed (cropped) image")
        plt.axis("off")
        plt.show()

    # 3) Seed acquisition
    if args.reuse_meta_path:
        clicked_seeds_original, seeds, seed_source = load_reusable_seeds(
            args.reuse_meta_path,
            expected_seed_count=args.max_seeds,
        )
        clicked_seeds_working = convert_seed_coords_to_working(clicked_seeds_original, crop_box)
        snapped_from_working = [[int(row), int(col)] for row, col in seeds]
        print(f"Reusing saved seeds from: {args.reuse_meta_path}")
        print("Used seeds:", seeds)
    else:
        clicked_seeds_working = get_multiple_seeds_from_click(pre, max_seeds=args.max_seeds)
        if not clicked_seeds_working:
            raise RuntimeError("No seeds selected. Aborting segmentation.")
        clicked_seeds_original = convert_seed_coords_to_original(clicked_seeds_working, crop_box)

        # NEW: snap seeds to local bright ridge
        snapped_from_working = [(int(row), int(col)) for row, col in clicked_seeds_working]
        seeds = snap_seeds_to_bright_line(pre, clicked_seeds_working, window=args.snap_window)
        seed_source = "manual_click_plus_snap" if seeds != snapped_from_working else "manual_click"
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
    # 5) (Optional) mask cleaning.
    # To re-enable cleaning, uncomment the clean_mask line below and remove the direct cast.
    # mask_clean = clean_mask(combined_mask, seeds, y_band=args.y_band,
    #                         min_area=args.min_area, open_r=args.open_r, close_w=args.close_w)
    mask_clean = (combined_mask.astype(np.uint8) * 255)
    if args.show:
        plt.figure()
        plt.imshow(mask_clean, cmap="gray")
        plt.title("Region growing – cleaned mask")
        plt.axis("off")
        plt.show()

    # 6) Active contour refinement (for all contours in the mask)
    init_list, snake_list = active_contour_refinement_from_mask(
        pre,
        mask_clean,
        sigma=args.snake_sigma,
        alpha=args.snake_alpha,
        beta=args.snake_beta,
        gamma=args.snake_gamma,
        w_line=args.snake_w_line,
        w_edge=args.snake_w_edge,
    )

    if args.show:
        plt.figure()
        plt.imshow(pre, cmap="gray")
        for init, snake in zip(init_list, snake_list):
            plt.plot(init[:, 1], init[:, 0], "--r", alpha=0.6)
            plt.plot(snake[:, 1], snake[:, 0], "-b", alpha=0.8)
        plt.title("Active contour refinement (all bone fragments)")
        plt.axis("off")
        plt.show()

   # 7) Build final mask:
    #    take snake regions for all fragments, thicken them a bit,
    #    and combine them with the cleaned region-growing mask.

    # Mask from all snake contours (0/255)
    snake_mask = np.zeros_like(pre, dtype=np.uint8)
    for snake in snake_list:
        snake_mask |= create_mask_from_contour(pre.shape, snake)

    # Convert both to {0,1}
    clean_bin = (mask_clean > 0).astype(np.uint8)
    snake_bin = (snake_mask > 0).astype(np.uint8)

    # Slightly thicken the snake region so it covers seeds & realistic bone thickness
    # radius=3 is a good starting point; you can tune to 2–4
    if args.snake_dilate > 0:
        snake_thick = dilation(snake_bin, disk(args.snake_dilate)).astype(np.uint8)
    else:
        snake_thick = snake_bin.copy()

    if args.final_mask_mode == "snake_only":
        final_bin = snake_thick.astype(np.uint8)
    else:
        # Keep all cleaned region-growing regions and reinforce with the snake
        final_bin = ((snake_thick | clean_bin).astype(np.uint8))

    if args.final_open_r > 0:
        final_bin = opening(final_bin.astype(bool), disk(args.final_open_r)).astype(np.uint8)
    if args.boundary_smooth:
        final_bin = smooth_mask_boundary(final_bin, args.boundary_smooth_sigma)

    # Back to 0/255 for saving
    final_mask = final_bin * 255

    # Ensure output directory exists
    out_dir = os.path.dirname(str(output_mask_path))
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)

    save_ok = cv2.imwrite(str(output_mask_path), final_mask)
    if not save_ok:
        raise RuntimeError(f"Failed to save segmentation mask to: {output_mask_path}")
    saved_overlay_path = save_overlay_image(img, final_mask, str(output_overlay_path))

    preprocessing = {
        "crop_applied": True,
        "clahe_used": True,
        "clahe_clip_limit": args.clahe_clip_limit,
        "gaussian_blur_used": True,
        "gaussian_kernel": list(gaussian_kernel),
        "resize_factor": None,
    }
    refinement = {
        "region_growing_used": True,
        "mask_cleaning_used": False,
        "mask_cleaning_currently_bypassed": True,
        "clean_y_band": args.y_band,
        "clean_min_area": args.min_area,
        "open_disk_radius": args.open_r,
        "close_rect_width": args.close_w,
        "snap_used": True,
        "snap_window": args.snap_window,
        "snake_used": True,
        "snake_sigma": args.snake_sigma,
        "snake_alpha": args.snake_alpha,
        "snake_beta": args.snake_beta,
        "snake_gamma": args.snake_gamma,
        "snake_w_line": args.snake_w_line,
        "snake_w_edge": args.snake_w_edge,
        "snake_dilate_radius": args.snake_dilate,
        "final_mask_mode": args.final_mask_mode,
        "final_open_radius": args.final_open_r,
        "boundary_smooth": args.boundary_smooth,
        "boundary_smooth_sigma": args.boundary_smooth_sigma,
    }
    metadata = build_metadata_dict(
        image_path=image_path,
        mask_path=output_mask_path,
        overlay_path=saved_overlay_path,
        crop_box=crop_box,
        original_image_shape=img_full.shape[:2],
        working_image_shape=pre.shape[:2],
        clicked_seeds_original=clicked_seeds_original,
        used_seeds_working=seeds,
        snapped_from_working=snapped_from_working,
        tolerance=args.tolerance,
        preprocessing=preprocessing,
        refinement=refinement,
        accepted_mask=True,
    )
    metadata["seed_source"] = seed_source
    meta_path = save_metadata_json(str(output_mask_path), metadata)

    print(f"Saved segmentation mask to: {output_mask_path}")
    print(f"Saved overlay image to: {saved_overlay_path}")
    print(f"Saved metadata JSON to: {meta_path}")


if __name__ == "__main__":
    main()
