"""
compare_preprocessing.py — Preprocessing comparison driver.

For every labeled (image, GT mask) pair in metadata_labeled.csv, re-runs the
full segmentation pipeline (crop → preprocess → region growing → mask cleaning
→ active contour → final mask) under each preprocessing method, then scores
the result against the ground-truth mask.

Metrics per frame/method
------------------------
  Dice       2·TP / (2·TP + FP + FN)
  IoU        TP / (TP + FP + FN)
  Precision  TP / (TP + FP)
  Recall     TP / (TP + FN)
  Hausdorff  max directed Hausdorff distance between nonzero pixel sets (px)

Usage
-----
  python compare_preprocessing.py                                  # 3 frames (default)
  python compare_preprocessing.py --n_frames 0                     # all 44 frames
  python compare_preprocessing.py --methods baseline fft_bandpass wavelet_bayes_soft
  python compare_preprocessing.py --n_frames 5 --output_dir quick_test/
  python compare_preprocessing.py --skip_snake                      # region-growing only (fast)

Outputs (in --output_dir)
-------------------------
  per_frame_metrics.csv           one row per (patient, scan, frame, method)
  summary_metrics.csv             mean ± std per method, sorted by Dice descending
  figures/<scan>_f<N>/<method>.png  4-panel comparison image per (frame, method)
"""

import argparse
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for script use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.metrics import hausdorff_distance
from skimage.morphology import dilation, disk

# ── Path setup ───────────────────────────────────────────────────────────────
_HERE      = Path(__file__).resolve().parent                    # Preprocessing/
_BONE_SEG  = _HERE.parent                                       # Bone Segmentation/
_REPO_ROOT = _BONE_SEG.parent                                   # repo root
_SEG_DIR   = _BONE_SEG / "Region Growing Segmentation" / "seg"

for _p in (_HERE, str(_SEG_DIR)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ultrasound_bone_segmentation_cli import (
    crop_ultrasound_region,
    region_growing,
    clean_mask,
    active_contour_refinement_from_mask,
    create_mask_from_contour,
    trim_mask_above_seeds,
    trim_mask_below_seeds,
    CLEAN_Y_BAND, CLEAN_MIN_AREA, OPEN_DISK_RADIUS, CLOSE_RECT_WIDTH,
    ACTIVE_CONTOUR_SIGMA, ACTIVE_CONTOUR_ALPHA, ACTIVE_CONTOUR_BETA,
    ACTIVE_CONTOUR_GAMMA, ACTIVE_CONTOUR_W_LINE, ACTIVE_CONTOUR_W_EDGE,
    SNAKE_DILATE_RADIUS,
)
from preprocessing_api import preprocess, list_methods

# Suppress active_contour convergence warnings — they are cosmetic
warnings.filterwarnings("ignore", category=UserWarning)

# ── Default paths ────────────────────────────────────────────────────────────
DEFAULT_CSV        = _REPO_ROOT / "Dataset" / "metadata_labeled.csv"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "compare_preprocessing_results"


# ── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(pred_mask: np.ndarray,
                    gt_mask:   np.ndarray) -> dict:
    """
    Compare predicted and ground-truth binary masks (uint8, 0/255).
    Returns a dict with dice, iou, precision, recall, hausdorff_px.
    """
    pred = (pred_mask > 0).astype(bool)
    gt   = (gt_mask   > 0).astype(bool)

    tp = int((pred &  gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())

    eps   = 1e-8
    dice  = 2 * tp / (2 * tp + fp + fn + eps)
    iou   = tp / (tp + fp + fn + eps)
    prec  = tp / (tp + fp + eps)
    rec   = tp / (tp + fn + eps)

    # Hausdorff on the nonzero pixel sets (thin bone masks ≈ boundary)
    if pred.any() and gt.any():
        haus = float(hausdorff_distance(pred, gt))
    else:
        haus = float("nan")

    return {
        "dice":         round(dice, 5),
        "iou":          round(iou,  5),
        "precision":    round(prec, 5),
        "recall":       round(rec,  5),
        "hausdorff_px": round(haus, 2) if not np.isnan(haus) else float("nan"),
    }


# ── Segmentation pipeline (non-interactive) ──────────────────────────────────

def _seeds_from_meta(meta: dict) -> list:
    """Return a list of (row, col) seed tuples from a meta JSON dict."""
    raw = meta.get("used_seeds_working") or []
    return [(int(s[0]), int(s[1])) for s in raw if len(s) == 2]


def run_segmentation(pre: np.ndarray,
                     seeds: list,
                     meta: dict,
                     skip_snake: bool = False) -> np.ndarray:
    """
    Run the segmentation pipeline on an already-preprocessed cropped image,
    faithfully replaying every parameter recorded in the frame's meta JSON.

    Parameters
    ----------
    pre        : preprocessed cropped uint8 image (600×600)
    seeds      : list of (row, col) working-coordinate seed points
    meta       : loaded meta JSON dict — provides all segmentation parameters
    skip_snake : if True, stop after mask cleaning (no active contour)

    Returns
    -------
    Binary mask, uint8 (0/255), same shape as pre.
    """
    tol = int(meta.get("tolerance", 20))
    ref = meta.get("refinement", {})

    # Cleaning
    y_band   = int(ref.get("clean_y_band",   CLEAN_Y_BAND))
    min_area = int(ref.get("clean_min_area", CLEAN_MIN_AREA))
    open_r   = int(ref.get("open_disk_radius", OPEN_DISK_RADIUS))
    close_w  = int(ref.get("close_rect_width", CLOSE_RECT_WIDTH))

    # Per-seed flood-fill band restrictions
    seed_x_band    = int(ref.get("seed_x_band",    0))
    seed_y_band    = int(ref.get("seed_y_band",    0))
    seed_y_band_up = int(ref.get("seed_y_band_up", 0))

    # Snake
    sigma    = float(ref.get("snake_sigma",  ACTIVE_CONTOUR_SIGMA))
    alpha    = float(ref.get("snake_alpha",  ACTIVE_CONTOUR_ALPHA))
    beta     = float(ref.get("snake_beta",   ACTIVE_CONTOUR_BETA))
    gamma    = float(ref.get("snake_gamma",  ACTIVE_CONTOUR_GAMMA))
    w_line   = float(ref.get("snake_w_line", ACTIVE_CONTOUR_W_LINE))
    w_edge   = float(ref.get("snake_w_edge", ACTIVE_CONTOUR_W_EDGE))
    dilate_r = int(ref.get("snake_dilate_radius", SNAKE_DILATE_RADIUS))

    # Post-processing
    pre_snake_dilate = int(ref.get("pre_snake_dilate", 0))
    post_trim_up     = int(ref.get("post_trim_up",     0))
    post_trim_down   = int(ref.get("post_trim_down",   0))
    final_mask_mode  = ref.get("final_mask_mode", "union")

    # Region growing with per-seed band restrictions
    combined = np.zeros(pre.shape[:2], dtype=bool)
    for seed in seeds:
        mask = region_growing(pre, seed, tol)
        row, col = seed[0], seed[1]
        if seed_x_band > 0:
            col_mask = np.zeros_like(mask)
            col_mask[:, max(0, col - seed_x_band):col + seed_x_band + 1] = True
            mask = mask & col_mask
        if seed_y_band > 0 or seed_y_band_up > 0:
            up   = seed_y_band_up if seed_y_band_up > 0 else seed_y_band
            down = seed_y_band    if seed_y_band    > 0 else seed_y_band_up
            row_mask = np.zeros_like(mask)
            row_mask[max(0, row - up):row + down + 1, :] = True
            mask = mask & row_mask
        combined |= mask

    # Mask cleaning — skip if it was bypassed when the GT mask was created
    cleaning_bypassed = ref.get("mask_cleaning_currently_bypassed", False)
    if cleaning_bypassed:
        mask_clean = (combined > 0).astype(np.uint8) * 255
    else:
        mask_clean = clean_mask(combined, seeds,
                                y_band=y_band, min_area=min_area,
                                open_r=open_r, close_w=close_w)

    if skip_snake:
        return (mask_clean > 0).astype(np.uint8) * 255

    # Optional pre-snake dilation
    if pre_snake_dilate > 0:
        mask_clean = dilation(mask_clean > 0, disk(pre_snake_dilate)).astype(np.uint8) * 255

    # Active contour refinement
    try:
        _, snake_list = active_contour_refinement_from_mask(
            pre, mask_clean,
            sigma=sigma, alpha=alpha, beta=beta, gamma=gamma,
            w_line=w_line, w_edge=w_edge,
        )
    except ValueError:
        return (mask_clean > 0).astype(np.uint8) * 255

    snake_mask = np.zeros(pre.shape[:2], dtype=np.uint8)
    for snake in snake_list:
        snake_mask |= create_mask_from_contour(pre.shape[:2], snake)

    clean_bin = (mask_clean > 0).astype(np.uint8)
    snake_bin = (snake_mask > 0).astype(np.uint8)
    if dilate_r > 0:
        snake_bin = dilation(snake_bin, disk(dilate_r)).astype(np.uint8)

    if final_mask_mode == "snake_only":
        final_bin = snake_bin
    else:
        final_bin = (snake_bin | clean_bin).astype(np.uint8)

    if post_trim_up > 0:
        final_bin = trim_mask_above_seeds(final_bin, seeds, post_trim_up)
    if post_trim_down > 0:
        final_bin = trim_mask_below_seeds(final_bin, seeds, post_trim_down)

    return final_bin.astype(np.uint8) * 255


# ── Frame data loader ────────────────────────────────────────────────────────

def load_frame(row: dict, repo_root: Path):
    """
    Load image, GT mask, and meta JSON for one CSV row.
    Returns (img_cropped_raw, gt_mask, meta, crop_box) or raises.
    """
    img_path  = repo_root / row["image_path"]
    mask_path = repo_root / row["mask_path"]
    meta_path = repo_root / row["meta_path"]

    img_full = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_full is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    gt_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if gt_mask is None:
        raise FileNotFoundError(f"GT mask not found: {mask_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta JSON not found: {meta_path}")
    with meta_path.open(encoding="utf-8") as fh:
        meta = json.load(fh)

    crop_box = meta.get("crop_box")
    if crop_box is None:
        raise ValueError(f"crop_box missing in meta JSON: {meta_path}")

    img_cropped = crop_ultrasound_region(img_full, crop_box)
    return img_cropped, gt_mask, meta


# ── Summary table ────────────────────────────────────────────────────────────

def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-frame metrics into mean ± std per method.
    Sorted by mean Dice descending.
    """
    metric_cols = ["dice", "iou", "precision", "recall", "hausdorff_px"]
    rows = []
    for method, grp in df.groupby("method", sort=False):
        row = {"method": method, "n_frames": int(grp["status"].eq("ok").sum())}
        for col in metric_cols:
            valid = grp.loc[grp["status"] == "ok", col].dropna()
            row[f"{col}_mean"] = round(valid.mean(), 5) if len(valid) else float("nan")
            row[f"{col}_std"]  = round(valid.std(),  5) if len(valid) else float("nan")
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values("dice_mean", ascending=False).reset_index(drop=True)


# ── Figure saving ────────────────────────────────────────────────────────────

def _draw_mask_contours(base_bgr: np.ndarray,
                        mask: np.ndarray,
                        color: tuple,
                        thickness: int = 2) -> np.ndarray:
    """Draw filled mask contours on a BGR image copy."""
    out = base_bgr.copy()
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(out, contours, -1, color, thickness)
    return out


def save_comparison_figure(img_cropped:  np.ndarray,
                           pre:          np.ndarray,
                           pred_mask:    np.ndarray,
                           gt_mask:      np.ndarray,
                           method:       str,
                           metrics:      dict,
                           save_path:    Path) -> None:
    """
    Save a 4-panel comparison figure for one (frame, method) pair.

    Panels (left to right):
      1. Original cropped image (raw, before any preprocessing)
      2. Preprocessed image (output of the method under test)
      3. Ground-truth mask contour (green) on original
      4. Predicted mask contour (red) + GT contour (green dashed) on original,
         with Dice / IoU / Hausdorff annotated in the title
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    base_bgr = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR)

    gt_overlay   = _draw_mask_contours(base_bgr, gt_mask,   color=(0, 200, 0))
    pred_overlay = _draw_mask_contours(base_bgr, pred_mask, color=(0, 0, 220))
    pred_overlay = _draw_mask_contours(pred_overlay, gt_mask, color=(0, 200, 0), thickness=1)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    axes[0].imshow(img_cropped, cmap="gray")
    axes[0].set_title("Original (cropped)", fontsize=9)

    axes[1].imshow(pre, cmap="gray")
    axes[1].set_title(f"Preprocessed — {method}", fontsize=9)

    axes[2].imshow(cv2.cvtColor(gt_overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title("Ground-truth mask (green)", fontsize=9)

    dice = metrics.get("dice", float("nan"))
    iou  = metrics.get("iou",  float("nan"))
    haus = metrics.get("hausdorff_px", float("nan"))
    axes[3].imshow(cv2.cvtColor(pred_overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title(
        f"Prediction (red) vs GT (green)\n"
        f"Dice={dice:.3f}  IoU={iou:.3f}  Haus={haus:.1f}px",
        fontsize=9,
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare preprocessing methods on labeled Patient 1 frames."
    )
    parser.add_argument(
        "--metadata_csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to metadata_labeled.csv (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write results (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to run. Default: all. "
             f"Available: {list_methods()}",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=3,
        help="Number of frames to process. Default: 3. Pass 0 to run all frames.",
    )
    parser.add_argument(
        "--skip_snake",
        action="store_true",
        help="Skip active contour — evaluate region-growing quality only (much faster).",
    )
    parser.add_argument(
        "--no_figures",
        action="store_true",
        help="Skip saving comparison figures (faster, metrics-only run).",
    )
    parser.add_argument(
        "--repo_root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root directory (auto-detected from script location).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate and resolve methods
    all_methods = list_methods()
    if args.methods:
        invalid = [m for m in args.methods if m not in all_methods]
        if invalid:
            raise ValueError(f"Unknown methods: {invalid}. Valid: {all_methods}")
        methods = args.methods
    else:
        methods = all_methods

    # Load CSV
    csv_path = args.metadata_csv.resolve()
    if not csv_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {csv_path}")
    df_csv = pd.read_csv(csv_path)
    rows = df_csv.to_dict("records")
    if args.n_frames and args.n_frames > 0:
        rows = rows[: args.n_frames]

    repo_root = args.repo_root.resolve()
    out_dir   = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Frames to process : {len(rows)}")
    print(f"Methods           : {methods}")
    print(f"Skip snake        : {args.skip_snake}")
    print(f"Save figures      : {not args.no_figures}")
    print(f"Output directory  : {out_dir}")
    print()

    results = []
    total   = len(rows) * len(methods)
    done    = 0
    t_start = time.time()

    for row in rows:
        frame_label = (
            f"{row.get('patient_id','?')} / "
            f"{row.get('scan_id','?')} / "
            f"f{row.get('frame_id','?')}"
        )

        try:
            img_cropped, gt_mask, meta = load_frame(row, repo_root)
        except Exception as exc:
            print(f"  [SKIP] {frame_label} — could not load: {exc}")
            for method in methods:
                results.append({
                    "patient_id": row.get("patient_id"),
                    "scan_id":    row.get("scan_id"),
                    "frame_id":   row.get("frame_id"),
                    "method":     method,
                    "status":     "load_error",
                    "error":      str(exc),
                    "dice": None, "iou": None,
                    "precision": None, "recall": None,
                    "hausdorff_px": None,
                    "elapsed_s": None,
                })
                done += 1
            continue

        seeds = _seeds_from_meta(meta)
        if not seeds:
            print(f"  [SKIP] {frame_label} — no seeds in meta JSON")
            for method in methods:
                results.append({
                    "patient_id": row.get("patient_id"),
                    "scan_id":    row.get("scan_id"),
                    "frame_id":   row.get("frame_id"),
                    "method":     method,
                    "status":     "no_seeds",
                    "error":      "used_seeds_working is empty",
                    "dice": None, "iou": None,
                    "precision": None, "recall": None,
                    "hausdorff_px": None,
                    "elapsed_s": None,
                })
                done += len(methods)
            continue

        for method in methods:
            t0 = time.time()
            result_row = {
                "patient_id": row.get("patient_id"),
                "scan_id":    row.get("scan_id"),
                "frame_id":   row.get("frame_id"),
                "method":     method,
            }
            try:
                pre = preprocess(img_cropped, method)
                pred_mask = run_segmentation(pre, seeds, meta,
                                             skip_snake=args.skip_snake)
                metrics = compute_metrics(pred_mask, gt_mask)
                result_row.update(metrics)
                result_row["status"]    = "ok"
                result_row["error"]     = ""
                result_row["elapsed_s"] = round(time.time() - t0, 2)

                if not args.no_figures:
                    scan_tag  = str(row.get("scan_id", "scan")).replace("/", "_")
                    frame_tag = f"f{row.get('frame_id', 0)}"
                    fig_path  = (out_dir / "figures"
                                 / f"{scan_tag}_{frame_tag}"
                                 / f"{method}.png")
                    save_comparison_figure(
                        img_cropped, pre, pred_mask, gt_mask,
                        method, metrics, fig_path,
                    )

            except Exception as exc:
                result_row.update({
                    "status":     "error",
                    "error":      traceback.format_exc(limit=3),
                    "dice": None, "iou": None,
                    "precision": None, "recall": None,
                    "hausdorff_px": None,
                    "elapsed_s": round(time.time() - t0, 2),
                })

            results.append(result_row)
            done += 1

            status_str = (
                f"dice={result_row.get('dice', 'err'):.4f}"
                if result_row["status"] == "ok"
                else result_row["status"]
            )
            elapsed_total = time.time() - t_start
            rate = done / elapsed_total if elapsed_total > 0 else 0
            eta  = (total - done) / rate if rate > 0 else 0
            print(
                f"  [{done:3d}/{total}] {frame_label:50s} "
                f"{method:26s}  {status_str}  "
                f"(ETA {eta/60:.1f} min)"
            )

    # Save per-frame results
    df_results = pd.DataFrame(results)
    per_frame_path = out_dir / "per_frame_metrics.csv"
    df_results.to_csv(per_frame_path, index=False)
    print(f"\nPer-frame metrics -> {per_frame_path}")

    # Save summary
    summary = build_summary(df_results)
    summary_path = out_dir / "summary_metrics.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary metrics  -> {summary_path}")

    # Print summary table
    print()
    print(summary.to_string(index=False))

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
