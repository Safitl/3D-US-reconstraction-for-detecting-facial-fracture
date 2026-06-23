"""
generate_pipeline_figures.py — Classical segmentation pipeline stage figures.

Outputs (all 200 dpi, saved to report_figures/):
  fig_442_seed_snapping.png
  fig_443_region_growing.png
  fig_444_morphological_cleaning.png
  fig_445_active_contour.png
  fig_446_mask_output.png
  fig_44_gallery.png
"""

import json
import sys
import warnings
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import find_contours

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
_SEG  = _REPO / "Bone Segmentation" / "Region Growing Segmentation" / "seg"

if str(_SEG) not in sys.path:
    sys.path.insert(0, str(_SEG))

from ultrasound_bone_segmentation_cli import (
    preprocess_image,
    region_growing,
    clean_mask,
    CLEAN_Y_BAND, CLEAN_MIN_AREA, OPEN_DISK_RADIUS, CLOSE_RECT_WIDTH,
)

# ── Config ─────────────────────────────────────────────────────────────────────
CROP_Y_MIN, CROP_Y_MAX = 100, 700
CROP_X_MIN, CROP_X_MAX = 200, 800

FRAMES_DIR = _REPO / "Dataset" / "Patient1" / "IMG_frames"
MASKS_DIR  = _REPO / "Dataset" / "Patient1" / "Masks"
OUT_DIR    = _REPO / "report_figures"

MAIN_FRAME = "image_406314327901_f032"
GALLERY_FRAMES = [
    ("image_383229031802_f000", "image_383229031802\nframe 0"),
    ("image_406314327901_f041", "image_406314327901\nframe 41"),
    ("image_258976846007",      "image_258976846007\n(static)"),
]

DPI = 200
FS  = 14

# ── Gallery: parameters that DIFFER across frames (identical ones are excluded)
# Excluded (same across all 3): snap_window=7, snake_sigma=1.0, snake_gamma=0.01,
#   snake_w_line=0.0, snake_w_edge=1.0, final_open_r=0, y_band=35,
#   min_area=200, open_r=1, seed_x_band=0
GALLERY_PARAMS = [
    # (display label,         value extractor)
    ("num_seeds",          lambda m, r: m.get("num_seeds", "?")),
    ("tolerance",          lambda m, r: m.get("tolerance", "?")),
    ("snake_alpha",        lambda m, r: r.get("snake_alpha", "?")),
    ("snake_beta",         lambda m, r: r.get("snake_beta", "?")),
    ("final_mask_mode",    lambda m, r: r.get("final_mask_mode", "?")),
    ("boundary_smooth",    lambda m, r: r.get("boundary_smooth", False)),
    ("boundary_smooth_σ",  lambda m, r: r.get("boundary_smooth_sigma", 1.0)),
    ("close_w",            lambda m, r: r.get("close_rect_width", CLOSE_RECT_WIDTH)),
    ("seed_y_band",        lambda m, r: r.get("seed_y_band", 0)),
    ("seed_y_band_up",     lambda m, r: r.get("seed_y_band_up", 0)),
    ("pre_snake_dilate",   lambda m, r: r.get("pre_snake_dilate", 0)),
    ("post_trim_up",       lambda m, r: r.get("post_trim_up", 0)),
    ("post_trim_down",     lambda m, r: r.get("post_trim_down", 0)),
]


# ── Low-level helpers ──────────────────────────────────────────────────────────

def load_meta(name: str) -> dict:
    with open(MASKS_DIR / f"{name}_meta.json") as fh:
        return json.load(fh)


def load_frame_and_preprocess(name: str):
    """Returns (img_full_gray u8, img_crop u8, img_pre u8)."""
    img_full = cv2.imread(str(FRAMES_DIR / f"{name}.png"), cv2.IMREAD_GRAYSCALE)
    if img_full is None:
        raise FileNotFoundError(FRAMES_DIR / f"{name}.png")
    img_crop = img_full[CROP_Y_MIN:CROP_Y_MAX, CROP_X_MIN:CROP_X_MAX]
    img_pre  = preprocess_image(img_crop, clahe_clip_limit=0.01,
                                gaussian_kernel=(7, 7))
    return img_full, img_crop, img_pre


def load_saved_mask(name: str) -> np.ndarray:
    return cv2.imread(str(MASKS_DIR / f"{name}_mask.png"), cv2.IMREAD_GRAYSCALE)


def color_overlay(img_u8: np.ndarray, mask_bool: np.ndarray,
                  color=(0.18, 0.42, 1.0), alpha=0.45) -> np.ndarray:
    rgb = np.stack([img_u8] * 3, axis=-1).astype(np.float32) / 255.0
    out = rgb.copy()
    c   = np.array(color, dtype=np.float32)
    out[mask_bool] = rgb[mask_bool] * (1 - alpha) + c * alpha
    return out


def run_region_growing(img_pre: np.ndarray, seeds: list, meta: dict) -> np.ndarray:
    ref = meta.get("refinement", {})
    tol = int(meta.get("tolerance", 20))
    xb  = int(ref.get("seed_x_band",    0))
    yb  = int(ref.get("seed_y_band",    0))
    ybu = int(ref.get("seed_y_band_up", 0))

    combined = np.zeros(img_pre.shape[:2], dtype=bool)
    for (row, col) in seeds:
        m = region_growing(img_pre, (row, col), tol)
        if xb:
            cm = np.zeros_like(m)
            cm[:, max(0, col - xb):col + xb + 1] = True
            m &= cm
        if yb or ybu:
            up = ybu if ybu else yb
            dn = yb  if yb  else ybu
            rm = np.zeros_like(m)
            rm[max(0, row - up):row + dn + 1, :] = True
            m &= rm
        combined |= m
    return combined


def run_cleaning(rg_bool: np.ndarray, seeds: list, meta: dict) -> np.ndarray:
    ref = meta.get("refinement", {})
    if ref.get("mask_cleaning_currently_bypassed", False):
        return rg_bool.astype(np.uint8) * 255
    return clean_mask(
        rg_bool, seeds,
        y_band   = int(ref.get("clean_y_band",     CLEAN_Y_BAND)),
        min_area = int(ref.get("clean_min_area",   CLEAN_MIN_AREA)),
        open_r   = int(ref.get("open_disk_radius", OPEN_DISK_RADIUS)),
        close_w  = int(ref.get("close_rect_width", CLOSE_RECT_WIDTH)),
    )


# ── Plotting helper ────────────────────────────────────────────────────────────

def show_panel(ax, img, title,
               seeds=None, seed_color="#FF4444", seed_size=120,
               contours=None, contour_color="#FFD700", contour_lw=1.8):
    if img.ndim == 2:
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
    else:
        ax.imshow(np.clip(img, 0, 1))

    if seeds:
        rows, cols = zip(*seeds)
        ax.scatter(cols, rows, s=seed_size, c=seed_color, marker="+",
                   linewidths=2.0, zorder=5)

    if contours:
        for c in contours:
            ax.plot(c[:, 1], c[:, 0], color=contour_color, lw=contour_lw)

    ax.set_title(title, fontsize=FS, fontweight="bold", pad=9)
    ax.axis("off")


def save_fig(fig, filename):
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / filename
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {filename}")


# ══════════════════════════════════════════════════════════════════════════════
# Load main frame & run pipeline stages
# ══════════════════════════════════════════════════════════════════════════════

print(f"Loading {MAIN_FRAME} ...")
meta_m = load_meta(MAIN_FRAME)
img_full, img_crop, img_pre = load_frame_and_preprocess(MAIN_FRAME)
seeds_raw  = [tuple(s) for s in meta_m["snapped_from_working"]]
seeds_snap = [tuple(s) for s in meta_m["used_seeds_working"]]
saved_mask = load_saved_mask(MAIN_FRAME)

print("  Region growing ...")
rg_mask = run_region_growing(img_pre, seeds_snap, meta_m)

print("  Morphological cleaning ...")
mask_clean = run_cleaning(rg_mask, seeds_snap, meta_m)

pre_rgb = np.stack([img_pre] * 3, axis=-1).astype(np.float32) / 255.0


# ══════════════════════════════════════════════════════════════════════════════
# fig_442  Seed snapping
# ══════════════════════════════════════════════════════════════════════════════

print("\nfig_442_seed_snapping.png")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
show_panel(axes[0], img_pre, "Raw click positions",
           seeds=seeds_raw, seed_color="#FF3333")
show_panel(axes[1], img_pre, "After snap to nearest bright ridge",
           seeds=seeds_snap, seed_color="#00EE55")
plt.tight_layout()
save_fig(fig, "fig_442_seed_snapping.png")


# ══════════════════════════════════════════════════════════════════════════════
# fig_443  Region growing
# ══════════════════════════════════════════════════════════════════════════════

print("fig_443_region_growing.png")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
show_panel(axes[0], img_pre, "Preprocessed frame (CLAHE + Gaussian)")
show_panel(axes[1], color_overlay(img_pre, rg_mask),
           "Flood-fill region growing (blue)")
plt.tight_layout()
save_fig(fig, "fig_443_region_growing.png")


# ══════════════════════════════════════════════════════════════════════════════
# fig_444  Morphological cleaning
# ══════════════════════════════════════════════════════════════════════════════

print("fig_444_morphological_refinement.png")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
show_panel(axes[0], color_overlay(img_pre, rg_mask),
           "Raw region-growing mask")
show_panel(axes[1], color_overlay(img_pre, mask_clean > 0),
           "After morphological refinement")
plt.tight_layout()
save_fig(fig, "fig_444_morphological_refinement.png")


# ══════════════════════════════════════════════════════════════════════════════
# fig_445  Active contour
# ══════════════════════════════════════════════════════════════════════════════

print("fig_445_active_contour.png")
clean_contours = find_contours(mask_clean > 0, 0.5)
final_contours = find_contours(saved_mask  > 0, 0.5)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
show_panel(axes[0], pre_rgb,
           "Cleaned mask boundary (input to snake)",
           contours=clean_contours, contour_color="#00FF7F", contour_lw=2.0)
show_panel(axes[1], pre_rgb,
           "After active contour (snake) refinement",
           contours=final_contours, contour_color="#FFD700", contour_lw=2.0)
plt.tight_layout()
save_fig(fig, "fig_445_active_contour.png")


# ══════════════════════════════════════════════════════════════════════════════
# fig_446  Final mask output
# ══════════════════════════════════════════════════════════════════════════════

print("fig_446_mask_output.png")
mask_in_full = np.zeros(img_full.shape, dtype=bool)
mask_in_full[CROP_Y_MIN:CROP_Y_MAX, CROP_X_MIN:CROP_X_MAX] = (saved_mask > 0)
full_overlay = color_overlay(img_full, mask_in_full)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
show_panel(axes[0], saved_mask, "Final binary mask")
show_panel(axes[1], full_overlay, "Mask overlaid on original frame")
plt.tight_layout()
save_fig(fig, "fig_446_mask_output.png")


# ══════════════════════════════════════════════════════════════════════════════
# fig_44_gallery  — 3 rows × 3 cols
# ══════════════════════════════════════════════════════════════════════════════

print("\nfig_44_gallery.png")

fig, axes = plt.subplots(
    3, 3, figsize=(18, 14),
    gridspec_kw={"width_ratios": [2, 2, 1.6]},
)

for row_i, (frame_name, row_label) in enumerate(GALLERY_FRAMES):
    gm = load_meta(frame_name)
    gr = gm.get("refinement", {})
    _, _, g_pre = load_frame_and_preprocess(frame_name)
    g_mask  = load_saved_mask(frame_name)
    g_seeds = [tuple(s) for s in gm["used_seeds_working"]]

    # col 0: frame + seeds
    show_panel(axes[row_i, 0], g_pre, "",
               seeds=g_seeds, seed_color="#00EE55", seed_size=70)
    axes[row_i, 0].set_ylabel(row_label, fontsize=10, fontweight="bold",
                               rotation=0, labelpad=95, va="center")

    # col 1: mask overlay
    show_panel(axes[row_i, 1], color_overlay(g_pre, g_mask > 0), "")

    # col 2: parameter text box
    ax_t = axes[row_i, 2]
    ax_t.axis("off")
    lines = []
    for label, extractor in GALLERY_PARAMS:
        val = extractor(gm, gr)
        lines.append(f"{label:<20s}: {val}")
    ax_t.text(0.5, 0.5, "\n".join(lines),
              transform=ax_t.transAxes,
              ha="center", va="center",
              fontsize=11, fontfamily="monospace",
              bbox=dict(boxstyle="round,pad=0.5",
                        facecolor="#F0F0F0", edgecolor="#BBBBBB", linewidth=1))

plt.tight_layout(h_pad=2.0, w_pad=0.8)

# Column headers set after loop so show_panel("") doesn't overwrite them
for col_i, ct in enumerate(["Frame + seeds", "Final mask overlay",
                             "Segmentation parameters"]):
    axes[0, col_i].set_title(ct, fontsize=FS, fontweight="bold", pad=9)
save_fig(fig, "fig_44_gallery.png")

print("\nAll figures done.")
