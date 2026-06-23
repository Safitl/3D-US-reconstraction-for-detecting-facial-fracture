"""
generate_preprocessing_comparison_figures.py — Figures A–D for the report.

  A — FFT spectrum: cropped frame + 2D log-magnitude spectrum side by side
  B — Frequency filter comparison: Original | Low-pass | High-pass | Band-pass
  C — SVD patch shape comparison: Original | Square 32×32 | Horiz 16×64 | Vert 64×16
  D — Bar chart: mean Dice per method (from compare_preprocessing_results/summary_metrics.csv)

All outputs saved to report_figures/
"""

import os
import sys
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE     = Path(__file__).resolve().parent
_REPO     = _HERE.parent
_PREPROC  = _REPO / "Bone Segmentation" / "Preprocessing"
_SEG      = _REPO / "Bone Segmentation" / "Region Growing Segmentation" / "seg"

for _p in (str(_PREPROC), str(_SEG)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import frequency_filters as ff
from preprocessing_api import preprocess

# ── Config ────────────────────────────────────────────────────────────────────
CROP_Y_MIN, CROP_Y_MAX = 100, 700
CROP_X_MIN, CROP_X_MAX = 200, 800

FRAME_PATH  = _REPO / "Dataset" / "Patient1" / "IMG_frames" / "image_406314327901_f032.png"
OUT_DIR     = _REPO / "report_figures"
SUMMARY_CSV = _REPO / "compare_preprocessing_results" / "summary_metrics.csv"

TITLE_FS = 17


def load_frame():
    img_full = cv2.imread(str(FRAME_PATH), cv2.IMREAD_GRAYSCALE)
    if img_full is None:
        raise FileNotFoundError(f"Cannot load: {FRAME_PATH}")
    cropped = img_full[CROP_Y_MIN:CROP_Y_MAX, CROP_X_MIN:CROP_X_MAX]
    return cropped, cropped.astype(np.float32) / 255.0  # uint8, float32


# ── Figure A: FFT spectrum ─────────────────────────────────────────────────────

def figure_a(img_u8, img_f32, out_path):
    F = np.fft.fftshift(np.fft.fft2(img_f32))
    log_mag = np.log1p(np.abs(F))
    freq_extent = [-0.5, 0.5, 0.5, -0.5]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(img_u8, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Cropped ultrasound frame", fontsize=TITLE_FS, fontweight="bold", pad=10)
    axes[0].axis("off")

    im = axes[1].imshow(log_mag, cmap="plasma", extent=freq_extent, aspect="auto")
    axes[1].set_title("2D FFT log-magnitude spectrum", fontsize=TITLE_FS, fontweight="bold", pad=10)
    axes[1].set_xlabel("fx (cycles/pixel)", fontsize=11)
    axes[1].set_ylabel("fy (cycles/pixel)", fontsize=11)
    plt.colorbar(im, ax=axes[1], shrink=0.85, label="log(1 + |F|)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure B: frequency filter comparison ─────────────────────────────────────

def figure_b(img_f32, out_path):
    lp = ff.apply_filter(img_f32, "fft_lowpass")
    hp = ff.apply_filter(img_f32, "fft_highpass")
    bp = ff.apply_filter(img_f32, "fft_bandpass")

    titles = ["Original", "Low-pass filter", "High-pass filter", "Band-pass filter"]
    images = [img_f32, lp, hp, bp]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(np.clip(img, 0, 1), cmap="gray")
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=10)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure C: SVD patch shape comparison ──────────────────────────────────────

def figure_c(img_u8, out_path):
    sq    = preprocess(img_u8, "patch_svd_square_32x32")
    horiz = preprocess(img_u8, "patch_svd_horizontal_16x64")
    vert  = preprocess(img_u8, "patch_svd_vertical_64x16")

    titles = [
        "Original (cropped)",
        "SVD — Square 32×32",
        "SVD — Horizontal 16×64",
        "SVD — Vertical 64×16",
    ]
    images = [img_u8, sq, horiz, vert]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=10)
        ax.axis("off")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Figure D: Dice bar chart ───────────────────────────────────────────────────

DISPLAY_NAMES = {
    "baseline":                   "Baseline\n(CLAHE + Gaussian)",
    "patch_svd_square_32x32":     "SVD Square\n32×32",
    "patch_svd_vertical_64x16":   "SVD Vertical\n64×16",
    "patch_svd_square_48x48":     "SVD Square\n48×48",
    "patch_svd_horizontal_16x64": "SVD Horizontal\n16×64",
    "patch_svd_horizontal_24x96": "SVD Horizontal\n24×96",
    "patch_svd_horizontal_32x96": "SVD Horizontal\n32×96",
}


def figure_d(out_path):
    df = pd.read_csv(SUMMARY_CSV)
    df = df.sort_values("dice_mean", ascending=False).reset_index(drop=True)

    labels = [DISPLAY_NAMES.get(m, m) for m in df["method"]]
    colors = ["#1565C0" if m == "baseline" else "#90CAF9" for m in df["method"]]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(labels, df["dice_mean"], color=colors,
                  yerr=df["dice_std"], capsize=5,
                  edgecolor="white", linewidth=0.5, error_kw={"elinewidth": 1.5})

    for bar, val in zip(bars, df["dice_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + df["dice_std"].max() + 0.015,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        )

    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Mean Dice Score  (n = 5 frames)", fontsize=12)
    ax.set_title(
        "Preprocessing Method Comparison — Mean Dice Score on Patient 1 Labeled Frames",
        fontsize=13, fontweight="bold", pad=14,
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    baseline_patch = mpatches.Patch(color="#1565C0", label="Baseline (selected method)")
    alt_patch      = mpatches.Patch(color="#90CAF9", label="Alternative preprocessing")
    ax.legend(handles=[baseline_patch, alt_patch], fontsize=10, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    img_u8, img_f32 = load_frame()

    figure_a(img_u8, img_f32, OUT_DIR / "fig_A_fft_spectrum.png")
    figure_b(img_f32,         OUT_DIR / "fig_B_fft_filter_comparison.png")
    figure_c(img_u8,          OUT_DIR / "fig_C_svd_patch_comparison.png")
    figure_d(                 OUT_DIR / "fig_D_dice_barchart.png")
