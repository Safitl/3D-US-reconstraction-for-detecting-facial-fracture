"""
generate_overlay_comparison.py
Section 6.5 — Qualitative comparison overlays, Patient 2 test set.

For each selected frame: 4 panels in one row
  1. Original cropped frame (grayscale)
  2. GT mask contour in green
  3. U-Net prediction contour in blue
  4. SAM2Rad prediction contour in red

Frames selected by U-Net Dice (top 3 = success, bottom 3 = failure).

Outputs (200 dpi, report_figures/):
  fig_65_best_cases.png
  fig_65_worst_cases.png
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO     = Path(__file__).resolve().parent.parent
RUNS     = REPO / "Bone Segmentation" / "Deep Learning-Based Segmentation" / "runs"
RUNS_SAM2 = REPO / "Bone Segmentation" / "runs"
OUT_DIR  = REPO / "report_figures"

UNET_CSV    = RUNS / "unet_with_augmentation_v2_20260612_143014" / "eval_patient2" / "metrics_per_sample.csv"
SAM2_CSV    = RUNS_SAM2 / "sam2rad_bone_seg_v3_eval_ep85" / "metrics_per_sample.csv"
UNET_OV_DIR = RUNS / "unet_with_augmentation_v2_20260612_143014" / "eval_patient2" / "overlays"
SAM2_OV_DIR = RUNS_SAM2 / "sam2rad_bone_seg_v3_eval_ep85" / "overlays"

DISPLAY_W = 512   # common display size (U-Net panels are already 512×512)

# ── Frame selection ────────────────────────────────────────────────────────────
df_unet = pd.read_csv(UNET_CSV).sort_values("dice", ascending=False).reset_index(drop=True)
df_sam2 = pd.read_csv(SAM2_CSV).set_index("filename")

best_frames  = df_unet.head(3)["filename"].tolist()
worst_frames = df_unet.tail(3)["filename"].tolist()[::-1]   # lowest first → show ascending

print("Best frames (U-Net Dice):")
for f in best_frames:
    print(f"  {f}  Dice={df_unet.loc[df_unet.filename==f,'dice'].values[0]:.3f}")
print("Worst frames (U-Net Dice):")
for f in worst_frames:
    print(f"  {f}  Dice={df_unet.loc[df_unet.filename==f,'dice'].values[0]:.3f}")


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_panel_masks(filename: str):
    """
    Slice the saved 3-panel overlay strips to extract (img_bgr, gt_bin, pred_bin).
    U-Net panels: 512×512.  SAM2Rad panels: 1024×1024 → resized to DISPLAY_W.
    Returns uint8 arrays, all at DISPLAY_W × DISPLAY_W.
    """
    stem = Path(filename).stem

    # ── U-Net overlay ──────────────────────────────────────────────────────────
    unet_ov = cv2.imread(str(UNET_OV_DIR / f"{stem}_overlay.png"))
    pw = unet_ov.shape[1] // 3
    img_bgr      = unet_ov[:, :pw]           # panel 0: grayscale frame as BGR
    gt_panel     = unet_ov[:, pw:2*pw]       # panel 1: GT binary mask
    unet_panel   = unet_ov[:, 2*pw:]         # panel 2: U-Net prediction

    # resize to DISPLAY_W if needed
    if img_bgr.shape[1] != DISPLAY_W:
        img_bgr    = cv2.resize(img_bgr,    (DISPLAY_W, DISPLAY_W), interpolation=cv2.INTER_LINEAR)
        gt_panel   = cv2.resize(gt_panel,   (DISPLAY_W, DISPLAY_W), interpolation=cv2.INTER_NEAREST)
        unet_panel = cv2.resize(unet_panel, (DISPLAY_W, DISPLAY_W), interpolation=cv2.INTER_NEAREST)

    # ── SAM2Rad overlay ────────────────────────────────────────────────────────
    sam2_ov = cv2.imread(str(SAM2_OV_DIR / f"{stem}_overlay.png"))
    sw = sam2_ov.shape[1] // 3
    sam2_panel = sam2_ov[:, 2*sw:]           # panel 2: SAM2Rad prediction
    sam2_panel = cv2.resize(sam2_panel, (DISPLAY_W, DISPLAY_W), interpolation=cv2.INTER_NEAREST)

    # binarise all mask panels (0 / 255)
    def binarise(bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        _, b  = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        return b

    return img_bgr, binarise(gt_panel), binarise(unet_panel), binarise(sam2_panel)


def contour_overlay(img_bgr: np.ndarray, mask_bin: np.ndarray,
                    color_bgr: tuple, thickness: int = 2) -> np.ndarray:
    """Draw the contour of mask_bin on img_bgr."""
    out = img_bgr.copy()
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, color_bgr, thickness)
    return out


# ── Figure factory ─────────────────────────────────────────────────────────────

def make_figure(filenames, title, out_filename):
    n    = len(filenames)
    fig, axes = plt.subplots(n, 4, figsize=(16, 4.2 * n),
                             gridspec_kw={"wspace": 0.04, "hspace": 0.28})

    # Column headers (only on row 0)
    col_headers = [
        "Original frame",
        "Ground truth  (green)",
        "U-Net prediction  (blue)",
        "SAM2Rad prediction  (red)",
    ]
    for ci, hdr in enumerate(col_headers):
        axes[0, ci].set_title(hdr, fontsize=11, fontweight="bold", pad=7)

    for ri, fname in enumerate(filenames):
        img_bgr, gt_bin, unet_bin, sam2_bin = load_panel_masks(fname)

        unet_dice = df_unet.loc[df_unet.filename == fname, "dice"].values[0]
        sam2_dice = float(df_sam2.at[fname, "dice"])

        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_rgb    = cv2.cvtColor(contour_overlay(img_bgr, gt_bin,   (0, 200, 0)),   cv2.COLOR_BGR2RGB)
        unet_rgb  = cv2.cvtColor(contour_overlay(img_bgr, unet_bin, (255, 0, 0)),   cv2.COLOR_BGR2RGB)
        sam2_rgb  = cv2.cvtColor(contour_overlay(img_bgr, sam2_bin, (0, 0, 255)),   cv2.COLOR_BGR2RGB)

        stem = Path(fname).stem
        row_label = stem.replace("image_", "").replace("_f", "\nframe ")

        for ci, (panel, extra) in enumerate([
            (img_rgb,  ""),
            (gt_rgb,   ""),
            (unet_rgb, f"Dice = {unet_dice:.3f}"),
            (sam2_rgb, f"Dice = {sam2_dice:.3f}"),
        ]):
            ax = axes[ri, ci]
            ax.imshow(panel)
            ax.axis("off")
            if extra:
                ax.set_xlabel(extra, fontsize=9.5, labelpad=3)
                ax.xaxis.set_label_position("bottom")
                ax.xaxis.label.set_visible(True)

        # Row label on the left
        axes[ri, 0].set_ylabel(row_label, fontsize=8.5, rotation=0,
                               labelpad=60, va="center")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / out_filename
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_filename}")


# ── Generate both figures ──────────────────────────────────────────────────────

make_figure(
    best_frames,
    "Success Cases — 3 Frames with Highest U-Net Dice (Patient 2 Test Set)",
    "fig_65_best_cases.png",
)

make_figure(
    worst_frames,
    "Failure Cases — 3 Frames with Lowest U-Net Dice (Patient 2 Test Set)",
    "fig_65_worst_cases.png",
)

print("\nDone.")
