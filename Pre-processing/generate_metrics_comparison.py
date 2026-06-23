"""
generate_metrics_comparison.py
Grouped bar chart: U-Net vs SAM2Rad on Patient 2 test set (30 frames).
Metrics: Dice, IoU, Precision, Recall  (Hausdorff excluded — different scale).
Saves report_figures/fig_metrics_comparison.png at 200 dpi.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
RUNS = REPO / "Bone Segmentation" / "Deep Learning-Based Segmentation" / "runs"
RUNS_SAM2 = REPO / "Bone Segmentation" / "runs"

UNET_CSV   = RUNS / "unet_with_augmentation_v2_20260612_143014" / "eval_patient2" / "metrics_per_sample.csv"
SAM2_CSV   = RUNS_SAM2 / "sam2rad_bone_seg_v3_eval_ep85" / "metrics_per_sample.csv"
OUT_PATH   = REPO / "report_figures" / "fig_metrics_comparison.png"

# ── Load data ──────────────────────────────────────────────────────────────────
unet = pd.read_csv(UNET_CSV)
sam2 = pd.read_csv(SAM2_CSV)

METRICS     = ["dice", "iou", "precision", "recall"]
LABELS      = ["Dice", "IoU", "Precision", "Recall"]

unet_means  = [unet[m].mean() for m in METRICS]
unet_stds   = [unet[m].std()  for m in METRICS]
sam2_means  = [sam2[m].mean() for m in METRICS]
sam2_stds   = [sam2[m].std()  for m in METRICS]

print(f"U-Net  (n={len(unet)}):  " + "  ".join(f"{m}={v:.3f}±{s:.3f}"
      for m, v, s in zip(LABELS, unet_means, unet_stds)))
print(f"SAM2Rad(n={len(sam2)}):  " + "  ".join(f"{m}={v:.3f}±{s:.3f}"
      for m, v, s in zip(LABELS, sam2_means, sam2_stds)))

# ── Plot ───────────────────────────────────────────────────────────────────────
x         = np.arange(len(METRICS))
bar_w     = 0.32
gap       = 0.06   # half-gap between the two bars of a group

UNET_COLOR  = "#1565C0"   # dark blue
SAM2_COLOR  = "#E8734A"   # coral / orange

fig, ax = plt.subplots(figsize=(9, 5.5))

bars_u = ax.bar(x - bar_w / 2 - gap / 2, unet_means, bar_w,
                yerr=unet_stds, capsize=5,
                color=UNET_COLOR, label="U-Net + Augmentation",
                error_kw=dict(elinewidth=1.4, ecolor="#333333", capthick=1.4))

bars_s = ax.bar(x + bar_w / 2 + gap / 2, sam2_means, bar_w,
                yerr=sam2_stds, capsize=5,
                color=SAM2_COLOR, label="SAM2Rad (epoch 85)",
                error_kw=dict(elinewidth=1.4, ecolor="#333333", capthick=1.4))

# Value labels above each bar (placed above the error cap)
for bars, means, stds in [(bars_u, unet_means, unet_stds),
                          (bars_s, sam2_means, sam2_stds)]:
    for bar, val, std in zip(bars, means, stds):
        label_y = bar.get_height() + std + 0.012
        ax.text(bar.get_x() + bar.get_width() / 2, label_y,
                f"{val:.3f}",
                ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#111111")

# Axes formatting
ax.set_xticks(x)
ax.set_xticklabels(LABELS, fontsize=13)
ax.set_ylim(0, 1.0)
ax.set_ylabel("Score  (60 Patient 2 frames)", fontsize=12)
ax.set_title("U-Net vs SAM2Rad — Segmentation Metrics on Patient 2 Test Set",
             fontsize=13, fontweight="bold", pad=13)

ax.yaxis.grid(False)
ax.xaxis.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(fontsize=11, loc="lower right", framealpha=0.9,
          edgecolor="#CCCCCC")

ax.tick_params(axis="y", labelsize=11)

plt.tight_layout()
OUT_PATH.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved -> {OUT_PATH}")
