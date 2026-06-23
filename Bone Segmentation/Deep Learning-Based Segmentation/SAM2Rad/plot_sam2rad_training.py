"""
plot_sam2rad_training.py
Plot SAM2Rad training curves from the CSVLogger output.

For v3 (val every epoch), reads directly from:
    logs/csv_metrics/version_0/metrics.csv
No wandb binary parsing needed.

Outputs saved to OUT_DIR:
    metrics.csv   — copy of the cleaned per-epoch CSV
    loss.png      — train_loss_seg vs val_loss_seg
    dice.png      — train_dice vs val_dice
    metrics.png   — val Dice + val IoU every epoch

--- UPDATE THESE TWO AFTER TRAINING ---
"""

import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ── Paths (update OUT_DIR and BEST_EPOCH after training) ───────────────────────
_HERE = Path(__file__).resolve().parent

# CSVLogger output.
# The v3 run was split across two CSVLogger versions because training was
# resumed: version_6 holds epochs 0-37 (first launch) and version_7 holds
# epochs 38-99 (resumed launch).  List every version whose CSV belongs to the
# run you want to plot; they are concatenated below.
CSV_INS = [
    _HERE / "logs" / "csv_metrics" / "version_6" / "metrics.csv",
    _HERE / "logs" / "csv_metrics" / "version_7" / "metrics.csv",
]
CSV_INS = [p for p in CSV_INS if p.exists()]
if not CSV_INS:
    raise FileNotFoundError("None of the listed CSVLogger metrics.csv files exist. "
                            "Has training completed?")

# TODO: set BEST_EPOCH to the epoch with the highest val_dice after training
BEST_EPOCH = None   # e.g. 47  ← fill in after training

# TODO: set OUT_DIR to the eval output directory for the best epoch
OUT_DIR = _HERE.parent.parent / "runs" / "sam2rad_bone_seg_v3_eval_plots"
# e.g.: _HERE.parent.parent / "runs" / "sam2rad_bone_seg_v3_eval_ep47" / "plots"
# ───────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load CSV(s) ──────────────────────────────────────────────────────────────
df = pd.concat([pd.read_csv(p) for p in CSV_INS], ignore_index=True)

# PL CSVLogger writes several rows per epoch: train metrics land on one row and
# val metrics on another (each metric is non-null in exactly one row per epoch).
# Collapse to one row per epoch by taking the mean per column, which skips NaNs
# and therefore recovers the single logged value for every metric — keeping
# both the train AND val curves (drop_duplicates(keep="last") would lose one).
df = df.groupby("epoch", as_index=False).mean(numeric_only=True)
df = df.sort_values("epoch").reset_index(drop=True)

# epoch column is 0-indexed in PL → convert to 1-indexed for plots
df["epoch_1"] = df["epoch"] + 1

TRAIN_KEYS = ["train_loss_seg", "train_dice", "train_iou",
              "train_loss_box", "train_loss_object", "interim_mask_loss"]
VAL_KEYS   = ["val_loss_seg", "val_dice", "val_iou", "val_loss_object"]

print(f"Loaded {len(df)} epochs from {[str(p.parent.name) for p in CSV_INS]}")
print(f"Columns: {list(df.columns)}")

# Save a clean copy
out_csv = OUT_DIR / "metrics.csv"
df.to_csv(out_csv, index=False)
print(f"CSV saved -> {out_csv}")

# Auto-detect best epoch if not set
if BEST_EPOCH is None:
    if "val_dice" in df.columns:
        best_row = df.loc[df["val_dice"].idxmax()]
        BEST_EPOCH = int(best_row["epoch"])
        print(f"Auto-detected best epoch (0-indexed): {BEST_EPOCH}  "
              f"val_dice={best_row['val_dice']:.4f}")
    else:
        BEST_EPOCH = int(df["epoch"].iloc[-1])
        print(f"val_dice column not found — defaulting BEST_EPOCH={BEST_EPOCH}")

# ── Helper ─────────────────────────────────────────────────────────────────────
def plot_series(ax, df, key, label, color, style="-"):
    col_data = df[["epoch_1", key]].dropna()
    if col_data.empty:
        return
    ax.plot(col_data["epoch_1"], col_data[key], style, label=label,
            color=color, linewidth=1.2)

# ── Plot 1: Loss ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
plot_series(ax, df, "train_loss_seg", "Train loss (Dice+Focal)", "tab:blue")
plot_series(ax, df, "val_loss_seg",   "Val loss (Dice+Focal)",   "tab:orange", "-")
ax.axvline(BEST_EPOCH + 1, color="gray", linestyle="--", linewidth=0.9,
           label=f"Best epoch ({BEST_EPOCH})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Segmentation Loss (Dice + Focal)")
ax.set_title("Training vs Validation Loss — SAM2Rad")
ax.set_ylim(0, 0.85)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "loss.png", dpi=150)
plt.close(fig)
print("loss.png saved")

# ── Plot 2: Dice ───────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
plot_series(ax, df, "train_dice", "Train Dice", "tab:blue")
plot_series(ax, df, "val_dice",   "Val Dice",   "tab:orange", "o-")
ax.axvline(BEST_EPOCH + 1, color="gray", linestyle="--", linewidth=0.9,
           label=f"Best epoch ({BEST_EPOCH})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Dice Score")
ax.set_title("Training vs Validation Dice — SAM2Rad")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "dice.png", dpi=150)
plt.close(fig)
print("dice.png saved")

# ── Plot 3: Val Dice + IoU ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 4))
plot_series(ax, df, "val_dice", "Val Dice", "tab:blue",   "o-")
plot_series(ax, df, "val_iou",  "Val IoU",  "tab:orange", "o-")
ax.axvline(BEST_EPOCH + 1, color="gray", linestyle="--", linewidth=0.9,
           label=f"Best epoch ({BEST_EPOCH})")
ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.set_title("Validation Metrics — SAM2Rad")
ax.set_ylim(0, 1)
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(OUT_DIR / "metrics.png", dpi=150)
plt.close(fig)
print("metrics.png saved")

# ── Summary ────────────────────────────────────────────────────────────────────
best_row = df[df["epoch"] == BEST_EPOCH]
if not best_row.empty:
    r = best_row.iloc[0]
    vd = r.get("val_dice", float("nan"))
    vi = r.get("val_iou",  float("nan"))
    print(f"\nBest epoch (1-indexed): {BEST_EPOCH + 1} | "
          f"Val Dice={vd:.4f} | Val IoU={vi:.4f}")

print(f"\nAll plots saved to: {OUT_DIR.resolve()}")
print("\nReminder: update OUT_DIR in this script to the actual eval directory "
      "(e.g. sam2rad_bone_seg_v3_eval_ep{BEST_EPOCH}) once you know the best epoch.")
