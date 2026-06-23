"""
compute_cnr_comparison.py — CNR analysis across all explored preprocessing methods.

CNR = |mean_bone - mean_background| / std_background
  bone        : pixels where GT mask > 0   (cortical bone ridge)
  background  : pixels where GT mask == 0

Applied to all labeled Patient 1 frames (83 frames).

Outputs
-------
  compare_preprocessing_results/cnr_per_frame.csv   — one row per (frame, method)
  compare_preprocessing_results/cnr_summary.csv     — mean ± std per method
  report_figures/fig_D_cnr_comparison.png           — bar chart
"""

import sys
import time
from pathlib import Path

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
_REPO    = _HERE.parent.parent

for _p in (str(_HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from preprocessing_api import preprocess

CROP_Y_MIN, CROP_Y_MAX = 100, 700
CROP_X_MIN, CROP_X_MAX = 200, 800

CSV_PATH    = _REPO / "Dataset" / "metadata_labeled.csv"
OUT_DIR     = _REPO / "compare_preprocessing_results"
FIG_OUT     = _REPO / "report_figures" / "fig_D_cnr_comparison.png"

# ── Methods to benchmark ──────────────────────────────────────────────────────
METHODS = [
    "baseline",
    "fft_lowpass",
    "fft_highpass",
    "fft_bandpass",
    "patch_svd_square_32x32",
    "patch_svd_square_48x48",
    "patch_svd_vertical_64x16",
    "patch_svd_horizontal_16x64",
    "patch_svd_horizontal_24x96",
    "patch_svd_horizontal_32x96",
    "wavelet_bayes_soft",
]

DISPLAY_NAMES = {
    "baseline":                   "Baseline\n(CLAHE+Gaussian)",
    "fft_lowpass":                "FFT\nLow-pass",
    "fft_highpass":               "FFT\nHigh-pass",
    "fft_bandpass":               "FFT\nBand-pass",
    "patch_svd_square_32x32":     "SVD Square\n32×32",
    "patch_svd_square_48x48":     "SVD Square\n48×48",
    "patch_svd_vertical_64x16":   "SVD Vertical\n64×16",
    "patch_svd_horizontal_16x64": "SVD Horiz\n16×64",
    "patch_svd_horizontal_24x96": "SVD Horiz\n24×96",
    "patch_svd_horizontal_32x96": "SVD Horiz\n32×96",
    "wavelet_bayes_soft":         "Wavelet\nBayesShrink",
}


# ── CNR computation ────────────────────────────────────────────────────────────

def compute_cnr(preprocessed: np.ndarray, mask_bool: np.ndarray) -> float:
    """
    CNR = |mean_bone - mean_background| / std_background

    preprocessed : uint8 image (values 0-255) after preprocessing
    mask_bool    : bool array, True = bone, same shape as preprocessed
    """
    img_f = preprocessed.astype(np.float32) / 255.0
    bone_px = img_f[mask_bool]
    bg_px   = img_f[~mask_bool]
    if len(bone_px) == 0 or len(bg_px) < 2:
        return float("nan")
    std_bg = bg_px.std()
    if std_bg < 1e-8:
        return float("nan")
    return float(abs(bone_px.mean() - bg_px.mean()) / std_bg)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df_csv = pd.read_csv(CSV_PATH)
    # Patient 1 only
    df_p1 = df_csv[df_csv["patient_id"] == "Patient1"].reset_index(drop=True)
    print(f"Patient 1 frames : {len(df_p1)}")
    print(f"Methods          : {len(METHODS)}")
    print(f"Total runs       : {len(df_p1) * len(METHODS)}")
    print()

    rows = []
    t_start = time.time()
    total = len(df_p1) * len(METHODS)
    done  = 0

    for _, row in df_p1.iterrows():
        img_path  = _REPO / row["image_path"]
        mask_path = _REPO / row["mask_path"]

        img_full = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img_full is None or mask_raw is None:
            print(f"  [SKIP] {img_path.name}")
            done += len(METHODS)
            continue

        cropped   = img_full[CROP_Y_MIN:CROP_Y_MAX, CROP_X_MIN:CROP_X_MAX]
        mask_bool = mask_raw > 127          # masks are already at crop resolution

        for method in METHODS:
            pre = preprocess(cropped, method)
            cnr = compute_cnr(pre, mask_bool)

            rows.append({
                "patient_id": row["patient_id"],
                "scan_id":    row["scan_id"],
                "frame_id":   row["frame_id"],
                "method":     method,
                "cnr":        round(cnr, 5),
            })
            done += 1

        elapsed = time.time() - t_start
        rate    = done / elapsed if elapsed > 0 else 1e-9
        eta     = (total - done) / rate
        print(
            f"  [{done:4d}/{total}]  {row['scan_id']:25s}  f{row['frame_id']:<4}  "
            f"ETA {eta/60:.1f} min",
            end="\r",
        )

    print()

    df_results = pd.DataFrame(rows)
    OUT_DIR.mkdir(exist_ok=True)

    per_frame_path = OUT_DIR / "cnr_per_frame.csv"
    df_results.to_csv(per_frame_path, index=False)
    print(f"\nPer-frame CNR  -> {per_frame_path}")

    # Summary
    summary_rows = []
    for method in METHODS:
        vals = df_results.loc[df_results["method"] == method, "cnr"].dropna()
        summary_rows.append({
            "method":   method,
            "n_frames": len(vals),
            "cnr_mean": round(vals.mean(), 4),
            "cnr_std":  round(vals.std(),  4),
        })
    df_summary = pd.DataFrame(summary_rows).sort_values("cnr_mean", ascending=False)
    df_summary.to_csv(OUT_DIR / "cnr_summary.csv", index=False)

    print("\n-- CNR Summary (mean +/- std, sorted) -----------------------------------")
    for _, r in df_summary.iterrows():
        bar = "#" * int(r["cnr_mean"] * 15)
        print(f"  {r['method']:35s}: {r['cnr_mean']:.4f} +/- {r['cnr_std']:.4f}  {bar}")

    _save_chart(df_summary)
    print(f"\nTotal time: {(time.time() - t_start)/60:.1f} min")


def _save_chart(df_summary: pd.DataFrame):
    labels = [DISPLAY_NAMES.get(m, m) for m in df_summary["method"]]
    colors = ["#1565C0" if m == "baseline" else "#90CAF9" for m in df_summary["method"]]

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(
        labels, df_summary["cnr_mean"], color=colors,
        yerr=df_summary["cnr_std"], capsize=5,
        edgecolor="white", linewidth=0.5,
        error_kw={"elinewidth": 1.5},
    )

    for bar, val in zip(bars, df_summary["cnr_mean"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + df_summary["cnr_std"].max() + 0.02,
            f"{val:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
        )

    ax.set_ylabel("Mean CNR  (n = 83 Patient 1 frames)", fontsize=12)
    ax.set_title(
        "Preprocessing Method Comparison — Contrast-to-Noise Ratio on Patient 1",
        fontsize=15, fontweight="bold", pad=14,
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.yaxis.grid(True, alpha=0.35, linestyle="--")
    ax.set_axisbelow(True)

    baseline_patch = mpatches.Patch(color="#1565C0", label="Baseline (selected method)")
    alt_patch      = mpatches.Patch(color="#90CAF9", label="Alternative preprocessing")
    ax.legend(handles=[baseline_patch, alt_patch], fontsize=10, loc="upper right")

    plt.tight_layout()
    FIG_OUT.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"CNR chart      -> {FIG_OUT}")


if __name__ == "__main__":
    main()
