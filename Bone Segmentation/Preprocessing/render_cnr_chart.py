"""One-shot: rebuild CNR summary CSV and chart from the already-saved per-frame CSV."""
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd

REPO    = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO / "compare_preprocessing_results"
FIG_OUT = REPO / "report_figures" / "fig_D_cnr_comparison.png"

METHODS = [
    "baseline", "fft_lowpass", "fft_highpass", "fft_bandpass",
    "patch_svd_square_32x32", "patch_svd_square_48x48",
    "patch_svd_vertical_64x16", "patch_svd_horizontal_16x64",
    "patch_svd_horizontal_24x96", "patch_svd_horizontal_32x96",
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

df = pd.read_csv(OUT_DIR / "cnr_per_frame.csv")

rows = []
for method in METHODS:
    vals = df.loc[df["method"] == method, "cnr"].dropna()
    rows.append({
        "method": method, "n_frames": len(vals),
        "cnr_mean": round(vals.mean(), 4), "cnr_std": round(vals.std(), 4),
    })
df_s = pd.DataFrame(rows).sort_values("cnr_mean", ascending=False)
df_s.to_csv(OUT_DIR / "cnr_summary.csv", index=False)

print("-- CNR Summary (mean +/- std, sorted by CNR) --")
for _, r in df_s.iterrows():
    print(f"  {r['method']:35s}: {r['cnr_mean']:.4f} +/- {r['cnr_std']:.4f}")

# Exclude fft_highpass from chart — extreme outlier (0.49) compresses the y-axis
df_plot = df_s[df_s["method"] != "fft_highpass"].reset_index(drop=True)

labels = [DISPLAY_NAMES.get(m, m) for m in df_plot["method"]]
colors = ["#1565C0" if m == "baseline" else "#90CAF9" for m in df_plot["method"]]

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(labels, df_plot["cnr_mean"], color=colors,
              yerr=df_plot["cnr_std"], capsize=5,
              edgecolor="white", linewidth=0.5, error_kw={"elinewidth": 1.5})

max_err = df_plot["cnr_std"].max()
for bar, val in zip(bars, df_plot["cnr_mean"]):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max_err + 0.005,
            f"{val:.3f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

y_min = 0
y_max = df_plot["cnr_mean"].max() + df_plot["cnr_std"].max() + 0.65
ax.set_ylim(y_min, y_max)
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
print(f"Chart saved -> {FIG_OUT}")
