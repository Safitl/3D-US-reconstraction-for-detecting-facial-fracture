import argparse
from pathlib import Path


import matplotlib.pyplot as plt
import pandas as pd


def plot_training(csv_path: str, out_dir: str,
                  title: str = "", loss_ylim: tuple = None):
    df = pd.read_csv(csv_path)
    df["val_hausdorff"] = pd.to_numeric(df["val_hausdorff"], errors="coerce")

    best_epoch = df.loc[df["val_dice"].idxmax(), "epoch"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Loss ──────────────────────────────────────────────────────────────
    loss_title = f"Training vs Validation Loss — {title}" if title else "Training vs Validation Loss"
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["epoch"], df["train_loss"], label="Train loss")
    ax.plot(df["epoch"], df["val_loss"],   label="Val loss")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.9,
               label=f"Best epoch ({int(best_epoch)})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(loss_title)
    if loss_ylim is not None:
        ax.set_ylim(loss_ylim)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "loss.png", dpi=150)
    plt.close(fig)

    # ── 2. Segmentation metrics ───────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for col, label in [("val_dice", "Dice"), ("val_iou", "IoU"),
                       ("val_precision", "Precision"), ("val_recall", "Recall")]:
        ax.plot(df["epoch"], df[col], label=label)
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.9,
               label=f"Best epoch ({int(best_epoch)})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Segmentation Metrics")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "metrics.png", dpi=150)
    plt.close(fig)

    # ── 3. Hausdorff distance ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["epoch"], df["val_hausdorff"], color="tab:red", label="Hausdorff (px)")
    ax.axvline(best_epoch, color="gray", linestyle="--", linewidth=0.9,
               label=f"Best epoch ({int(best_epoch)})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pixels")
    ax.set_title("Validation Hausdorff Distance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "hausdorff.png", dpi=150)
    plt.close(fig)

    print(f"Plots saved to: {out.resolve()}")
    print(f"Best epoch: {int(best_epoch)}  |  "
          f"Dice={df.loc[df['val_dice'].idxmax(), 'val_dice']:.4f}  |  "
          f"IoU={df.loc[df['val_dice'].idxmax(), 'val_iou']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training metrics. Pass either --run_dir or --csv_path."
    )
    parser.add_argument("--run_dir",  type=str, default="",
                        help="Path to a run directory (e.g. runs/unet_baseline_20260531_143022). "
                             "Plots are saved inside it automatically.")
    parser.add_argument("--csv_path", type=str, default="",
                        help="Direct path to metrics.csv (legacy).")
    parser.add_argument("--out_dir",  type=str, default="",
                        help="Output directory for plots (used with --csv_path).")
    parser.add_argument("--title", type=str, default="",
                        help="Optional subtitle appended to each plot title.")
    parser.add_argument("--loss_ylim", type=float, nargs=2, default=None,
                        metavar=("YMIN", "YMAX"),
                        help="Y-axis limits for the loss plot, e.g. --loss_ylim 0 0.85")
    args = parser.parse_args()

    if args.run_dir:
        run_dir  = Path(args.run_dir)
        csv_path = str(run_dir / "metrics.csv")
        out_dir  = str(run_dir / "plots")
    else:
        csv_path = args.csv_path or "results/metrics.csv"
        out_dir  = args.out_dir  or "results/plots"

    loss_ylim = tuple(args.loss_ylim) if args.loss_ylim is not None else None
    plot_training(csv_path, out_dir, title=args.title, loss_ylim=loss_ylim)


if __name__ == "__main__":
    main()
