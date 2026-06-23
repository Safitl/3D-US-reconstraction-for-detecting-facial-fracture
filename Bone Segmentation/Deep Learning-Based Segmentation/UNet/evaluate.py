"""
U-Net evaluation script — computes Dice, IoU, Precision, Recall, Hausdorff
on the Patient 2 val set and saves results in the same format as SAM2Rad.

Usage (from repo root):
    python "Bone Segmentation/Deep Learning-Based Segmentation/UNet/evaluate.py" \
        --checkpoint "Bone Segmentation/Deep Learning-Based Segmentation/runs/unet_with_augmentation_20260531_143449/best_model.pth" \
        --csv_path Dataset/metadata_labeled.csv \
        --base_dir . \
        --val_patients Patient2
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import JawBoneSegmentationDataset
from unet_model import UNet2D


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(pred_bin: np.ndarray, target_bin: np.ndarray) -> dict:
    pred   = pred_bin.astype(np.float32).ravel()
    target = target_bin.astype(np.float32).ravel()
    tp = float((pred * target).sum())
    fp = float((pred * (1 - target)).sum())
    fn = float(((1 - pred) * target).sum())
    eps = 1e-6
    return {
        "dice":      (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou":       (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall":    (tp + eps) / (tp + fn + eps),
    }


def hausdorff_distance(pred_bin: np.ndarray, target_bin: np.ndarray) -> float:
    p = np.argwhere(pred_bin)
    t = np.argwhere(target_bin)
    if len(p) == 0 or len(t) == 0:
        return float("nan")
    return max(directed_hausdorff(p, t)[0], directed_hausdorff(t, p)[0])


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, loader, device, out_dir: Path, threshold: float = 0.5):
    model.eval()
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    results = []
    df = loader.dataset.df

    for i, batch in enumerate(loader):
        imgs  = batch["image"].to(device)
        masks = batch["mask"]

        probs = torch.sigmoid(model(imgs))
        preds = (probs > threshold).float().cpu()

        for j in range(imgs.size(0)):
            global_idx = i * loader.batch_size + j
            fname = df.iloc[global_idx]["image_path"] if global_idx < len(df) else f"sample_{global_idx}"

            pred_np = preds[j, 0].numpy().astype(np.uint8)
            gt_np   = masks[j, 0].numpy().astype(np.uint8)

            m = compute_metrics(pred_np, gt_np)
            m["hausdorff"] = hausdorff_distance(pred_np, gt_np)
            m["filename"]  = Path(fname).name
            results.append(m)

            # Overlay: image | ground truth | prediction
            img_np   = (imgs[j, 0].cpu().numpy() * 255).astype(np.uint8)
            img_bgr  = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
            gt_bgr   = cv2.cvtColor((gt_np   * 255), cv2.COLOR_GRAY2BGR)
            pred_bgr = cv2.cvtColor((pred_np  * 255), cv2.COLOR_GRAY2BGR)
            panel = np.concatenate([img_bgr, gt_bgr, pred_bgr], axis=1)
            cv2.imwrite(str(overlay_dir / f"{Path(fname).stem}_overlay.png"), panel)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to best_model.pth from a training run.")
    parser.add_argument("--csv_path",     default="../../Dataset/metadata_labeled.csv")
    parser.add_argument("--base_dir",     default="../../",
                        help="Repo root — CSV paths resolved relative to this.")
    parser.add_argument("--val_patients", nargs="+", default=["Patient2"])
    parser.add_argument("--img_size",     type=int, nargs=2, default=[512, 512])
    parser.add_argument("--base_ch",      type=int, default=32)
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--out_dir",      default="",
                        help="Output directory. Defaults to <run_dir>/eval_patient2/.")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint).resolve()
    out_dir   = Path(args.out_dir).resolve() if args.out_dir else ckpt_path.parent / "eval_patient2"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dir = str(Path(args.base_dir).resolve())
    csv_path = str(Path(args.csv_path).resolve())

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = tuple(args.img_size)

    val_ds = JawBoneSegmentationDataset(
        csv_path=csv_path, base_dir=base_dir,
        patient_ids=args.val_patients, img_size=img_size,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)
    print(f"Evaluating on {len(val_ds)} samples  ({args.val_patients})")

    model = UNet2D(in_channels=1, out_channels=1, base_ch=args.base_ch).to(device)
    model.load_state_dict(torch.load(str(ckpt_path), map_location=device))
    print(f"Loaded checkpoint: {ckpt_path}")

    results = evaluate(model, val_loader, device, out_dir)

    metrics = ["dice", "iou", "precision", "recall", "hausdorff"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in results if not (isinstance(r[m], float) and np.isnan(r[m]))]
        agg[m] = float(np.mean(vals)) if vals else float("nan")

    # Per-sample CSV
    csv_path_out = out_dir / "metrics_per_sample.csv"
    with open(csv_path_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename"] + metrics)
        writer.writeheader()
        for r in results:
            row = {"filename": r["filename"]}
            for m in metrics:
                v = r[m]
                row[m] = round(v, 6) if not (isinstance(v, float) and np.isnan(v)) else "nan"
            writer.writerow(row)

    # Summary CSV
    summary_path = out_dir / "metrics_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model"] + metrics)
        writer.writeheader()
        row = {"model": f"unet_{ckpt_path.parent.name}"}
        row.update({m: round(agg[m], 4) for m in metrics})
        writer.writerow(row)

    print(f"\n{'='*50}")
    print(f"U-Net evaluation on {args.val_patients}")
    print(f"Image size: {img_size}  |  Note: Hausdorff in {img_size[0]}px space")
    print(f"{'='*50}")
    for m in metrics:
        print(f"  {m:<12} {agg[m]:.4f}")
    print(f"\nSummary CSV : {summary_path}")
    print(f"Overlays    : {out_dir / 'overlays'}")


if __name__ == "__main__":
    main()
