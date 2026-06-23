"""
U-Net training script.

Usage:
    python UNet/train.py                                      # default config
    python UNet/train.py --config UNet/configs/default.yaml  # explicit config
    python UNet/train.py --config UNet/configs/default.yaml --run_name my_exp
"""

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from scipy.spatial.distance import directed_hausdorff
from torch.utils.data import DataLoader

# Allow importing from the parent DL folder (dataset.py lives there)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import JawBoneSegmentationDataset
from augmentation import build_train_augmentation
from unet_model import UNet2D


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def dice_loss(probs, targets, smooth=1.0):
    p = probs.view(-1)
    t = targets.view(-1)
    intersection = (p * t).sum()
    return 1.0 - (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)


def combined_loss(probs, targets):
    return 0.5 * F.binary_cross_entropy(probs, targets) + 0.5 * dice_loss(probs, targets)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_batch_metrics(probs, targets, threshold=0.5):
    pred = (probs > threshold).float()
    tp = (pred * targets).sum()
    fp = (pred * (1 - targets)).sum()
    fn = ((1 - pred) * targets).sum()
    eps = 1e-6
    return {
        "dice":      ((2 * tp + eps) / (2 * tp + fp + fn + eps)).item(),
        "iou":       ((tp + eps) / (tp + fp + fn + eps)).item(),
        "precision": ((tp + eps) / (tp + fp + eps)).item(),
        "recall":    ((tp + eps) / (tp + fn + eps)).item(),
    }


def hausdorff_distance(pred_bin, target_bin):
    p = np.argwhere(pred_bin)
    t = np.argwhere(target_bin)
    if len(p) == 0 or len(t) == 0:
        return float("nan")
    return max(directed_hausdorff(p, t)[0], directed_hausdorff(t, p)[0])


def batch_hausdorff(pred_np, target_np):
    values = []
    for i in range(pred_np.shape[0]):
        hd = hausdorff_distance(pred_np[i, 0], target_np[i, 0])
        if not np.isnan(hd):
            values.append(hd)
    return float(np.mean(values)) if values else float("nan")


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        imgs  = batch["image"].to(device)
        masks = batch["mask"].to(device)
        probs = torch.sigmoid(model(imgs))
        loss  = combined_loss(probs, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    all_metrics = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
    hd_values = []
    n = 0
    for batch in loader:
        imgs  = batch["image"].to(device)
        masks = batch["mask"].to(device)
        probs = torch.sigmoid(model(imgs))
        total_loss += combined_loss(probs, masks).item() * imgs.size(0)
        m = compute_batch_metrics(probs, masks)
        for k in all_metrics:
            all_metrics[k] += m[k] * imgs.size(0)
        pred_np = (probs.cpu().numpy() > 0.5).astype(np.uint8)
        tgt_np  = masks.cpu().numpy().astype(np.uint8)
        hd = batch_hausdorff(pred_np, tgt_np)
        if not np.isnan(hd):
            hd_values.append(hd * imgs.size(0))
        n += imgs.size(0)
    avg = {k: v / n for k, v in all_metrics.items()}
    avg["hausdorff"] = float(np.sum(hd_values) / n) if hd_values else float("nan")
    avg["loss"] = total_loss / n
    return avg


# ---------------------------------------------------------------------------
# Overlay saving
# ---------------------------------------------------------------------------

def save_overlays(model, loader, device, out_dir, max_images=20):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    saved = 0
    with torch.no_grad():
        for batch in loader:
            imgs  = batch["image"].to(device)
            masks = batch["mask"]
            preds = (torch.sigmoid(model(imgs)) > 0.5).float().cpu()
            for i in range(imgs.size(0)):
                if saved >= max_images:
                    return
                img_np  = (imgs[i, 0].cpu().numpy() * 255).astype(np.uint8)
                gt_np   = (masks[i, 0].numpy()       * 255).astype(np.uint8)
                pred_np = (preds[i, 0].numpy()        * 255).astype(np.uint8)
                panel = np.concatenate([
                    cv2.cvtColor(img_np,  cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(gt_np,   cv2.COLOR_GRAY2BGR),
                    cv2.cvtColor(pred_np, cv2.COLOR_GRAY2BGR),
                ], axis=1)
                cv2.imwrite(os.path.join(out_dir, f"overlay_{saved:04d}.png"), panel)
                saved += 1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",   type=str,
                        default=str(Path(__file__).parent / "configs" / "default.yaml"))
    parser.add_argument("--run_name", type=str, default="",
                        help="Override the run_name in the config.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg["run_name"] = args.run_name

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{cfg['run_name']}_{timestamp}"
    runs_dir  = (Path(__file__).parent / cfg["output"]["runs_dir"]).resolve()
    run_dir   = runs_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy(args.config, run_dir / "config.yaml")
    print(f"Run directory: {run_dir}\n")

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = tuple(cfg["data"]["img_size"])
    base_dir = str((Path(__file__).parent / cfg["data"]["base_dir"]).resolve())
    csv_path = str((Path(__file__).parent / cfg["data"]["csv_path"]).resolve())

    df = pd.read_csv(csv_path)
    available_patients  = set(df["patient_id"].unique())
    val_patients        = cfg["data"]["val_patients"]
    val_patients_avail  = all(p in available_patients for p in val_patients)

    if val_patients_avail:
        train_patient_ids = cfg["data"]["train_patients"]
        val_patient_ids   = val_patients
        train_scan_ids    = None
        val_scan_ids      = None
        print(f"Patient-wise split: train={train_patient_ids}, val={val_patient_ids}")
    else:
        all_scans      = sorted(df[df["patient_id"].isin(cfg["data"]["train_patients"])]["scan_id"].unique())
        val_scan_ids   = cfg["data"]["val_scans"] or [all_scans[-1]]
        train_scan_ids = [s for s in all_scans if s not in val_scan_ids]
        train_patient_ids = val_patient_ids = cfg["data"]["train_patients"]
        print(f"Scan-level split: train_scans={train_scan_ids}, val_scans={val_scan_ids}")

    aug_cfg   = cfg.get("augmentation", {})
    train_aug = build_train_augmentation(aug_cfg)
    if train_aug is not None:
        print("Augmentation enabled for training set.")

    train_ds = JawBoneSegmentationDataset(
        csv_path=csv_path, base_dir=base_dir,
        patient_ids=train_patient_ids, scan_ids=train_scan_ids,
        img_size=img_size, transform=train_aug,
    )
    val_ds = JawBoneSegmentationDataset(
        csv_path=csv_path, base_dir=base_dir,
        patient_ids=val_patient_ids, scan_ids=val_scan_ids,
        img_size=img_size,  # no augmentation on val
    )
    print(f"Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}\n")

    train_loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"],
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["training"]["batch_size"],
                              shuffle=False, num_workers=0)

    m_cfg  = cfg["model"]
    model  = UNet2D(in_channels=m_cfg["in_channels"],
                    out_channels=m_cfg["out_channels"],
                    base_ch=m_cfg["base_ch"]).to(device)
    t_cfg  = cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=t_cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max",
        factor=t_cfg["scheduler_factor"],
        patience=t_cfg["scheduler_patience"],
    )

    best_dice  = 0.0
    best_path  = run_dir / "best_model.pth"
    log_path   = run_dir / "metrics.csv"
    log_fields = ["epoch", "train_loss", "val_loss",
                  "val_dice", "val_iou", "val_precision", "val_recall", "val_hausdorff"]

    with open(log_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=log_fields).writeheader()

    for epoch in range(1, t_cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_m      = eval_one_epoch(model, val_loader, device)
        scheduler.step(val_m["dice"])

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f}  val_loss={val_m['loss']:.4f}  "
            f"dice={val_m['dice']:.4f}  iou={val_m['iou']:.4f}  "
            f"prec={val_m['precision']:.4f}  rec={val_m['recall']:.4f}  "
            f"hd={val_m['hausdorff']:.2f}"
        )

        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=log_fields).writerow({
                "epoch":         epoch,
                "train_loss":    round(train_loss, 6),
                "val_loss":      round(val_m["loss"], 6),
                "val_dice":      round(val_m["dice"], 6),
                "val_iou":       round(val_m["iou"], 6),
                "val_precision": round(val_m["precision"], 6),
                "val_recall":    round(val_m["recall"], 6),
                "val_hausdorff": round(val_m["hausdorff"], 4)
                                 if not np.isnan(val_m["hausdorff"]) else "nan",
            })

        if val_m["dice"] > best_dice:
            best_dice = val_m["dice"]
            torch.save(model.state_dict(), best_path)
            print(f"  -> New best Dice={best_dice:.4f}, model saved.")

    print(f"\nTraining complete. Best val Dice: {best_dice:.4f}")
    print(f"Run saved to: {run_dir}")

    print("Saving overlay figures...")
    model.load_state_dict(torch.load(best_path, map_location=device))
    save_overlays(model, val_loader, device, out_dir=str(run_dir / "overlays"))
    print(f"Overlays saved to: {run_dir / 'overlays'}")


if __name__ == "__main__":
    main()
