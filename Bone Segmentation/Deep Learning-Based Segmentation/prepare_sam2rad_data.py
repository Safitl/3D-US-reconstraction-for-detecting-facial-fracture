"""
Prepares data for SAM2Rad training.

Reads metadata_labeled.csv and creates the folder structure expected by SAM2Rad:
    SAM2Rad/datasets/bone_segmentation/
        Train/
            imgs/   ← cropped grayscale PNGs  (Patient1)
            gts/    ← binary masks, 0=bg 1=bone (Patient1)
        Test/
            imgs/   ← cropped grayscale PNGs  (Patient2)
            gts/    ← binary masks, 0=bg 1=bone (Patient2)

Images are cropped using the crop_box stored in each frame's meta JSON,
so they match the 600x600 mask spatial extent.
Masks are converted from 0/255 → 0/1 (pixel value = class ID).
"""

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def process_split(df_split, base_dir: Path, out_imgs: Path, out_gts: Path):
    out_imgs.mkdir(parents=True, exist_ok=True)
    out_gts.mkdir(parents=True, exist_ok=True)

    for _, row in df_split.iterrows():
        # ── load meta to get crop box ──────────────────────────────────────
        meta_path = base_dir / row["meta_path"]
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        x_min, y_min, x_max, y_max = meta["crop_box"]

        # ── image: load full frame, apply crop, save as grayscale PNG ─────
        img_path = base_dir / row["image_path"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  WARNING: could not load image {img_path}, skipping.")
            continue
        img_crop = img[y_min:y_max, x_min:x_max]

        stem = Path(row["image_path"]).stem
        out_img_file = out_imgs / f"{stem}.png"
        cv2.imwrite(str(out_img_file), img_crop)

        # ── mask: convert 0/255 → 0/1 (class IDs) ────────────────────────
        mask_path = base_dir / row["mask_path"]
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  WARNING: could not load mask {mask_path}, skipping.")
            continue
        mask_class = (mask > 0).astype(np.uint8)   # 0=background, 1=bone

        out_gts_file = out_gts / f"{stem}.png"
        cv2.imwrite(str(out_gts_file), mask_class)

    print(f"  Saved {len(df_split)} pairs -> {out_imgs.parent}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path",     type=str,
                        default="../../Dataset/metadata_labeled.csv")
    parser.add_argument("--base_dir",     type=str, default="../../",
                        help="Repo root — CSV paths resolved relative to this.")
    parser.add_argument("--out_dir",      type=str,
                        default="SAM2Rad/datasets/bone_segmentation",
                        help="Output root (relative to this script).")
    parser.add_argument("--train_patient", type=str, default="Patient1")
    parser.add_argument("--test_patient",  type=str, default="Patient2")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    base_dir   = (script_dir / args.base_dir).resolve()
    csv_path   = (script_dir / args.csv_path).resolve()
    out_dir    = (script_dir / args.out_dir).resolve()

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    df_train = df[df["patient_id"] == args.train_patient]
    df_test  = df[df["patient_id"] == args.test_patient]

    print(f"\nTrain ({args.train_patient}): {len(df_train)} samples")
    process_split(df_train, base_dir,
                  out_dir / "Train" / "imgs",
                  out_dir / "Train" / "gts")

    if len(df_test) > 0:
        print(f"\nTest ({args.test_patient}): {len(df_test)} samples")
        process_split(df_test, base_dir,
                      out_dir / "Test" / "imgs",
                      out_dir / "Test" / "gts")
    else:
        print(f"\nNo {args.test_patient} data found in CSV — Test split skipped.")

    print(f"\nDone. Dataset written to: {out_dir}")


if __name__ == "__main__":
    main()
