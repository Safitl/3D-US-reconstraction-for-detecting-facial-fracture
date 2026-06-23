import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class JawBoneSegmentationDataset(Dataset):
    """
    Loads image-mask pairs from metadata_labeled.csv.

    Paths in the CSV are relative to `base_dir` (repo root by default).
    Each image is cropped to the working region recorded in the meta JSON
    so that image and mask are the same spatial extent before resizing.

    Args:
        csv_path:    Path to metadata_labeled.csv.
        base_dir:    Directory from which CSV paths are resolved (repo root).
        patient_ids: Optional list of patient_id strings to keep.
                     None = keep all rows.
        img_size:    (H, W) to resize both image and mask before returning.
        transform:   Optional callable applied to the {"image", "mask"} dict
                     before tensor conversion.
    """

    def __init__(self, csv_path, base_dir, patient_ids=None, scan_ids=None,
                 img_size=(512, 512), transform=None):
        df = pd.read_csv(csv_path)
        if patient_ids is not None:
            df = df[df["patient_id"].isin(patient_ids)]
        if scan_ids is not None:
            df = df[df["scan_id"].isin(scan_ids)]
        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError(
                f"No rows found for patient_ids={patient_ids}, scan_ids={scan_ids} in {csv_path}"
            )
        self.df = df
        self.base_dir = base_dir
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        meta_path = os.path.join(self.base_dir, row["meta_path"])
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        x_min, y_min, x_max, y_max = meta["crop_box"]

        img_path = os.path.join(self.base_dir, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {img_path}")
        img = img[y_min:y_max, x_min:x_max].astype(np.float32)

        mask_path = os.path.join(self.base_dir, row["mask_path"])
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not load mask: {mask_path}")
        mask = mask.astype(np.float32)

        h, w = self.img_size
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        img = img / 255.0
        mask = (mask > 0).astype(np.float32)

        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        sample = {"image": img, "mask": mask}
        if self.transform is not None:
            sample = self.transform(sample)

        sample["image"] = torch.from_numpy(sample["image"])
        sample["mask"] = torch.from_numpy(sample["mask"])
        return sample
