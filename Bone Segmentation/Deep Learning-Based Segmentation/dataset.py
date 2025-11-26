import os
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset


class JawBoneSegmentationDataset(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        """
        csv_path: path to metadata_labeled.csv
        root_dir: Dataset root folder (e.g. 'Dataset')
        transform: optional transforms applied to (image, mask)
        """
        self.df = pd.read_csv(csv_path)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        path = os.path.join(self.root_dir, rel_path)
        img = Image.open(path).convert("L")
        return np.array(img, dtype=np.float32)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_image(row["image_path"])
        mask = self._load_image(row["mask_path"])

        # Normalize image to [0,1], mask to {0,1}
        img = img / 255.0
        mask = (mask > 0).astype(np.float32)

        # Add channel dimension
        img = np.expand_dims(img, axis=0)
        mask = np.expand_dims(mask, axis=0)

        sample = {"image": img, "mask": mask}

        if self.transform is not None:
            sample = self.transform(sample)

        # Convert to torch tensors
        sample["image"] = torch.from_numpy(sample["image"])
        sample["mask"] = torch.from_numpy(sample["mask"])
        return sample
