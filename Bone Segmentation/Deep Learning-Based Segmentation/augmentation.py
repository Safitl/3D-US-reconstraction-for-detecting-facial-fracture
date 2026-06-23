"""
Augmentation transforms for ultrasound bone segmentation.

Each transform receives and returns a dict {"image": ndarray, "mask": ndarray}
where both are float32 arrays with a channel dimension (1, H, W).
Image values are in [0, 1]; mask values are in {0, 1}.
"""

import random

import cv2
import numpy as np


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample["image"] = sample["image"][:, :, ::-1].copy()
            sample["mask"]  = sample["mask"][:, :, ::-1].copy()
        return sample


class RandomVerticalFlip:
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample["image"] = sample["image"][:, ::-1, :].copy()
            sample["mask"]  = sample["mask"][:, ::-1, :].copy()
        return sample


class RandomRotation:
    """Rotate image and mask by a random angle in [-max_angle, +max_angle] degrees."""
    def __init__(self, max_angle=15, p=0.5):
        self.max_angle = max_angle
        self.p = p

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        angle = random.uniform(-self.max_angle, self.max_angle)
        img  = sample["image"][0]
        mask = sample["mask"][0]
        h, w = img.shape
        M    = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        img  = cv2.warpAffine(img,  M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_REFLECT)
        sample["image"] = img[np.newaxis]
        sample["mask"]  = (mask > 0.5).astype(np.float32)[np.newaxis]
        return sample


class RandomBrightnessContrast:
    """Randomly adjust brightness and contrast of the image only."""
    def __init__(self, brightness=0.2, contrast=0.2, p=0.5):
        self.brightness = brightness
        self.contrast   = contrast
        self.p          = p

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        alpha = 1.0 + random.uniform(-self.contrast,   self.contrast)
        beta  =       random.uniform(-self.brightness,  self.brightness)
        img   = sample["image"] * alpha + beta
        sample["image"] = np.clip(img, 0.0, 1.0).astype(np.float32)
        return sample


class GaussianNoise:
    """Add Gaussian noise to the image only."""
    def __init__(self, std=0.02, p=0.3):
        self.std = std
        self.p   = p

    def __call__(self, sample):
        if random.random() >= self.p:
            return sample
        noise = np.random.normal(0, self.std, sample["image"].shape).astype(np.float32)
        sample["image"] = np.clip(sample["image"] + noise, 0.0, 1.0)
        return sample


def build_train_augmentation(cfg: dict):
    """
    Build augmentation pipeline from config dict.
    Expected keys (all optional, with defaults):
        horizontal_flip_p, vertical_flip_p, rotation_max_angle, rotation_p,
        brightness, contrast, brightness_contrast_p, noise_std, noise_p
    """
    transforms = []

    p_hflip = cfg.get("horizontal_flip_p", 0.5)
    if p_hflip > 0:
        transforms.append(RandomHorizontalFlip(p=p_hflip))

    p_vflip = cfg.get("vertical_flip_p", 0.0)
    if p_vflip > 0:
        transforms.append(RandomVerticalFlip(p=p_vflip))

    rot_p = cfg.get("rotation_p", 0.5)
    rot_a = cfg.get("rotation_max_angle", 15)
    if rot_p > 0 and rot_a > 0:
        transforms.append(RandomRotation(max_angle=rot_a, p=rot_p))

    bc_p  = cfg.get("brightness_contrast_p", 0.5)
    bri   = cfg.get("brightness", 0.2)
    con   = cfg.get("contrast", 0.2)
    if bc_p > 0:
        transforms.append(RandomBrightnessContrast(brightness=bri, contrast=con, p=bc_p))

    noise_p   = cfg.get("noise_p", 0.3)
    noise_std = cfg.get("noise_std", 0.02)
    if noise_p > 0:
        transforms.append(GaussianNoise(std=noise_std, p=noise_p))

    return Compose(transforms) if transforms else None
