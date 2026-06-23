"""
preprocessing_api.py — Unified preprocessing entry point.

Wraps all four preprocessing modules (FFT filters, SVD/PCA denoising, wavelet
denoising, and the existing CLAHE+Gaussian baseline) behind a single
`preprocess(img, method, **kwargs)` call.

All methods:
  • Accept a uint8 [0,255] or float32 [0,1] grayscale image
  • Return the same dtype/range as input
  • Are identified by a short string method name

Available methods
-----------------
  baseline            CLAHE (clip=0.01) + Gaussian blur 7×7 — existing pipeline
  fft_lowpass         Gaussian low-pass (σ = 0.1663 cyc/px)
  fft_highpass        Gaussian high-pass (σ = 0.1663 cyc/px)
  fft_bandpass        Gaussian band-pass (0.015 – 0.0975 cyc/px)
  fft_directional     Directional notch filter (93°, ±10°)
  svd_global          Global rank-k SVD (rank=30)
  svd_patch           Overlapping patch PCA (k=6, patch=16, stride=8)
  wavelet_visu_soft   VisuShrink + soft thresholding (db6, 4 levels)
  wavelet_visu_hard   VisuShrink + hard thresholding
  wavelet_bayes_soft  BayesShrink + soft thresholding  (default recommended)
  wavelet_bayes_hard  BayesShrink + hard thresholding

Public API
----------
preprocess(img, method, **kwargs)   → preprocessed image (same dtype as input)
list_methods()                      → list of all method name strings
"""

import sys
from pathlib import Path

import cv2
import numpy as np
from skimage.exposure import equalize_adapthist

# ── Locate sibling modules regardless of working directory ──────────────────
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from frequency_filters import apply_filter
from svd_denoising import apply_svd_filter
from wavelet_denoising import apply_wavelet_filter

# ── Method registry ─────────────────────────────────────────────────────────
_FFT_METHODS      = ('fft_lowpass', 'fft_highpass', 'fft_bandpass', 'fft_directional')
_SVD_METHODS      = (
    'svd_global',
    'svd_patch',
    'patch_svd_square_32x32',
    'patch_svd_square_48x48',
    'patch_svd_horizontal_16x64',
    'patch_svd_horizontal_24x96',
    'patch_svd_horizontal_32x96',
    'patch_svd_vertical_64x16',
)
_WAVELET_METHODS  = ('wavelet_visu_soft', 'wavelet_visu_hard',
                     'wavelet_bayes_soft', 'wavelet_bayes_hard')

METHODS = (
    ('baseline',) + _FFT_METHODS + _SVD_METHODS + _WAVELET_METHODS
)


# ── Baseline: CLAHE + Gaussian (existing pipeline default) ──────────────────

def _baseline(img: np.ndarray,
              clahe_clip_limit: float = 0.01,
              gaussian_kernel: tuple  = (7, 7)) -> np.ndarray:
    """
    Replicate the preprocessing used in ultrasound_bone_segmentation_cli.py:
    adaptive histogram equalization (CLAHE) followed by Gaussian blur.
    Always returns uint8.
    """
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        img_norm = arr / 255.0
    else:
        img_norm = np.clip(arr, 0.0, 1.0).astype(np.float64)

    clahe = equalize_adapthist(img_norm, clip_limit=clahe_clip_limit)
    clahe_8bit = (clahe * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(clahe_8bit, gaussian_kernel, 0)

    if np.asarray(img).dtype != np.uint8:
        return (blurred / 255.0).astype(np.float32)
    return blurred


# ── Unified entry point ─────────────────────────────────────────────────────

def preprocess(img: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Apply a named preprocessing method to a grayscale ultrasound frame.

    Parameters
    ----------
    img    : grayscale image, uint8 [0,255] or float32 [0,1]
    method : preprocessing method name — see module docstring for the full list
    kwargs : optional parameter overrides passed directly to the underlying
             function (e.g. rank=20, cutoff_freq=0.15, level=3)

    Returns
    -------
    Preprocessed image in the same dtype/range as input.

    Examples
    --------
    >>> out = preprocess(frame, 'baseline')
    >>> out = preprocess(frame, 'fft_bandpass', low_freq=0.02, high_freq=0.10)
    >>> out = preprocess(frame, 'svd_global', rank=20)
    >>> out = preprocess(frame, 'wavelet_bayes_soft', level=3)
    """
    if method == 'baseline':
        return _baseline(img, **{k: kwargs[k]
                                 for k in ('clahe_clip_limit', 'gaussian_kernel')
                                 if k in kwargs})

    if method in _FFT_METHODS:
        return apply_filter(img, method, **kwargs)

    if method in _SVD_METHODS:
        return apply_svd_filter(img, method, **kwargs)

    if method in _WAVELET_METHODS:
        return apply_wavelet_filter(img, method, **kwargs)

    raise ValueError(
        f"Unknown method '{method}'. "
        f"Use list_methods() or call with one of:\n  {list(METHODS)}"
    )


def list_methods() -> list:
    """Return a list of all available preprocessing method names."""
    return list(METHODS)
