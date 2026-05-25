"""
wavelet_denoising.py — Wavelet-based denoising for ultrasound preprocessing.

Multi-level 2D DWT denoising with soft or hard thresholding of detail
sub-bands. Two threshold estimators are provided:

  VisuShrink  : universal threshold T = σ · √(2 log N).  One threshold
                per level, estimated from the HH (diagonal) sub-band.
                Conservative — good starting point for noisy US frames.

  BayesShrink : per-sub-band Bayes-optimal threshold T = σ_n² / σ_s,
                where σ_n is the noise std and σ_s is the estimated
                signal std in each sub-band.  More adaptive than
                VisuShrink; better preserves fine bone-ridge detail.

Wavelet bases
-------------
  db6  — Daubechies 6-tap: good energy compaction, moderate smoothing.
         Default and recommended for ultrasound bone segmentation.
  sym8 — Symmlet 8-tap: near-symmetric, less ringing than Daubechies.
         Use when db6 introduces visible edge artefacts.
  Any PyWavelets-supported 2D wavelet name is accepted.

Public API
----------
wavelet_denoise(img, wavelet, level, threshold_method, threshold_mode, ...)
estimate_noise_std(sub_band)                — MAD-based noise estimate
apply_wavelet_filter(img, method, **kwargs) — unified entry point
"""

import numpy as np

try:
    import pywt
except ImportError as _e:
    raise ImportError(
        "PyWavelets is required for wavelet_denoising.py.  "
        "Install it with:  pip install PyWavelets"
    ) from _e

# ── Default parameters ──────────────────────────────────────────────────────
WAVELET_DEFAULT        = 'db6'    # basis wavelet
LEVEL_DEFAULT          = 4        # decomposition levels
THRESHOLD_METHOD       = 'bayes'  # 'visu' | 'bayes'
THRESHOLD_MODE         = 'soft'   # 'soft' | 'hard'
SIGMA_SCALE            = 1.0      # multiplier on the computed threshold


# ── Internal helpers ────────────────────────────────────────────────────────

def _as_float32(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float64)
    if arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _restore_dtype(result_f32: np.ndarray, original: np.ndarray) -> np.ndarray:
    result_f32 = np.clip(result_f32, 0.0, 1.0).astype(np.float32)
    if np.asarray(original).dtype == np.uint8:
        return (result_f32 * 255).astype(np.uint8)
    return result_f32


def estimate_noise_std(sub_band: np.ndarray) -> float:
    """
    Robust noise standard deviation via the Median Absolute Deviation (MAD)
    estimator applied to a wavelet detail sub-band.

    σ̂ = median(|coefficients|) / 0.6745

    This is the standard estimator for Gaussian noise in wavelet denoising
    (Donoho & Johnstone 1994).  It is applied to the finest-scale diagonal
    (HH) sub-band where signal energy is lowest and the coefficient
    distribution is closest to Gaussian noise.

    Parameters
    ----------
    sub_band : 2-D array of wavelet detail coefficients

    Returns
    -------
    Estimated noise standard deviation (float, ≥ 0).
    """
    return float(np.median(np.abs(sub_band)) / 0.6745)


def _visu_threshold(coeffs_flat: np.ndarray, sigma: float) -> float:
    """Universal (VisuShrink) threshold: T = σ √(2 log N)."""
    N = coeffs_flat.size
    if N == 0 or sigma == 0.0:
        return 0.0
    return float(sigma * np.sqrt(2.0 * np.log(max(N, 2))))


def _bayes_threshold(sub_band: np.ndarray, sigma_n: float) -> float:
    """
    BayesShrink threshold: T = σ_n² / σ_s, where σ_s is estimated as
    sqrt(max(var(sub_band) − σ_n², 0)).
    """
    if sigma_n == 0.0:
        return 0.0
    var_y = float(np.var(sub_band))
    sigma_s = np.sqrt(max(var_y - sigma_n ** 2, 0.0))
    if sigma_s == 0.0:
        return float(np.abs(sub_band).max())   # threshold everything
    return float(sigma_n ** 2 / sigma_s)


def _apply_threshold(coeff: np.ndarray, threshold: float,
                     mode: str) -> np.ndarray:
    """Apply soft or hard thresholding to a coefficient array."""
    if mode == 'soft':
        return pywt.threshold(coeff, threshold, mode='soft')
    if mode == 'hard':
        return pywt.threshold(coeff, threshold, mode='hard')
    raise ValueError(f"Unknown threshold mode '{mode}'. Use 'soft' or 'hard'.")


# ── Main denoising function ─────────────────────────────────────────────────

def wavelet_denoise(img: np.ndarray,
                    wavelet:          str   = WAVELET_DEFAULT,
                    level:            int   = LEVEL_DEFAULT,
                    threshold_method: str   = THRESHOLD_METHOD,
                    threshold_mode:   str   = THRESHOLD_MODE,
                    sigma_scale:      float = SIGMA_SCALE) -> np.ndarray:
    """
    Multi-level 2D DWT denoising with adaptive thresholding.

    Algorithm
    ---------
    1. Decompose the image using a `level`-level 2D DWT.
    2. Estimate σ (noise std) from the finest-scale HH diagonal sub-band
       using the MAD estimator.
    3. For each detail sub-band at every level, compute a threshold using
       either VisuShrink or BayesShrink.
    4. Apply soft or hard thresholding to zero out or shrink small
       coefficients (assumed to be noise).
    5. Reconstruct with IDWT; clip to [0, 1].

    The approximation (LL) sub-band is never thresholded — it holds the
    low-frequency structure (bone ridges, tissue boundaries) that we want
    to preserve.

    Parameters
    ----------
    img              : grayscale image, uint8 [0,255] or float32 [0,1]
    wavelet          : PyWavelets wavelet name. Default: 'db6'.
                       Try 'sym8' if db6 introduces visible ringing.
    level            : decomposition depth. Default: 4.
                       For 600×600 images, level 4 gives the finest
                       sub-band at ~38 px — comfortably above the
                       bone-ridge scale (~10–30 px) and below tissue.
    threshold_method : 'visu' (universal) | 'bayes' (per-sub-band).
                       Default: 'bayes'.
    threshold_mode   : 'soft' | 'hard'. Default: 'soft'.
                       Soft thresholding produces smoother edges; hard
                       preserves coefficient magnitudes above the threshold.
    sigma_scale      : multiplicative scale on the computed threshold.
                       Default: 1.0.  Increase to denoise more aggressively;
                       decrease to preserve more fine detail.

    Returns
    -------
    Denoised image in the same dtype/range as input.
    """
    img_f = _as_float32(img)

    max_level = pywt.dwt_max_level(min(img_f.shape), wavelet)
    level = min(level, max_level)

    coeffs = pywt.wavedec2(img_f, wavelet=wavelet, level=level)

    # Estimate noise from finest-scale HH sub-band (coeffs[1][2])
    sigma_n = estimate_noise_std(coeffs[1][2])

    denoised_coeffs = [coeffs[0]]  # keep LL approximation unchanged

    for detail_tuple in coeffs[1:]:
        thresholded = []
        for sub_band in detail_tuple:   # LH, HL, HH at this level
            if threshold_method == 'visu':
                T = _visu_threshold(sub_band.ravel(), sigma_n)
            elif threshold_method == 'bayes':
                T = _bayes_threshold(sub_band, sigma_n)
            else:
                raise ValueError(
                    f"Unknown threshold_method '{threshold_method}'. "
                    "Use 'visu' or 'bayes'."
                )
            T *= sigma_scale
            thresholded.append(_apply_threshold(sub_band, T, threshold_mode))
        denoised_coeffs.append(tuple(thresholded))

    result = pywt.waverec2(denoised_coeffs, wavelet=wavelet)

    # waverec2 may add a row/col if the original shape was odd
    result = result[:img_f.shape[0], :img_f.shape[1]]
    result = result.astype(np.float32)
    return _restore_dtype(np.clip(result, 0.0, 1.0), img)


# ── Unified entry point ─────────────────────────────────────────────────────

_VALID_METHODS = ('wavelet_visu_soft', 'wavelet_visu_hard',
                  'wavelet_bayes_soft', 'wavelet_bayes_hard')


def apply_wavelet_filter(img: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Unified entry point for wavelet denoising.

    Parameters
    ----------
    img    : grayscale image (uint8 or float32)
    method : one of
               'wavelet_visu_soft'  — VisuShrink + soft thresholding
               'wavelet_visu_hard'  — VisuShrink + hard thresholding
               'wavelet_bayes_soft' — BayesShrink + soft thresholding  (default)
               'wavelet_bayes_hard' — BayesShrink + hard thresholding
    kwargs : override any parameter accepted by wavelet_denoise():
               wavelet, level, sigma_scale

    Returns
    -------
    Denoised image in the same dtype/range as input.

    Examples
    --------
    >>> out = apply_wavelet_filter(img, 'wavelet_bayes_soft')
    >>> out = apply_wavelet_filter(img, 'wavelet_visu_soft', wavelet='sym8', level=3)
    >>> out = apply_wavelet_filter(img, 'wavelet_bayes_soft', sigma_scale=1.5)
    """
    method_map = {
        'wavelet_visu_soft':  ('visu',  'soft'),
        'wavelet_visu_hard':  ('visu',  'hard'),
        'wavelet_bayes_soft': ('bayes', 'soft'),
        'wavelet_bayes_hard': ('bayes', 'hard'),
    }
    if method not in method_map:
        raise ValueError(
            f"Unknown method '{method}'. Valid options: {list(method_map)}"
        )
    t_method, t_mode = method_map[method]
    allowed = ('wavelet', 'level', 'sigma_scale')
    extra = {k: kwargs[k] for k in allowed if k in kwargs}
    return wavelet_denoise(img,
                           threshold_method=t_method,
                           threshold_mode=t_mode,
                           **extra)
