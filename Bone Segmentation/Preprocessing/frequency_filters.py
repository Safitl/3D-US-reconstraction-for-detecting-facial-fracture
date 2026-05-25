"""
frequency_filters.py — FFT-based spatial frequency filters for ultrasound preprocessing.

Default parameters are derived from Phase 1 Fourier analysis (fourier_analysis.ipynb,
Patient 1 data). All public functions accept uint8 [0,255] or float32 [0,1] grayscale
images and return the same dtype/range as the input.

Public API
----------
gaussian_lowpass(img, cutoff_freq)
gaussian_highpass(img, cutoff_freq)
gaussian_bandpass(img, low_freq, high_freq)
directional_notch(img, angle_deg, half_width_deg, r_min_frac)
apply_filter(img, method, **kwargs)          — unified entry point
visualize_filter_masks(shape, save_path)     — diagnostic
"""

import numpy as np

# ── Phase 1 parameter defaults ─────────────────────────────────────────────────
# Derived from fourier_analysis.ipynb on Patient 1 labeled frames.
# BP_LOW_FREQ is adjusted from the raw 0.005 estimate to exclude tissue-scale
# gradients that were captured by the patch PSD lower bound.

LP_CUTOFF_FREQ     = 0.1663   # cyc/px  (period ~  6 px) — speckle boundary
HP_CUTOFF_FREQ     = 0.1663   # cyc/px  same boundary, high-pass side
BP_LOW_FREQ        = 0.0150   # cyc/px  (period ~ 67 px) — bone band lower
BP_HIGH_FREQ       = 0.0975   # cyc/px  (period ~ 10 px) — bone band upper
BP_PEAK_FREQ       = 0.0375   # cyc/px  (period ~ 27 px) — bone peak
DOMINANT_ANGLE_DEG = 93.0     # degrees — dominant scan-line artifact direction
NOTCH_HALF_WIDTH   = 10.0     # degrees — half-width of the angular notch


# ── Internal helpers ───────────────────────────────────────────────────────────

def _radial_map(shape: tuple):
    """Return (R, FX, FY) radial and component frequency maps, DC at centre."""
    H, W = shape
    fy = np.fft.fftshift(np.fft.fftfreq(H)).astype(np.float32)
    fx = np.fft.fftshift(np.fft.fftfreq(W)).astype(np.float32)
    FX, FY = np.meshgrid(fx, fy)
    return np.sqrt(FX ** 2 + FY ** 2), FX, FY


def _angular_distance(theta: np.ndarray, ref_deg: float) -> np.ndarray:
    """Minimum angular distance (degrees) between theta and ref_deg, range [0, 180]."""
    diff = np.abs(theta - ref_deg) % 360.0
    return np.minimum(diff, 360.0 - diff)


def _as_float32(img: np.ndarray) -> np.ndarray:
    """Convert any grayscale image to float32 [0, 1]."""
    arr = np.asarray(img, dtype=np.float64)
    if arr.max() > 1.0 + 1e-6:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0).astype(np.float32)


def _restore_dtype(result_f32: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Clip float32 result to [0,1] and convert back to the original dtype."""
    result_f32 = np.clip(result_f32, 0.0, 1.0).astype(np.float32)
    if np.asarray(original).dtype == np.uint8:
        return (result_f32 * 255).astype(np.uint8)
    return result_f32


def _fft_apply(img_f32: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Multiply the centred FFT of img_f32 by mask H, invert, and return float32.
    Output is min-max normalised to [0, 1] so all filter types have a consistent
    output range regardless of whether they are LP, HP, or BP.
    """
    F_shift  = np.fft.fftshift(np.fft.fft2(img_f32))
    result   = np.real(
        np.fft.ifft2(np.fft.ifftshift(F_shift * H))
    ).astype(np.float32)
    lo, hi = result.min(), result.max()
    if hi > lo:
        result = (result - lo) / (hi - lo)
    return result


# ── Public filter functions ────────────────────────────────────────────────────

def gaussian_lowpass(img: np.ndarray,
                     cutoff_freq: float = LP_CUTOFF_FREQ) -> np.ndarray:
    """
    Gaussian low-pass filter.

    Passes spatial frequencies below cutoff_freq and attenuates speckle noise
    at higher frequencies while preserving broad tissue structure and bone ridges.

    Parameters
    ----------
    img          : grayscale image, uint8 [0,255] or float32 [0,1]
    cutoff_freq  : 1/e Gaussian radius in frequency space (cycles/pixel).
                   Default from Phase 1: 0.1663 cyc/px (~6 px period).

    Returns
    -------
    Filtered image in the same dtype/range as input.
    """
    img_f = _as_float32(img)
    R, _, _ = _radial_map(img_f.shape)
    H = np.exp(-(R / cutoff_freq) ** 2).astype(np.float32)
    return _restore_dtype(_fft_apply(img_f, H), img)


def gaussian_highpass(img: np.ndarray,
                      cutoff_freq: float = HP_CUTOFF_FREQ) -> np.ndarray:
    """
    Gaussian high-pass filter.

    Complement of the Gaussian LP: H_hp = 1 − exp(−(r/σ)²).
    Enhances fine bone-ridge edges and surface discontinuities.
    Bipolar output is normalised to [0, 1] (neutral edge ≈ 0.5).

    Parameters
    ----------
    img          : grayscale image
    cutoff_freq  : 1/e Gaussian radius; frequencies above this pass through.
                   Default: 0.1663 cyc/px.
    """
    img_f = _as_float32(img)
    R, _, _ = _radial_map(img_f.shape)
    H = (1.0 - np.exp(-(R / cutoff_freq) ** 2)).astype(np.float32)
    return _restore_dtype(_fft_apply(img_f, H), img)


def gaussian_bandpass(img: np.ndarray,
                      low_freq:  float = BP_LOW_FREQ,
                      high_freq: float = BP_HIGH_FREQ) -> np.ndarray:
    """
    Gaussian band-pass filter.

    Isolates the spatial frequency band [low_freq, high_freq] (cycles/pixel).
    Default bounds target the bone-ridge dominant frequency range derived from
    Phase 1 patch-based PSD analysis (~10–67 px period).

    Implemented as the product of a high-pass Gaussian (at low_freq) and a
    low-pass Gaussian (at high_freq).

    Parameters
    ----------
    img       : grayscale image
    low_freq  : lower cutoff — 1/e radius of the HP component (cycles/pixel).
                Default: 0.0150 cyc/px (~67 px period).
    high_freq : upper cutoff — 1/e radius of the LP component (cycles/pixel).
                Default: 0.0975 cyc/px (~10 px period).
    """
    if low_freq >= high_freq:
        raise ValueError(
            f"low_freq ({low_freq:.4f}) must be less than high_freq ({high_freq:.4f})"
        )
    img_f = _as_float32(img)
    R, _, _ = _radial_map(img_f.shape)
    H_hp = 1.0 - np.exp(-(R / low_freq)  ** 2)
    H_lp = np.exp(-(R / high_freq) ** 2)
    H    = (H_hp * H_lp).astype(np.float32)
    return _restore_dtype(_fft_apply(img_f, H), img)


def directional_notch(img: np.ndarray,
                      angle_deg:      float = DOMINANT_ANGLE_DEG,
                      half_width_deg: float = NOTCH_HALF_WIDTH,
                      r_min_frac:     float = 0.02) -> np.ndarray:
    """
    Directional notch filter.

    Zeros spectral energy at a specific orientation to remove scan-line artifacts.
    Phase 1 analysis identified ~93° (near-horizontal) as the dominant artifact
    direction for Patient 1.

    The notch is applied symmetrically at angle_deg and angle_deg ± 180° to
    respect FFT conjugate symmetry. Frequencies within r_min_frac × max(H,W)
    of DC are left untouched to prevent global brightness shifts.

    Parameters
    ----------
    img            : grayscale image
    angle_deg      : centre of the suppressed angular wedge (degrees, −180 to 180).
                     Default: 93.0°.
    half_width_deg : half-width of the wedge in degrees. Default: 10.0°.
    r_min_frac     : DC exclusion radius as a fraction of max(H, W). Default: 0.02.
    """
    img_f = _as_float32(img)
    R, FX, FY = _radial_map(img_f.shape)
    theta = np.degrees(np.arctan2(FY, FX))

    dc_ring = (R * max(img_f.shape)) <= (r_min_frac * max(img_f.shape))
    notch   = (
        (_angular_distance(theta, angle_deg)         < half_width_deg) |
        (_angular_distance(theta, angle_deg + 180.0) < half_width_deg)
    ) & ~dc_ring

    H = np.where(notch, 0.0, 1.0).astype(np.float32)
    return _restore_dtype(_fft_apply(img_f, H), img)


# ── Diagnostic visualisation ───────────────────────────────────────────────────

def visualize_filter_masks(shape: tuple = (600, 600),
                           save_path: str = None) -> None:
    """
    Plot all four filter masks side-by-side using the default Phase 1 parameters.
    Useful for sanity-checking filter shapes before applying to real images.

    Parameters
    ----------
    shape     : image shape (H, W) to build the masks for. Default: (600, 600).
    save_path : if given, save the figure to this path instead of displaying.
    """
    import matplotlib.pyplot as plt

    R, FX, FY = _radial_map(shape)
    theta = np.degrees(np.arctan2(FY, FX))

    H_lp = np.exp(-(R / LP_CUTOFF_FREQ) ** 2)
    H_hp = 1.0 - np.exp(-(R / HP_CUTOFF_FREQ) ** 2)
    H_bp = (
        (1.0 - np.exp(-(R / BP_LOW_FREQ) ** 2)) *
        np.exp(-(R / BP_HIGH_FREQ) ** 2)
    )
    dc_ring   = (R * max(shape)) <= (0.02 * max(shape))
    notch_pts = (
        (_angular_distance(theta, DOMINANT_ANGLE_DEG)         < NOTCH_HALF_WIDTH) |
        (_angular_distance(theta, DOMINANT_ANGLE_DEG + 180.0) < NOTCH_HALF_WIDTH)
    ) & ~dc_ring
    H_notch = np.where(notch_pts, 0.0, 1.0).astype(np.float32)

    labels = [
        f'Low-pass  (σ = {LP_CUTOFF_FREQ:.3f} cyc/px)',
        f'High-pass (σ = {HP_CUTOFF_FREQ:.3f} cyc/px)',
        f'Band-pass ({BP_LOW_FREQ:.3f} – {BP_HIGH_FREQ:.3f} cyc/px)',
        f'Notch  ({DOMINANT_ANGLE_DEG:.0f}° ± {NOTCH_HALF_WIDTH:.0f}°)',
    ]
    masks = [H_lp, H_hp, H_bp, H_notch]
    ext   = [-0.5, 0.5, 0.5, -0.5]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, label, mask in zip(axes, labels, masks):
        im = ax.imshow(mask, cmap='gray', vmin=0, vmax=1, extent=ext, aspect='auto')
        ax.set_title(label, fontsize=9)
        ax.set_xlabel('fx (cyc/px)', fontsize=8)
        ax.set_ylabel('fy (cyc/px)', fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle('Frequency-domain filter masks  (DC at centre)', fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'Saved to {save_path}')
    else:
        plt.show()


# ── Unified entry point ────────────────────────────────────────────────────────

_VALID_METHODS = ('fft_lowpass', 'fft_highpass', 'fft_bandpass', 'fft_directional')


def apply_filter(img: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Unified entry point for all FFT-based filters.

    Parameters
    ----------
    img    : grayscale image (uint8 or float32)
    method : one of 'fft_lowpass' | 'fft_highpass' | 'fft_bandpass' | 'fft_directional'
    kwargs : override any default parameter for the selected filter

              fft_lowpass    : cutoff_freq
              fft_highpass   : cutoff_freq
              fft_bandpass   : low_freq, high_freq
              fft_directional: angle_deg, half_width_deg, r_min_frac

    Returns
    -------
    Filtered image in the same dtype/range as input.

    Examples
    --------
    >>> out = apply_filter(img, 'fft_lowpass')
    >>> out = apply_filter(img, 'fft_bandpass', low_freq=0.02, high_freq=0.10)
    >>> out = apply_filter(img, 'fft_directional', angle_deg=90.0, half_width_deg=15.0)
    """
    if method == 'fft_lowpass':
        keys = ('cutoff_freq',)
        return gaussian_lowpass(img, **{k: kwargs[k] for k in keys if k in kwargs})
    if method == 'fft_highpass':
        keys = ('cutoff_freq',)
        return gaussian_highpass(img, **{k: kwargs[k] for k in keys if k in kwargs})
    if method == 'fft_bandpass':
        keys = ('low_freq', 'high_freq')
        return gaussian_bandpass(img, **{k: kwargs[k] for k in keys if k in kwargs})
    if method == 'fft_directional':
        keys = ('angle_deg', 'half_width_deg', 'r_min_frac')
        return directional_notch(img, **{k: kwargs[k] for k in keys if k in kwargs})
    raise ValueError(
        f"Unknown method '{method}'. Valid options: {_VALID_METHODS}"
    )
