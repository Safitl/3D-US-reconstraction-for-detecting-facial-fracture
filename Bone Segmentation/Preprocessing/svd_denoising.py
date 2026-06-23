"""
svd_denoising.py — SVD/PCA-based denoising for ultrasound preprocessing.

Two complementary strategies exploit the fact that bone ridges are spatially
coherent (low-rank) while speckle noise is spatially random (full-rank):

  Global SVD  : rank-k matrix approximation of the full cropped frame.
                Bone structure is captured in the top singular vectors;
                speckle spreads across many and is discarded.

  Patch PCA   : PCA-based denoising of overlapping local patches.
                More spatially local than global SVD — better at preserving
                fine ridge detail while still suppressing diffuse speckle.

Rank / component selection
--------------------------
Both methods require choosing how many singular values / principal components
to keep.  Use the helpers below to choose data-adaptively:

  svd_scree(img)           — plot singular values and cumulative energy;
                             look for the elbow between signal and noise.
  rank_from_energy(img)    — keep enough SVs to explain a target energy fraction.
  rank_from_gap(img)       — keep up to the largest drop in consecutive SVs.

Public API
----------
svd_denoise(img, rank, ...)
pca_patch_denoise(img, n_components, patch_size, stride, ...)
svd_scree(img, n_show, ...)
rank_from_energy(img, energy_frac)
rank_from_gap(img, n_search)
apply_svd_filter(img, method, **kwargs)    — unified entry point
"""

import re as _re
import numpy as np

# ── Default parameters ─────────────────────────────────────────────────────────
SVD_RANK_DEFAULT        = 30    # global SVD: singular vectors to keep
PCA_N_COMPONENTS        = 6     # patch PCA: principal components to keep
PCA_PATCH_SIZE          = 16    # default patch side length (pixels)
ENERGY_FRAC_DEFAULT     = 0.90  # rank_from_energy target

# Pattern that matches all named rectangular-patch variants, e.g.
#   patch_svd_square_32x32 / patch_svd_horizontal_16x64 / patch_svd_vertical_64x16
_PATCH_SVD_RE = _re.compile(
    r'^patch_svd_(?:square|horizontal|vertical)_(\d+)x(\d+)$'
)


# ── Internal helpers ───────────────────────────────────────────────────────────

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


def _extract_patches(img: np.ndarray,
                     patch_h: int, patch_w: int,
                     stride_h: int, stride_w: int):
    """
    Extract all (patch_h × patch_w) patches with the given strides.
    Returns patches as a (N, patch_h*patch_w) matrix and the
    (row, col) top-left corner of each patch.
    Supports rectangular patches for directional denoising experiments.
    """
    H, W = img.shape
    positions, patches = [], []
    for r in range(0, H - patch_h + 1, stride_h):
        for c in range(0, W - patch_w + 1, stride_w):
            positions.append((r, c))
            patches.append(img[r:r + patch_h, c:c + patch_w].ravel())
    return np.array(patches, dtype=np.float32), positions


def _reconstruct_from_patches(patches: np.ndarray, positions: list,
                               img_shape: tuple,
                               patch_h: int, patch_w: int) -> np.ndarray:
    """
    Reconstruct an image from (possibly denoised) patches by averaging
    overlapping regions.  Handles any stride and rectangular shape implicitly.
    """
    H, W = img_shape
    accum  = np.zeros((H, W), dtype=np.float64)
    counts = np.zeros((H, W), dtype=np.float64)
    for patch_vec, (r, c) in zip(patches, positions):
        patch = patch_vec.reshape(patch_h, patch_w)
        accum[r:r + patch_h, c:c + patch_w]  += patch
        counts[r:r + patch_h, c:c + patch_w] += 1.0
    mask   = counts > 0
    result = np.zeros((H, W), dtype=np.float32)
    result[mask] = (accum[mask] / counts[mask]).astype(np.float32)
    return result


# ── Rank / component selection helpers ────────────────────────────────────────

def rank_from_energy(img: np.ndarray,
                     energy_frac: float = ENERGY_FRAC_DEFAULT) -> int:
    """
    Return the minimum rank needed to explain at least energy_frac of the
    total singular value energy (sum of squared singular values).

    This is a conservative rank — it keeps whatever is needed to recover
    energy_frac of total image variance, which typically includes both signal
    and some noise at the boundary.  Use rank_from_gap for a more aggressive
    denoising choice.

    Parameters
    ----------
    img         : grayscale image (any dtype)
    energy_frac : target cumulative energy fraction in [0, 1]. Default 0.90.
    """
    img_f = _as_float32(img)
    s = np.linalg.svd(img_f, compute_uv=False)
    cumulative = np.cumsum(s ** 2) / (s ** 2).sum()
    rank = int(np.searchsorted(cumulative, energy_frac)) + 1
    return min(rank, len(s))


def rank_from_gap(img: np.ndarray, n_search: int = 80) -> int:
    """
    Return the rank at the largest relative drop in consecutive singular values
    within the first n_search values.

    The largest gap marks the transition from structured image content (large SVs)
    to the noise floor (small, slowly decreasing SVs) — i.e. the elbow of the
    scree plot.

    Parameters
    ----------
    img      : grayscale image (any dtype)
    n_search : number of leading singular values to search for the elbow.
               Default 80.  Should be well below min(H, W).
    """
    img_f = _as_float32(img)
    s = np.linalg.svd(img_f, compute_uv=False)
    s_search = s[:n_search]
    # Relative drop: (s[i] - s[i+1]) / s[i]
    rel_drops = (s_search[:-1] - s_search[1:]) / (s_search[:-1] + 1e-10)
    elbow_idx = int(np.argmax(rel_drops))
    return max(elbow_idx + 1, 1)   # keep at least 1 component


# ── Global SVD denoising ───────────────────────────────────────────────────────

def svd_denoise(img: np.ndarray,
                rank: int = SVD_RANK_DEFAULT) -> np.ndarray:
    """
    Global rank-k SVD denoising.

    Decomposes the image matrix as U S Vt and reconstructs from the top-rank
    singular triplets.  The rank-k approximation retains spatially coherent
    structure (bone ridges, tissue boundaries) and discards diffuse speckle,
    which spreads energy across many small singular values.

    Parameters
    ----------
    img  : grayscale image, uint8 [0,255] or float32 [0,1]
    rank : number of singular vectors to keep. Use rank_from_gap() or
           rank_from_energy() to choose a data-adaptive value.
           Default: 30 (a conservative starting point for 600x600 US images).

    Returns
    -------
    Denoised image in the same dtype/range as input.
    """
    img_f = _as_float32(img)
    U, s, Vt = np.linalg.svd(img_f, full_matrices=False)
    rank = min(rank, len(s))
    result = (U[:, :rank] * s[:rank]) @ Vt[:rank, :]
    result = result.astype(np.float32)
    return _restore_dtype(np.clip(result, 0.0, 1.0), img)


# ── Patch PCA denoising ────────────────────────────────────────────────────────

def pca_patch_denoise(img: np.ndarray,
                      n_components: int  = PCA_N_COMPONENTS,
                      patch_h:      int  = PCA_PATCH_SIZE,
                      patch_w:      int  = PCA_PATCH_SIZE,
                      stride_h:     int  = None,
                      stride_w:     int  = None) -> np.ndarray:
    """
    Patch-based PCA denoising with overlapping reconstruction.
    Supports rectangular patches for directional denoising experiments.

    Algorithm
    ---------
    1. Extract all (patch_h × patch_w) patches with the given strides.
    2. Centre each patch by subtracting its mean (remove local DC).
    3. Compute PCA on the patch matrix; project onto the top n_components
       eigenvectors and reconstruct (low-rank approximation in patch space).
    4. Add the mean back to each denoised patch.
    5. Average overlapping patches to reconstruct the full image.

    Parameters
    ----------
    img          : grayscale image, uint8 or float32
    n_components : PCA components to keep per patch. Default: 6.
    patch_h      : patch height in pixels. Default: 16.
    patch_w      : patch width in pixels. Default: 16.
    stride_h     : vertical step. Default: patch_h // 2 (50 % overlap).
    stride_w     : horizontal step. Default: patch_w // 2 (50 % overlap).

    Returns
    -------
    Denoised image in the same dtype/range as input.
    """
    img_f = _as_float32(img)

    if stride_h is None:
        stride_h = max(1, patch_h // 2)
    if stride_w is None:
        stride_w = max(1, patch_w // 2)

    patches, positions = _extract_patches(img_f, patch_h, patch_w,
                                          stride_h, stride_w)

    if len(patches) == 0:
        return _restore_dtype(img_f, img)

    means     = patches.mean(axis=1, keepdims=True)
    patches_c = patches - means

    n_components = min(n_components, patches_c.shape[0], patches_c.shape[1])
    _, s, Vt = np.linalg.svd(patches_c, full_matrices=False)

    components    = Vt[:n_components]
    scores        = patches_c @ components.T
    patches_clean = scores @ components + means

    result = _reconstruct_from_patches(patches_clean, positions,
                                       img_f.shape, patch_h, patch_w)
    return _restore_dtype(np.clip(result, 0.0, 1.0), img)


# ── Diagnostic: scree plot ─────────────────────────────────────────────────────

def svd_scree(img: np.ndarray,
              n_show: int = 100,
              save_path: str = None) -> None:
    """
    Plot the singular value scree curve for the given image.

    Shows:
      Left axis  — singular values (log scale) vs rank index
      Right axis — cumulative energy (fraction of total) vs rank index
      Vertical markers — rank_from_gap and rank_from_energy estimates

    Use this to choose a rank for svd_denoise() or n_components for
    pca_patch_denoise() before applying to the full dataset.

    Parameters
    ----------
    img       : grayscale image (any dtype)
    n_show    : number of leading singular values to display. Default: 100.
    save_path : if given, save figure instead of displaying.
    """
    import matplotlib.pyplot as plt

    img_f = _as_float32(img)
    s = np.linalg.svd(img_f, compute_uv=False)

    n_show    = min(n_show, len(s))
    s_show    = s[:n_show]
    indices   = np.arange(1, n_show + 1)
    cum_energy = np.cumsum(s ** 2) / (s ** 2).sum()

    r_gap    = rank_from_gap(img)
    r_energy = rank_from_energy(img)

    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax2 = ax1.twinx()

    ax1.semilogy(indices, s_show, color='steelblue', lw=2, label='Singular values')
    ax2.plot(indices, cum_energy[:n_show] * 100, color='darkorange',
             lw=1.5, ls='--', label='Cumulative energy (%)')

    ax1.axvline(r_gap,    color='red',   ls='--', lw=1.5,
                label=f'rank_from_gap = {r_gap}')
    ax1.axvline(r_energy, color='green', ls=':',  lw=1.5,
                label=f'rank_from_energy ({ENERGY_FRAC_DEFAULT:.0%}) = {r_energy}')

    ax1.set_xlabel('Rank index')
    ax1.set_ylabel('Singular value (log scale)', color='steelblue')
    ax2.set_ylabel('Cumulative energy (%)', color='darkorange')
    ax1.set_xlim(1, n_show)
    ax2.set_ylim(0, 105)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc='upper right')

    ax1.set_title('SVD scree plot — use the elbow to choose denoising rank')
    ax1.grid(True, which='both', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        print(f'Saved to {save_path}')
    else:
        plt.show()


# ── Unified entry point ────────────────────────────────────────────────────────

_VALID_METHODS = (
    'svd_global',
    'svd_patch',
    'patch_svd_square_HxW',       # e.g. patch_svd_square_32x32
    'patch_svd_horizontal_HxW',   # e.g. patch_svd_horizontal_16x64
    'patch_svd_vertical_HxW',     # e.g. patch_svd_vertical_64x16
)


def apply_svd_filter(img: np.ndarray, method: str, **kwargs) -> np.ndarray:
    """
    Unified entry point for SVD/PCA denoising methods.

    Parameters
    ----------
    img    : grayscale image (uint8 or float32)
    method : one of
               'svd_global'                  — global rank-k SVD
               'svd_patch'                   — default 16×16 patch PCA
               'patch_svd_square_NxN'        — square patch, e.g. 32x32
               'patch_svd_horizontal_HxW'    — wide patch,  e.g. 16x64
               'patch_svd_vertical_HxW'      — tall patch,  e.g. 64x16
    kwargs : override any default parameter
               svd_global              : rank
               svd_patch               : n_components, patch_h, patch_w,
                                         stride_h, stride_w
               patch_svd_*_HxW         : n_components, stride_h, stride_w
                                         (patch dims are parsed from name)

    Returns
    -------
    Denoised image in the same dtype/range as input.

    Examples
    --------
    >>> out = apply_svd_filter(img, 'svd_global', rank=20)
    >>> out = apply_svd_filter(img, 'svd_patch', n_components=4)
    >>> out = apply_svd_filter(img, 'patch_svd_square_32x32')
    >>> out = apply_svd_filter(img, 'patch_svd_horizontal_16x64')
    """
    if method == 'svd_global':
        keys = ('rank',)
        return svd_denoise(img, **{k: kwargs[k] for k in keys if k in kwargs})

    if method == 'svd_patch':
        keys = ('n_components', 'patch_h', 'patch_w', 'stride_h', 'stride_w')
        return pca_patch_denoise(img, **{k: kwargs[k] for k in keys if k in kwargs})

    # Named rectangular-patch variants: patch_svd_{shape}_{H}x{W}
    m = _PATCH_SVD_RE.match(method)
    if m:
        patch_h = int(m.group(1))
        patch_w = int(m.group(2))
        keys = ('n_components', 'stride_h', 'stride_w')
        extra = {k: kwargs[k] for k in keys if k in kwargs}
        return pca_patch_denoise(img, patch_h=patch_h, patch_w=patch_w, **extra)

    raise ValueError(
        f"Unknown method '{method}'. "
        f"Named patch variants follow the pattern: patch_svd_{{square|horizontal|vertical}}_HxW"
    )
