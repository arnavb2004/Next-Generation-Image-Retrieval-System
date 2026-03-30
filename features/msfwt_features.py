"""
Task 4 — Novel Feature: MSFWT
Multi-Scale Frequency-Weighted Texture Descriptor

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Motivation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Existing approaches capture partial views of image content:
  • LBP   — local texture micro-patterns (pixel-level)
  • HOG   — oriented edges (gradient-domain, spatial)
  • PCA   — global linear variance (appearance modes)

MSFWT exploits the observation that discriminative texture information lives
at *specific frequency bands that vary per region*.  Adding a constant to all
pixels shifts only the DC coefficient in the DCT — frequency-domain statistics
are therefore intrinsically robust to uniform illumination changes that break
LBP and raw colour features.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mathematical Formulation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Let I be a grayscale image of size H × W, preprocessed with CLAHE.

Step 1 — Block DCT energy
  Partition I into non-overlapping b×b blocks B_{i,j}.
      D_{i,j}       = DCT2(B_{i,j})          (2-D type-II DCT, ortho norm)
      E_{i,j}(u,v)  = D_{i,j}(u,v)²          (per-frequency energy)

Step 2 — Adaptive frequency-saliency weighting
  Cross-block variance for each frequency position (u, v):
      σ²(u,v) = Var_{i,j}[ E_{i,j}(u,v) ]

  High variance ⟹ the coefficient is discriminative across regions.
  Normalised saliency weight:
      w(u,v) = σ²(u,v) / ( Σ_{u,v} σ²(u,v) + ε )

  Weighted scalar energy per block:
      F_{i,j} = Σ_{u,v}  w(u,v) · E_{i,j}(u,v)

  Result: frequency saliency map  S ∈ ℝ^{(H/b)×(W/b)}

Step 3 — Gradient co-occurrence histogram on S
  Compute spatial gradients of the saliency map S:
      G_x = ∂S/∂x,   G_y = ∂S/∂y
      Mag = √(G_x² + G_y²),   Ori = arctan2(G_y, G_x) mod π

  Build a joint (magnitude, orientation) 2-D histogram:
      H_co[m_bin, o_bin] += 1

  This co-occurrence encodes spatial transitions in frequency-saliency —
  a feature orthogonal to both LBP (raw pixel transitions) and HOG
  (spatial-domain gradient histogram).

Step 4 — Multi-scale concatenation
  Repeat Steps 1–3 at block sizes b ∈ {4, 8, 16}.
  L2-normalise the concatenated histogram:
      f_MSFWT = L2_norm( [H_co^{b=4} ‖ H_co^{b=8} ‖ H_co^{b=16}] )
  Dimension = 3 × n_mag_bins × n_ori_bins   (default 3×8×8 = 192)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Originality
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. The adaptive saliency weighting (Step 2) is per-image and requires no
     labelled training data — unlike CNN filters or supervised embeddings.
  2. Applying co-occurrence analysis (Step 3) on a frequency-derived surface
     rather than on raw intensities (GLCM) or spatial gradients (HOG) is
     novel: it fuses frequency content and spatial structure in one step.
  3. CLAHE preprocessing inside the descriptor computation (not as an
     external pipeline stage) makes the feature self-contained and
     illumination-normalised by construction.
"""

import numpy as np
import cv2
from scipy.fft import dctn


def _block_dct_energy(gray: np.ndarray, b: int) -> np.ndarray:
    """Return (H//b, W//b, b, b) block-wise DCT energy tensor."""
    H, W = gray.shape
    bH, bW = H // b, W // b
    E = np.zeros((bH, bW, b, b), dtype=np.float64)
    for i in range(bH):
        for j in range(bW):
            patch      = gray[i*b:(i+1)*b, j*b:(j+1)*b].astype(np.float64)
            D          = dctn(patch, norm="ortho")
            E[i, j]    = D ** 2
    return E


def _saliency_map(E: np.ndarray) -> np.ndarray:
    """
    Compute per-block scalar saliency F_{i,j} from energy tensor E.
    Shape: (bH, bW).
    """
    freq_var = E.var(axis=(0, 1))                      # (b, b)
    w        = freq_var / (freq_var.sum() + 1e-12)     # normalised weights
    S        = np.sum(E * w[None, None], axis=(2, 3))  # (bH, bW)
    return S.astype(np.float32)


def _cooccurrence_hist(S: np.ndarray,
                        n_mag: int = 8, n_ori: int = 8) -> np.ndarray:
    """Joint (magnitude, orientation) histogram of S's spatial gradients."""
    if S.shape[0] < 2 or S.shape[1] < 2:
        return np.zeros(n_mag * n_ori, dtype=np.float32)

    gx  = cv2.Sobel(S, cv2.CV_32F, 1, 0, ksize=1)
    gy  = cv2.Sobel(S, cv2.CV_32F, 0, 1, ksize=1)
    mag = np.sqrt(gx**2 + gy**2)
    ori = np.degrees(np.arctan2(gy, gx)) % 180

    mag_max  = mag.max() + 1e-8
    m_bins   = np.clip((mag / mag_max * n_mag).astype(int), 0, n_mag - 1)
    o_bins   = np.clip((ori / 180.0 * n_ori).astype(int), 0, n_ori - 1)

    hist = np.zeros((n_mag, n_ori), dtype=np.float32)
    np.add.at(hist, (m_bins.ravel(), o_bins.ravel()), 1)
    total = hist.sum() + 1e-8
    return (hist / total).ravel()


def compute_msfwt(image: np.ndarray,
                   block_sizes: list = [4, 8, 16],
                   n_mag: int = 8,
                   n_ori: int = 8,
                   target: tuple = (64, 64)) -> np.ndarray:
    """
    Compute the MSFWT descriptor for a single image.

    Parameters
    ----------
    image       : BGR or grayscale uint8
    block_sizes : block widths b for multi-scale analysis
    n_mag       : magnitude histogram bins
    n_ori       : orientation histogram bins
    target      : image is resized to this before extraction

    Returns
    -------
    descriptor  : float32, L2-normalised, length = 3 * n_mag * n_ori (= 192)
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = cv2.resize(gray, target)
    # CLAHE for intrinsic illumination normalisation
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gray  = clahe.apply(gray).astype(np.float32) / 255.0

    hists = []
    for b in block_sizes:
        H, W   = gray.shape
        Hc, Wc = (H // b) * b, (W // b) * b
        E      = _block_dct_energy(gray[:Hc, :Wc], b)
        S      = _saliency_map(E)
        hists.append(_cooccurrence_hist(S, n_mag, n_ori))

    feat = np.concatenate(hists).astype(np.float32)
    return feat / (np.linalg.norm(feat) + 1e-8)


def extract_msfwt_batch(images: list, **kw) -> np.ndarray:
    """Extract MSFWT for a list of images; returns (N, D) float32 array."""
    return np.array([compute_msfwt(img, **kw) for img in images],
                    dtype=np.float32)
