"""
Tasks 2 & 3 — NN, DNN, and CNN-equivalent feature extractors.

NN  : PCA linear auto-encoder
DNN : Stacked Random Fourier Feature encoder + PCA (non-linear, unsupervised)
CNN : HOG descriptor inside Spatial Pyramid Matching (spatial gradient hierarchy)

No PyTorch / TensorFlow required — everything runs on numpy + scipy + opencv.
"""

import numpy as np
from scipy.linalg import svd
import cv2


# ─────────────────────────────────────────────────────────────────────────────
# NN — Linear Auto-Encoder (PCA)
# ─────────────────────────────────────────────────────────────────────────────

class NNAutoEncoder:
    """
    Single-hidden-layer network equivalent to PCA.

    Encoder:  h = W1 @ (x - mean)
    Decoder:  x' = W1.T @ h   (optimal reconstruction)

    W1 are the top-d left singular vectors of the centred data matrix.
    Retrieval uses L2 distance in the d-dimensional latent space.

    Why this differs from LBP
    ─────────────────────────
    LBP applies a hand-designed rule (local threshold comparison) to encode
    texture micro-patterns. PCA/NN learns the globally optimal linear subspace
    that minimises pixel-level reconstruction error.  It captures dataset-
    specific variance directions — including global illumination gradients and
    background statistics — that LBP completely ignores.
    """

    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.components_  = None
        self.mean_         = None

    def fit(self, X: np.ndarray) -> "NNAutoEncoder":
        self.mean_       = X.mean(axis=0)
        _, _, Vt         = svd(X - self.mean_, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) @ self.components_.T).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# DNN — Stacked RFF encoder
# ─────────────────────────────────────────────────────────────────────────────

class DeepNNEncoder:
    """
    Two-layer non-linear encoder using Random Fourier Features (RFF).

    Each RFF layer approximates an RBF kernel via Bochner's theorem:
        h = tanh([cos(Wx + b) || sin(Wx + b)] / sqrt(d))

    The final layer is projected through PCA for compact representation.

    Why DNN > NN but < CNN
    ──────────────────────
    DNN captures non-linear feature interactions that PCA misses.  However,
    both NN and DNN treat the image as a flat vector — every pixel connects
    equally to every hidden unit, so spatial layout is completely lost.
    CNN-style features constrain weights to local receptive fields and pool
    hierarchically, explicitly encoding spatial structure that neither
    NN nor DNN can represent.
    """

    def __init__(self, layer_dims=(256, 128), out_dim=64,
                 gamma=0.01, seed=42):
        self.layer_dims = list(layer_dims)
        self.out_dim    = out_dim
        self.gamma      = gamma
        self.seed       = seed
        self._W = []
        self._b = []
        self.pca = NNAutoEncoder(n_components=out_dim)

    def _rff(self, X, d_out, rng):
        W = rng.normal(0, np.sqrt(2 * self.gamma), (d_out, X.shape[1]))
        b = rng.uniform(0, 2 * np.pi, d_out)
        self._W.append(W); self._b.append(b)
        Z = X @ W.T + b
        return np.concatenate([np.cos(Z), np.sin(Z)], axis=1) / np.sqrt(d_out)

    def _forward(self, X):
        H = X.copy()
        for W, b, d in zip(self._W, self._b, self.layer_dims):
            Z = H @ W.T + b
            H = np.tanh(np.concatenate([np.cos(Z), np.sin(Z)], axis=1)
                        / np.sqrt(d))
        return H

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        H   = X.copy()
        for d in self.layer_dims:
            H = np.tanh(self._rff(H, d, rng))
        self.pca.fit(H)
        return self

    def transform(self, X):
        return self.pca.transform(self._forward(X))

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ─────────────────────────────────────────────────────────────────────────────
# CNN substitute — HOG + Spatial Pyramid Matching
# ─────────────────────────────────────────────────────────────────────────────

def _hog_cell(patch: np.ndarray, n_bins: int = 9) -> np.ndarray:
    """Gradient orientation histogram for one cell (Sobel + unsigned angle)."""
    gx   = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=1)
    gy   = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=1)
    mag  = np.sqrt(gx**2 + gy**2)
    ang  = np.degrees(np.arctan2(gy, gx)) % 180
    hist = np.zeros(n_bins, dtype=np.float32)
    bw   = 180.0 / n_bins
    for b in range(n_bins):
        mask      = (ang >= b*bw) & (ang < (b+1)*bw)
        hist[b]   = mag[mask].sum()
    return hist


def compute_hog(image: np.ndarray, cell: int = 8,
                block: int = 2, n_bins: int = 9) -> np.ndarray:
    """
    Full HOG descriptor with L2-Hys block normalisation.

    Mimics what the first conv layer of a CNN learns (oriented edge detectors)
    but uses fixed Sobel filters instead of learned weights.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    else:
        gray = image.astype(np.float32)

    H, W   = gray.shape
    ncy    = H // cell;  ncx = W // cell
    cells  = np.zeros((ncy, ncx, n_bins), dtype=np.float32)

    for cy in range(ncy):
        for cx in range(ncx):
            cells[cy, cx] = _hog_cell(
                gray[cy*cell:(cy+1)*cell, cx*cell:(cx+1)*cell], n_bins)

    nby, nbx = ncy - block + 1, ncx - block + 1
    if nby <= 0 or nbx <= 0:
        v = cells.ravel();  return v / (np.linalg.norm(v) + 1e-6)

    eps    = 1e-6
    blocks = []
    for by in range(nby):
        for bx in range(nbx):
            v  = cells[by:by+block, bx:bx+block, :].ravel()
            v  = v / (np.linalg.norm(v) + eps)
            v  = np.clip(v, 0, 0.2)
            v /= (np.linalg.norm(v) + eps)
            blocks.append(v)
    return np.concatenate(blocks).astype(np.float32)


def spatial_pyramid_hog(image: np.ndarray,
                         levels: list = [1, 2, 4]) -> np.ndarray:
    """
    Spatial Pyramid Matching with HOG.

    Level L divides the image into L×L tiles.  Each tile gets its own HOG
    descriptor, and finer levels receive higher pyramid weights:
        w_L = 1 / 2^(max_L - L)

    This hierarchical pooling mirrors the spatial structure of a CNN:
        Level 0  →  global average pool
        Level 1  →  2×2 spatial pool
        Level 2  →  4×4 spatial pool
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    max_L = len(levels) - 1
    parts = []
    for li, L in enumerate(levels):
        w  = 1.0 / (2 ** (max_L - li)) if li < max_L else 1.0 / (2**max_L)
        H, W = gray.shape
        th   = H // L;  tw = W // L
        if th == 0 or tw == 0:
            continue
        for ty in range(L):
            for tx in range(L):
                tile = gray[ty*th:(ty+1)*th, tx*tw:(tx+1)*tw]
                tile = cv2.resize(tile, (32, 32))
                parts.append(compute_hog(tile) * w)

    feat = np.concatenate(parts)
    return (feat / (np.linalg.norm(feat) + 1e-8)).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Batch helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_flat(images, target=(32, 32)):
    """Flatten grayscale images to 1-D vectors (NN/DNN input)."""
    out = []
    for img in images:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        out.append(cv2.resize(g, target).astype(np.float32).ravel() / 255.0)
    return np.array(out, dtype=np.float32)


def extract_hog_batch(images, target=(64, 64), spm=True):
    """Extract HOG (or SPM-HOG) for a list of images; returns (N, D) array."""
    rows = []
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        r = cv2.resize(img, target)
        rows.append(spatial_pyramid_hog(r) if spm else
                    compute_hog(cv2.cvtColor(r, cv2.COLOR_BGR2GRAY)))
    max_d  = max(len(r) for r in rows)
    out    = np.zeros((len(rows), max_d), dtype=np.float32)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out
