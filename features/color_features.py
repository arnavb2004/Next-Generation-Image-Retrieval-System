"""
Task 6 — Inter- and Intra-Color Feature Analysis

Intra-channel: per-channel statistical moments and histograms
Inter-channel: Pearson correlations, log-ratios, joint 2-D histograms

Additional color spaces: HSV (chromatic/luminance separation),
CIE Lab* (perceptually uniform distance).
"""

import numpy as np
import cv2
from scipy.stats import skew, kurtosis as sp_kurt


# ─────────────────────────────────────────────────────────────────────────────
# Intra-channel
# ─────────────────────────────────────────────────────────────────────────────

def intra_moments(image: np.ndarray) -> np.ndarray:
    """
    4 statistical moments per channel (mean, std, skewness, excess kurtosis).
    f_intra = [μ_R, σ_R, skew_R, kurt_R, μ_G, σ_G, ..., μ_B, σ_B, ...]
    Length = 12.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img = image.astype(np.float32) / 255.0
    feats = []
    for c in range(3):
        ch = img[:, :, c].ravel()
        feats += [float(ch.mean()), float(ch.std()),
                  float(skew(ch)), float(sp_kurt(ch))]
    return np.array(feats, dtype=np.float32)


def intra_histogram(image: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """Per-channel normalised histogram (length = 3 * n_bins)."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hists = []
    for c in range(3):
        h = cv2.calcHist([image], [c], None, [n_bins], [0, 256]).ravel()
        hists.append(h / (h.sum() + 1e-8))
    return np.concatenate(hists).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Inter-channel
# ─────────────────────────────────────────────────────────────────────────────

def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    da = a - a.mean();  db = b - b.mean()
    return float((da * db).sum() / (np.sqrt((da**2).sum() * (db**2).sum()) + 1e-8))


def inter_stats(image: np.ndarray) -> np.ndarray:
    """
    corr(R,G), corr(G,B), corr(R,B)   — Pearson correlations
    log(μ_R/μ_G), log(μ_G/μ_B), log(μ_R/μ_B)  — illumination-invariant ratios
    energy_R, energy_G, energy_B               — per-channel L2 energy
    Length = 9.
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img = image.astype(np.float32) / 255.0
    B, G, R = [img[:, :, c].ravel() for c in range(3)]
    eps = 1e-6
    return np.array([
        _pearson(R, G), _pearson(G, B), _pearson(R, B),
        float(np.log((R.mean() + eps) / (G.mean() + eps))),
        float(np.log((G.mean() + eps) / (B.mean() + eps))),
        float(np.log((R.mean() + eps) / (B.mean() + eps))),
        float((R**2).mean()), float((G**2).mean()), float((B**2).mean()),
    ], dtype=np.float32)


def inter_joint_hist(image: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """
    Joint 2-D histograms for (B,G), (G,R), (B,R) pairs.
    Captures non-linear inter-channel dependencies beyond Pearson.
    Length = 3 * n_bins².
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img = image.astype(np.float32) / 255.0
    ch  = [img[:, :, c].ravel() for c in range(3)]
    out = []
    for i, j in [(0,1),(1,2),(0,2)]:
        h2d, _, _ = np.histogram2d(ch[i], ch[j], bins=n_bins,
                                    range=[[0,1],[0,1]])
        h2d /= (h2d.sum() + 1e-8)
        out.append(h2d.ravel())
    return np.concatenate(out).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Alternative color spaces
# ─────────────────────────────────────────────────────────────────────────────

def hsv_histogram(image: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """HSV histograms. Separates chroma (H,S) from luminance (V)."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ranges = [[0,180],[0,256],[0,256]]
    return np.concatenate([
        cv2.calcHist([hsv],[c],None,[n_bins],r).ravel() /
        (cv2.calcHist([hsv],[c],None,[n_bins],r).sum() + 1e-8)
        for c, r in enumerate(ranges)
    ]).astype(np.float32)


def lab_histogram(image: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """CIE Lab* histograms — perceptually uniform distances."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    return np.concatenate([
        cv2.calcHist([lab],[c],None,[n_bins],[0,256]).ravel() /
        (cv2.calcHist([lab],[c],None,[n_bins],[0,256]).sum() + 1e-8)
        for c in range(3)
    ]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Full color descriptor
# ─────────────────────────────────────────────────────────────────────────────

def full_color_descriptor(image: np.ndarray, n_bins: int = 16) -> np.ndarray:
    """
    Concatenate all color features and L2-normalise.
    Components: intra_moments | intra_hist | inter_stats |
                inter_joint_hist | hsv_hist | lab_hist
    """
    parts = [
        intra_moments(image),
        intra_histogram(image, n_bins),
        inter_stats(image),
        inter_joint_hist(image, n_bins // 2),
        hsv_histogram(image, n_bins),
        lab_histogram(image, n_bins),
    ]
    feat = np.concatenate(parts).astype(np.float32)
    return feat / (np.linalg.norm(feat) + 1e-8)


def color_lbp_cnn_combined(color: np.ndarray, lbp: np.ndarray,
                             cnn: np.ndarray,
                             w=(1.0, 1.0, 1.0)) -> np.ndarray:
    """L2-normalise each descriptor then combine with explicit weights."""
    def l2(v): return v / (np.linalg.norm(v) + 1e-8)
    feat = np.concatenate([w[0]*l2(color), w[1]*l2(lbp), w[2]*l2(cnn)])
    return feat / (np.linalg.norm(feat) + 1e-8)
