"""
Task 1 — Classical Retrieval using Local Binary Patterns (LBP)

Multi-scale circular LBP with uniform-pattern mapping, chi-squared distance,
and full evaluation metrics (Precision@K, Recall@K, AP, mAP).
"""

import numpy as np
import cv2
from scipy.spatial.distance import cdist


# ─────────────────────────────────────────────────────────────────────────────
# Core LBP
# ─────────────────────────────────────────────────────────────────────────────

def compute_lbp(image: np.ndarray, radius: int = 1, n_points: int = 8) -> np.ndarray:
    """
    Circular LBP with bilinear interpolation and uniform-pattern binning.

    For each pixel (x, y):
        LBP_{P,R}(x,y) = sum_{p=0}^{P-1}  s(g_p - g_c) * 2^p

    where g_c = centre intensity, g_p = bilinearly-interpolated neighbour on
    a circle of radius R, s(u) = 1 if u >= 0 else 0.

    Uniform patterns (<=2 circular bit-transitions) are indexed 0..P+1;
    non-uniform patterns collapse into one bin.

    Returns L1-normalised histogram.
    """
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = image.astype(np.float32)
    H, W = img.shape

    angles  = 2 * np.pi * np.arange(n_points) / n_points
    dx      = radius * np.cos(angles)
    dy      = -radius * np.sin(angles)

    lbp_map = np.zeros((H, W), dtype=np.uint32)

    for p in range(n_points):
        x0 = int(np.floor(dx[p]));  x1 = x0 + 1
        y0 = int(np.floor(dy[p]));  y1 = y0 + 1
        wx = dx[p] - x0;            wy = dy[p] - y0

        g00 = np.roll(np.roll(img, -y0, 0), -x0, 1)
        g10 = np.roll(np.roll(img, -y1, 0), -x0, 1)
        g01 = np.roll(np.roll(img, -y0, 0), -x1, 1)
        g11 = np.roll(np.roll(img, -y1, 0), -x1, 1)

        nb  = ((1-wx)*(1-wy)*g00 + (1-wx)*wy*g10
               + wx*(1-wy)*g01 + wx*wy*g11)

        lbp_map += (nb >= img).astype(np.uint32) * (2 ** p)

    n_bins = n_points * (n_points - 1) + 3
    hist   = np.zeros(n_bins, dtype=np.float32)

    for code in range(2 ** n_points):
        bits        = np.array([(code >> p) & 1 for p in range(n_points)])
        transitions = int(np.sum(np.abs(np.diff(np.append(bits, bits[0])))))
        bin_idx     = int(bits.sum()) if transitions <= 2 else n_points + 1
        hist[bin_idx] += int((lbp_map == code).sum())

    total = hist.sum()
    return (hist / total).astype(np.float32) if total > 0 else hist


def extract_multiscale_lbp(image: np.ndarray,
                            radii: list = [1, 2, 3],
                            n_pts:  list = [8, 8, 8]) -> np.ndarray:
    """Concatenate LBP histograms across (R, P) scales."""
    return np.concatenate([compute_lbp(image, r, p) for r, p in zip(radii, n_pts)])


# ─────────────────────────────────────────────────────────────────────────────
# Distance and retrieval
# ─────────────────────────────────────────────────────────────────────────────

def chi2_distance_batch(query: np.ndarray, db: np.ndarray,
                         eps: float = 1e-10) -> np.ndarray:
    """Vectorised chi-squared distance from query to each row in db."""
    return 0.5 * np.sum((query - db) ** 2 / (query + db + eps), axis=1)


def retrieve_topk(query_feat, db_feats, db_labels, k=10, metric="chi2"):
    if metric == "chi2":
        dists = chi2_distance_batch(query_feat, db_feats)
    else:
        dists = cdist(query_feat[None], db_feats, metric=metric).ravel()
    ranked = np.argsort(dists)[:k]
    return ranked, dists[ranked], db_labels[ranked]


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation metrics
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved_labels, query_label):
    return float(np.sum(retrieved_labels == query_label) / len(retrieved_labels))

def recall_at_k(retrieved_labels, query_label, total_relevant):
    return float(np.sum(retrieved_labels == query_label) / max(total_relevant, 1))

def average_precision(ranked_labels, query_label, total_relevant):
    hits, ap = 0, 0.0
    for i, lbl in enumerate(ranked_labels, 1):
        if lbl == query_label:
            hits += 1
            ap   += hits / i
    return ap / max(total_relevant, 1)

def mean_average_precision(all_ranked, query_labels, db_labels):
    aps = []
    for ranked, q in zip(all_ranked, query_labels):
        tr  = int(np.sum(db_labels == q))
        aps.append(average_precision(db_labels[ranked], q, tr))
    return float(np.mean(aps))
