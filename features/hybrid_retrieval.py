"""
Task 5 — Hybrid Retrieval Model

Two fusion strategies:
  1. Concatenation:    f = L2_norm([L2(f_CNN) ‖ L2(f_Proposed)])
  2. Weighted sum:     f = L2_norm(λ·L2(f_CNN) + (1-λ)·L2(f_Proposed))

Optimal λ is found by grid-search on a small validation subset
maximising Precision@10.
"""

import numpy as np


def _l2(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation."""
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def concat_fusion(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """L2-normalise each descriptor then concatenate and re-normalise."""
    return _l2(np.concatenate([_l2(A), _l2(B)], axis=1))


def weighted_fusion(A: np.ndarray, B: np.ndarray, lam: float = 0.5) -> np.ndarray:
    """
    λ·L2(A) + (1-λ)·L2(B), then L2-normalise.
    Pads the shorter descriptor with zeros if dimensions differ.
    """
    a, b = _l2(A), _l2(B)
    da, db = a.shape[1], b.shape[1]
    if da != db:
        d = max(da, db)
        a = np.pad(a, ((0, 0), (0, d - da)))
        b = np.pad(b, ((0, 0), (0, d - db)))
    return _l2(lam * a + (1 - lam) * b)


def find_optimal_lambda(A: np.ndarray, B: np.ndarray,
                         labels: np.ndarray,
                         n_q: int = 50, k: int = 10,
                         lambdas=None, seed: int = 0) -> float:
    """Grid-search for the λ that maximises Precision@k."""
    if lambdas is None:
        lambdas = np.linspace(0.1, 0.9, 9)

    rng     = np.random.default_rng(seed)
    q_idx   = rng.choice(len(labels), size=min(n_q, len(labels)), replace=False)
    best_lam, best_p = 0.5, -1.0

    for lam in lambdas:
        fused = weighted_fusion(A, B, lam)
        precs = []
        for qi in q_idx:
            dists      = np.linalg.norm(fused - fused[qi], axis=1)
            dists[qi]  = np.inf
            top         = np.argsort(dists)[:k]
            precs.append(np.sum(labels[top] == labels[qi]) / k)
        mean_p = float(np.mean(precs))
        if mean_p > best_p:
            best_p, best_lam = mean_p, float(lam)

    return best_lam


class HybridRetriever:
    """
    Pre-builds a fused index and answers single-image queries at inference.

    Usage
    -----
    hr = HybridRetriever("concat")
    hr.build(feats_cnn, feats_prop, labels)
    ranked, dists = hr.query(q_cnn, q_prop, k=10)
    """

    def __init__(self, strategy: str = "concat", lam: float = 0.5):
        assert strategy in ("concat", "weighted")
        self.strategy = strategy
        self.lam      = lam
        self.index    = None
        self.labels   = None

    def build(self, A: np.ndarray, B: np.ndarray, labels: np.ndarray):
        self.labels = labels
        if self.strategy == "concat":
            self.index = concat_fusion(A, B)
        else:
            self.index = weighted_fusion(A, B, self.lam)

    def query(self, qa: np.ndarray, qb: np.ndarray, k: int = 10):
        if self.strategy == "concat":
            qf = concat_fusion(qa[None], qb[None])[0]
        else:
            qf = weighted_fusion(qa[None], qb[None], self.lam)[0]
        dists  = np.linalg.norm(self.index - qf, axis=1)
        ranked = np.argsort(dists)[:k]
        return ranked, dists[ranked]
