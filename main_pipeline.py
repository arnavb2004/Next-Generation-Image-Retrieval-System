"""
main_pipeline.py — Full extraction, evaluation, and comparison table.

Usage
─────
python main_pipeline.py --dataset cifar10 --n_per_class 100 --k 10
python main_pipeline.py --dataset mnist   --n_per_class 100 --k 10
python main_pipeline.py --dataset stl10   --n_per_class  50 --k 10
"""

import argparse, sys, os, time
import numpy as np
from scipy.spatial.distance import cdist

# Ensure local imports work correctly
sys.path.insert(0, os.path.dirname(__file__))

from features.lbp_features    import extract_multiscale_lbp
from features.neural_features  import (NNAutoEncoder, DeepNNEncoder,
                                        extract_flat, extract_hog_batch)
from features.msfwt_features   import extract_msfwt_batch
from features.color_features   import full_color_descriptor
from features.hybrid_retrieval import (concat_fusion, find_optimal_lambda)
from data.loader import (load_cifar10, load_mnist, load_stl10,
                          preprocess, balanced_subset)


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_ap(ranked_labels, query_label, total_relevant_in_db):
    """
    Average Precision for one query.
    total_relevant_in_db = how many images of this class exist in the DB.
    """
    hits, ap = 0, 0.0
    for rank, lbl in enumerate(ranked_labels, start=1):
        if lbl == query_label:
            hits += 1
            ap += hits / rank
    return ap / max(total_relevant_in_db, 1)


def evaluate(db_f, q_f, db_lbl, q_lbl, k=10):
    """
    Evaluate retrieval for one feature type.
    Returns dict: precision@k, recall@k, mAP
    """
    def l2norm(X):
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        return X / n

    # Using Cosine distance on L2-normalised features
    D = cdist(l2norm(q_f), l2norm(db_f), "cosine")

    prec_l, rec_l, ap_l = [], [], []
    for qi in range(len(q_lbl)):
        q      = q_lbl[qi]
        dists  = D[qi]
        ranked = np.argsort(dists)[:k]
        rlbls  = db_lbl[ranked]

        total_rel = max(int(np.sum(db_lbl == q)), 1)
        rel       = int(np.sum(rlbls == q))

        prec_l.append(rel / k)
        rec_l.append(rel / total_rel)
        ap_l.append(compute_ap(rlbls, q, total_rel))

    return dict(
        precision = float(np.mean(prec_l)),
        recall    = float(np.mean(rec_l)),
        mAP       = float(np.mean(ap_l))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_all(images_proc, labels, verbose=True):
    feats = {}

    def run_task(name, fn):
        if verbose: print(f"  [{name}]", end=" ", flush=True)
        t = time.time()
        feats[name] = fn()
        if verbose: print(f"dim={feats[name].shape[1]}  ({time.time()-t:.1f}s)")

    run_task("LBP",   lambda: np.array([extract_multiscale_lbp(img)
                                    for img in images_proc], dtype=np.float32))
    
    flat = extract_flat(images_proc)
    nn   = NNAutoEncoder(128).fit(flat)
    run_task("NN",    lambda: nn.transform(flat))
    
    dnn  = DeepNNEncoder((256, 128), 128).fit(flat)
    run_task("DNN",   lambda: dnn.transform(flat))
    
    run_task("CNN",   lambda: extract_hog_batch(images_proc, spm=True))
    
    run_task("Color", lambda: np.array([full_color_descriptor(img)
                                    for img in images_proc], dtype=np.float32))
    
    run_task("Proposed", lambda: extract_msfwt_batch(images_proc))

    lam = find_optimal_lambda(feats["CNN"], feats["Proposed"], labels,
                               n_q=min(50, len(labels)))
    
    if verbose: print(f"  [Hybrid] optimal λ={lam:.2f}", end=" ", flush=True)
    feats["Hybrid"] = concat_fusion(feats["CNN"], feats["Proposed"])
    feats["_lam"]   = lam
    if verbose: print(f"dim={feats['Hybrid'].shape[1]}")

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(dataset="cifar10", n_per_class=100, k=10, verbose=True):
    print(f"\n{'═'*45}")
    print(f"  Dataset={dataset.upper()}  n/class={n_per_class}  Top-K={k}")
    print(f"{'═'*45}")

    if dataset == "cifar10":
        imgs, lbls = load_cifar10(split="train", max_samples=n_per_class * 10)
    elif dataset == "mnist":
        imgs, lbls = load_mnist(split="train",   max_samples=n_per_class * 10)
    elif dataset == "stl10": # stl10
        imgs, lbls = load_stl10(split="train",   max_samples=n_per_class * 10)
    else:
        print(f"Unknown dataset: {dataset}")
        return {}
    
    imgs, lbls = balanced_subset(imgs, lbls, n_per_class)
    print(f"  Loaded {len(imgs)} images, {len(np.unique(lbls))} classes")

    images_proc = preprocess(imgs)

    rng     = np.random.default_rng(42)
    d_idx_l, t_idx_l = [], []
    for c in np.unique(lbls):
        idx     = np.where(lbls == c)[0]
        n_test  = max(1, int(len(idx) * 0.2))
        perm    = rng.permutation(idx)
        t_idx_l.append(perm[:n_test])
        d_idx_l.append(perm[n_test:])
    
    d_idx = np.concatenate(d_idx_l)
    t_idx = np.concatenate(t_idx_l)

    lbls_db = lbls[d_idx]
    lbls_q  = lbls[t_idx]
    print(f"  DB={len(d_idx)} images  Queries={len(t_idx)} images")

    print("\nExtracting features …")
    feats = extract_all(images_proc, lbls, verbose)

    methods = ["LBP", "NN", "DNN", "CNN", "Color", "Proposed", "Hybrid"]
    results = {}
    print(f"\nEvaluating (k={k}) …")
    for m in methods:
        if m not in feats or not isinstance(feats[m], np.ndarray):
            continue
        res = evaluate(feats[m][d_idx], feats[m][t_idx], lbls_db, lbls_q, k)
        results[m] = res
        print(f"  {m:<12}  P@{k}={res['precision']:.3f}"
              f"  R@{k}={res['recall']:.3f}  mAP={res['mAP']:.3f}")
    
    return results


def print_table(results, k=10):
    """Prints the final comparative analysis table without remarks."""
    print(f"\n{'Method':<15} {'P@'+str(k):<12} {'R@'+str(k):<12} {'mAP':<12}")
    print("─" * 51)
    for m, r in results.items():
        print(f"{m:<15} {r['precision']:<12.4f} {r['recall']:<12.4f} {r['mAP']:<12.4f}")
    print("─" * 51)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",     default="cifar10",
                    choices=["cifar10", "mnist", "stl10"])
    ap.add_argument("--n_per_class", type=int, default=100)
    ap.add_argument("--k",           type=int, default=10)
    
    args = ap.parse_args()
    res  = run_pipeline(args.dataset, args.n_per_class, args.k)
    print_table(res, args.k)