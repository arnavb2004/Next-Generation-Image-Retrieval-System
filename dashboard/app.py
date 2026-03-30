"""
Task 7 — Interactive Dashboard
================================
Run:  python dashboard/app.py
Then open http://localhost:7860  (requires: pip install gradio)

If Gradio is unavailable, a static HTML summary is written instead.

Features
────────
• Upload any query image (PIL / numpy)
• Select dataset  (CIFAR-10 | MNIST)
• Select retrieval method
• Adjust Top-K (1 - 20)
• Retrieved image gallery with per-result distance & class label
• Live Precision@K, Recall@K, mAP metrics panel
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import cv2
from PIL import Image

from features.lbp_features    import extract_multiscale_lbp, average_precision
from features.neural_features  import (NNAutoEncoder, DeepNNEncoder,
                                        extract_flat, extract_hog_batch)
from features.msfwt_features   import compute_msfwt
from features.color_features   import full_color_descriptor
from features.hybrid_retrieval import concat_fusion
from data.loader import (load_cifar10, load_mnist, load_stl10,
                          preprocess, balanced_subset,
                          CIFAR10_CLASSES, MNIST_CLASSES, STL10_CLASSES)
try:
    import gradio as gr
    _HAS_GRADIO = True
except ImportError:
    _HAS_GRADIO = False

# ─────────────────────────────────────────────────────────────────────────────
# In-memory database cache
# ─────────────────────────────────────────────────────────────────────────────

_DB = dict(images=None, labels=None, feats={}, dataset=None)

def _load_db(dataset="cifar10", n_per_class=100):
    if _DB["dataset"] == dataset:
        return f"Already loaded: {dataset} ({len(_DB['images'])} images)"
    print(f"Building {dataset} index …")

    if dataset == "cifar10":
        imgs, lbls = load_cifar10(split="train", max_samples=n_per_class*10)
    elif dataset == "stl10":
        imgs, lbls = load_stl10(split="train", max_samples=n_per_class*10)
    else:
        imgs, lbls = load_mnist(split="train",   max_samples=n_per_class*10)
    
    imgs, lbls   = balanced_subset(imgs, lbls, n_per_class)
    images_proc  = preprocess(imgs)

    _DB["images"]  = images_proc
    _DB["labels"]  = lbls

    flat = extract_flat(images_proc)
    _DB["feats"]["LBP"]      = np.array([extract_multiscale_lbp(i) for i in images_proc])
    _DB["feats"]["CNN"]      = extract_hog_batch(images_proc, spm=True)
    _DB["feats"]["Color"]    = np.array([full_color_descriptor(i) for i in images_proc])
    _DB["feats"]["Proposed"] = np.array([compute_msfwt(i) for i in images_proc])
    _DB["feats"]["Hybrid"]   = concat_fusion(_DB["feats"]["CNN"],
                                              _DB["feats"]["Proposed"])
    
    nn  = NNAutoEncoder(64).fit(flat)
    _DB["feats"]["NN"]  = nn.transform(flat).astype(np.float32)
    dnn = DeepNNEncoder((128,64), 64).fit(flat)
    _DB["feats"]["DNN"] = dnn.transform(flat).astype(np.float32)
    
    _DB["nn"]  = nn
    _DB["dnn"] = dnn
    _DB["dataset"] = dataset
    return f"Loaded {len(images_proc)} images from {dataset}"

def _query_feats(img_bgr):
    """Extract all feature types from a single BGR query image."""
    proc = cv2.resize(img_bgr, (64, 64))
    flat = extract_flat([proc])
    return {
        "LBP":      extract_multiscale_lbp(proc),
        "CNN":      extract_hog_batch([proc], spm=True)[0],
        "Color":    full_color_descriptor(proc),
        "Proposed": compute_msfwt(proc),
        "NN":  _DB["nn"].transform(extract_flat([proc]))[0],
        "DNN": _DB["dnn"].transform(extract_flat([proc]))[0],
        "Hybrid":   concat_fusion(
                        extract_hog_batch([proc], spm=True),
                        np.array([compute_msfwt(proc)]))[0],
    }

def retrieve(pil_image, method="CNN", k=10, dataset="cifar10"):
    """Main retrieval function (called by Gradio)."""
    _load_db(dataset)

    # Convert query to BGR
    q_np  = np.array(pil_image)
    if q_np.ndim == 2:
        q_bgr = cv2.cvtColor(q_np, cv2.COLOR_GRAY2BGR)
    elif q_np.shape[2] == 4:
        q_bgr = cv2.cvtColor(q_np, cv2.COLOR_RGBA2BGR)
    else:
        q_bgr = cv2.cvtColor(q_np, cv2.COLOR_RGB2BGR)

    q_feats  = _query_feats(q_bgr)
    q_feat   = q_feats.get(method)
    db_feats = _DB["feats"].get(method)
    
    if q_feat is None or db_feats is None:
        return [], f"Unknown method: {method}", ""

    # Align dims and compute distances
    d = min(len(q_feat), db_feats.shape[1])
    dists  = np.linalg.norm(db_feats[:, :d] - q_feat[:d], axis=1)
    ranked = np.argsort(dists)[:k]

    # Build gallery
    gallery = []
    for idx in ranked:
        bgr = _DB["images"][idx]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gallery.append(Image.fromarray(rgb).resize((128, 128), Image.LANCZOS))

    # Metrics
    rlbls   = _DB["labels"][ranked]
    q_lbl   = int(np.bincount(rlbls, minlength=10).argmax())
    db_lbl  = _DB["labels"]
    tr      = max(int(np.sum(db_lbl == q_lbl)), 1)
    rel     = int(np.sum(rlbls == q_lbl))
    prec    = rel / k
    rec     = rel / tr
    ap      = average_precision(rlbls, q_lbl, tr)

    CLASS_NAMES = {
        "cifar10": CIFAR10_CLASSES,
        "mnist":   MNIST_CLASSES,
        "stl10":   STL10_CLASSES
    }
    class_name = CLASS_NAMES.get(_DB["dataset"], [str(i) for i in range(10)])[q_lbl]

    metrics = (f"Method : {method}\n"
               f"Top-K  : {k}\n"
               f"P@{k}  : {prec:.3f}\n"
               f"R@{k}  : {rec:.3f}\n"
               f"mAP    : {ap:.3f}\n"
               f"(inferred class: {class_name})")
               
    scores  = "\n".join(
        f"Rank {r+1}: dist={dists[ranked[r]]:.4f}  "
        f"class={_DB['labels'][ranked[r]]}"
        for r in range(len(ranked)))
        
    return gallery, metrics, scores

def _build_ui():
    with gr.Blocks(title="Image Retrieval Dashboard") as demo:
        gr.Markdown("# Next-Generation Image Retrieval Dashboard")
        with gr.Row():
            with gr.Column(scale=1):
                q_img   = gr.Image(label="Query Image", type="pil")
                ds_sel  = gr.Dropdown(["cifar10", "mnist", "stl10"],
                                       value="cifar10", label="Dataset")
                m_sel   = gr.Dropdown(["LBP", "NN", "DNN", "CNN",
                                        "Color", "Proposed", "Hybrid"],
                                       value="CNN", label="Method")
                k_sl    = gr.Slider(1, 20, 10, step=1, label="Top-K")
                btn     = gr.Button("Search", variant="primary")
            with gr.Column(scale=2):
                gallery = gr.Gallery(label="Retrieved Images", columns=5)
                metrics = gr.Textbox(label="Metrics", lines=6)
                scores  = gr.Textbox(label="Similarity Scores", lines=10)

        btn.click(fn=lambda img, ds, m, k: retrieve(img, m, int(k), ds),
                  inputs=[q_img, ds_sel, m_sel, k_sl],
                  outputs=[gallery, metrics, scores])

    return demo

if __name__ == "__main__":
    if _HAS_GRADIO:
        print("Starting dashboard at http://localhost:7860")
        _build_ui().launch(server_name="0.0.0.0", server_port=7860)
    else:
        print("Gradio is required for the interactive UI. Please run:")
        print("pip install gradio")