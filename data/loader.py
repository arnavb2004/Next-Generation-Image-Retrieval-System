"""
Data loading for CIFAR-10, MNIST, and STL-10.
All datasets are downloaded automatically on first use.

Additional dataset choice: STL-10
Justification
─────────────
• 96*96 colour images (3* larger than CIFAR-10) stress spatial feature
  methods (HOG cell density, MSFWT block resolution) more than 32×32 images.
• Designed specifically to benchmark unsupervised feature learning on
  real-world images — ideal for comparing classical vs. learned descriptors.
• Classes partially overlap with CIFAR-10, enabling cross-dataset transfer
  analysis.
• Difficulty gradient: MNIST (simple, greyscale) → CIFAR-10 (moderate,
  colour) → STL-10 (challenging, high-res colour).
"""


import os, gzip, struct, tarfile, pickle, urllib.request
import numpy as np
import cv2

_HERE    = os.path.dirname(__file__)
DATA_DIR = os.path.join(_HERE, "..", "data_cache")


def _mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10
# ─────────────────────────────────────────────────────────────────────────────

_C10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
_C10_DIR = os.path.join(DATA_DIR, "cifar10")
CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]


def _get_cifar10():
    extracted = os.path.join(_C10_DIR, "cifar-10-batches-py")
    if not os.path.exists(extracted):
        _mkdir(_C10_DIR)
        archive = os.path.join(_C10_DIR, "cifar10.tar.gz")
        print("Downloading CIFAR-10 …")
        urllib.request.urlretrieve(_C10_URL, archive)
        with tarfile.open(archive) as tf:
            tf.extractall(_C10_DIR)
    return extracted


def load_cifar10(split="train", max_samples=None):
    d = _get_cifar10()
    files = ([f"data_batch_{i}" for i in range(1,6)]
             if split == "train" else ["test_batch"])
    imgs, lbls = [], []
    for f in files:
        with open(os.path.join(d, f), "rb") as fh:
            b = pickle.load(fh, encoding="bytes")
        imgs.append(b[b"data"].reshape(-1,3,32,32).transpose(0,2,3,1))
        lbls.append(np.array(b[b"labels"]))
    images = np.concatenate(imgs);  labels = np.concatenate(lbls)
    if max_samples:
        images, labels = images[:max_samples], labels[:max_samples]
    return images, labels          # uint8 RGB, int labels


# ─────────────────────────────────────────────────────────────────────────────
# MNIST
# ─────────────────────────────────────────────────────────────────────────────

_FMN_BASE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

MNIST_CLASSES = [str(i) for i in range(10)]
FASHION_MNIST_CLASSES = ["T-shirt","Trouser","Pullover","Dress","Coat",
                          "Sandal","Shirt","Sneaker","Bag","Ankle boot"]
def load_fashion_mnist(split="train", max_samples=None):
    _mkdir(_MN_DIR)
    prefix = "train" if split == "train" else "t10k"
    for fname in _MN_FILES.values():
        fpath = os.path.join(_MN_DIR, "fashion_" + fname)
        if not os.path.exists(fpath):
            print(f"Downloading Fashion-MNIST {fname} …")
            urllib.request.urlretrieve(_FMN_BASE + fname, fpath)

    images = _read_mnist_images(
        os.path.join(_MN_DIR, f"fashion_{prefix}-images-idx3-ubyte.gz"))
    labels = _read_mnist_labels(
        os.path.join(_MN_DIR, f"fashion_{prefix}-labels-idx1-ubyte.gz"))
    if max_samples:
        images, labels = images[:max_samples], labels[:max_samples]
    return images, labels



_MN_BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_MN_FILES = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_lbl": "train-labels-idx1-ubyte.gz",
    "test_img":  "t10k-images-idx3-ubyte.gz",
    "test_lbl":  "t10k-labels-idx1-ubyte.gz",
}
_MN_DIR = os.path.join(DATA_DIR, "mnist")
MNIST_CLASSES = [str(i) for i in range(10)]


def _get_mnist():
    _mkdir(_MN_DIR)
    for fname in _MN_FILES.values():
        fpath = os.path.join(_MN_DIR, fname)
        if not os.path.exists(fpath):
            print(f"Downloading MNIST {fname} …")
            urllib.request.urlretrieve(_MN_BASE + fname, fpath)


def _read_mnist_images(path):
    with gzip.open(path, "rb") as f:
        _, n, h, w = struct.unpack(">IIII", f.read(16))
        return np.frombuffer(f.read(), np.uint8).reshape(n, h, w)


def _read_mnist_labels(path):
    with gzip.open(path, "rb") as f:
        _, n = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), np.uint8)


def load_mnist(split="train", max_samples=None):
    _get_mnist()
    p = "train" if split == "train" else "t10k"
    images = _read_mnist_images(os.path.join(_MN_DIR, f"{p}-images-idx3-ubyte.gz"))
    labels = _read_mnist_labels(os.path.join(_MN_DIR, f"{p}-labels-idx1-ubyte.gz"))
    if max_samples:
        images, labels = images[:max_samples], labels[:max_samples]
    return images, labels          # uint8 greyscale (H,W), int labels


# ─────────────────────────────────────────────────────────────────────────────
# STL-10
# ─────────────────────────────────────────────────────────────────────────────

_STL_URL = "https://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
_STL_DIR = os.path.join(DATA_DIR, "stl10")
STL10_CLASSES = [
    "airplane","bird","car","cat","deer",
    "dog","horse","monkey","ship","truck"
]


def _get_stl10():
    extracted = os.path.join(_STL_DIR, "stl10_binary")
    if not os.path.exists(extracted):
        _mkdir(_STL_DIR)
        archive = os.path.join(_STL_DIR, "stl10.tar.gz")
        print("Downloading STL-10 (~2.5 GB) …")
        urllib.request.urlretrieve(_STL_URL, archive)
        with tarfile.open(archive) as tf:
            tf.extractall(_STL_DIR)
    return extracted


def load_stl10(split="train", max_samples=500):
    d = _get_stl10()
    p = "train" if split == "train" else "test"
    with open(os.path.join(d, f"{p}_X.bin"), "rb") as f:
        raw = np.frombuffer(f.read(), np.uint8)
    N      = len(raw) // (3*96*96)
    images = raw.reshape(N, 3, 96, 96).transpose(0,2,3,1)   # RGB
    with open(os.path.join(d, f"{p}_y.bin"), "rb") as f:
        labels = np.frombuffer(f.read(), np.uint8) - 1       # 0-indexed
    if max_samples:
        images, labels = images[:max_samples], labels[:max_samples]
    return images, labels


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(images, target=(64,64), to_bgr=True):
    """Resize and optionally convert RGB→BGR for OpenCV. Returns list."""
    out = []
    for img in images:
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif to_bgr:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.append(cv2.resize(img, target))
    return out


def balanced_subset(images, labels, n_per_class=100, seed=42):
    """Draw a balanced random subset."""
    rng  = np.random.default_rng(seed)
    keep = []
    for c in np.unique(labels):
        idx  = np.where(labels == c)[0]
        keep.append(rng.choice(idx, size=min(n_per_class, len(idx)), replace=False))
    idx  = np.concatenate(keep)
    perm = rng.permutation(len(idx))
    return images[idx[perm]], labels[idx[perm]]
