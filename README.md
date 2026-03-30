# Image Retrieval System — README

## How to Run

### 1. Install Dependencies

```bash
pip install numpy scipy opencv-python pillow matplotlib gradio
```

---

### 2. Running the Evaluation Pipeline (`main_pipeline.py`)

The `main_pipeline.py` script is the central engine for batch feature extraction
and quantitative evaluation. It automatically handles data loading, stratified
splitting, and metric calculation (Precision@10, Recall@10, and mAP).

#### Basic Command Syntax

```bash
python main_pipeline.py --dataset [mnist|cifar10|stl10|fashion_mnist] --n_per_class [int] --k [int]
```

#### Recommended Commands

**Evaluate MNIST:**
```bash
python main_pipeline.py --dataset mnist --n_per_class 100 --k 10
```

**Evaluate CIFAR-10:**
```bash
python main_pipeline.py --dataset cifar10 --n_per_class 100 --k 10
```

**Evaluate STL-10:**
```bash
python main_pipeline.py --dataset stl10 --n_per_class 100 --k 10
```

#### Output

The script generates:
- A live log of feature extraction times and descriptor dimensions
- A **Comparative Analysis Table** with Precision@K, Recall@K, and mAP
  for all seven methods:

| Method | Description |
|--------|-------------|
| LBP | Multi-scale Local Binary Patterns |
| NN | PCA linear auto-encoder |
| DNN | Stacked Random Fourier Features + PCA |
| CNN | HOG + Spatial Pyramid Matching |
| Color | RGB / HSV / Lab statistics |
| Proposed (MSFWT) | Novel frequency-saliency co-occurrence descriptor |
| Hybrid | Concatenation fusion of CNN + MSFWT |

---

### 3. Using the Interactive Dashboard (`dashboard/app.py`)

The dashboard provides a visual interface to validate system performance
and conduct failure case analysis.

#### How to Launch

```bash
python dashboard/app.py
```

Then open the URL shown in your terminal — usually `http://127.0.0.1:7860`

#### Steps to Use

1. **Select Dataset** — Choose between MNIST, CIFAR-10,
   or STL-10 from the dropdown

2. **Select Method** — Choose a feature extractor (e.g., `Hybrid` to
   see the combined power of CNN and MSFWT)

3. **Upload Query Image** — Upload any image matching the dataset
   category (e.g., a photo of a cat for CIFAR-10, a digit for MNIST)

4. **Adjust Top-K** — Use the slider to define how many results to
   retrieve (1–20)

5. **Click Search** — The system displays:
   - Retrieved image gallery (upscaled to 128×128)
   - Similarity scores per retrieved image
   - Real-time metrics: Precision@K, Recall@K, mAP
   - Inferred query class name

#### What Image to Upload

| Dataset | Upload a photo of |
|---------|------------------|
| CIFAR-10 | airplane, car, bird, cat, deer, dog, frog, horse, ship, truck |
| MNIST | handwritten digit (0–9) |
| STL-10 | airplane, bird, car, cat, deer, dog, horse, monkey, ship, truck |

---

## 4. Project Structure

```
Image_Retrieval/
├── main_pipeline.py        ← Entry point for batch evaluation
├── dashboard/
│   └── app.py              ← Entry point for Gradio-based GUI
├── features/
│   ├── lbp_features.py     ← Task 1: LBP + chi-squared retrieval
│   ├── neural_features.py  ← Tasks 2 & 3: NN, DNN, HOG+SPM
│   ├── msfwt_features.py   ← Task 4: Novel MSFWT descriptor
│   ├── hybrid_retrieval.py ← Task 5: Fusion strategies
│   └── color_features.py   ← Task 6: Color feature analysis
├── data/
│   └── loader.py           ← Auto-download + preprocessing
└── data_cache/             ← (Auto-generated) Downloaded datasets
```

---

## 5. Dataset Information

| Dataset | Classes | Resolution | Size | Download |
|---------|---------|------------|------|----------|
| MNIST | 10 (digits) | 28×28 grayscale | ~10 MB | Auto |
| CIFAR-10 | 10 (objects) | 32×32 colour | ~170 MB | Auto |
| STL-10 | 10 (objects) | 96×96 colour | ~2.5 GB | Auto |

All datasets download automatically on first run — no manual setup needed.
