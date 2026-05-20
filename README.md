# NLP-Inspired Visual Tokenization Pipeline
## CLASSES → SIFT → KMeans (optimal K) → BoVW → TF-IDF → Neural Network

Classification pipeline for camera trap species images, inspired by natural language processing. Each image is treated as a "document" where visual descriptors are "words".

---

## Requirements

```bash
pip install opencv-python numpy scikit-learn torch pandas tqdm matplotlib seaborn threadpoolctl
```

---

## Expected directory structure

```
CLASES/
├── AVE_GRANDE/
│   ├── image1.jpg
│   └── ...
├── AVE_PEQUEÑA/
├── MAMIFERO_GRANDE/
├── MAMIFERO_MEDIANO/
├── MAMIFERO_PEQUEÑO/
└── OUTLIERS/          ← excluded automatically
```

Images must be `.jpg`, `.jpeg` or `.png`. The target size is **225×225 px** (resized automatically).

---

## Main configuration

| Parameter | Default value | Description |
|---|---|---|
| `INPUT_ROOT` | `"CLASES"` | Root folder with per-class subdirectories |
| `EXCLUDE_DIR` | `"OUTLIERS"` | Subdirectory to ignore |
| `IMG_SIZE` | `225` | Image size in pixels |
| `N_CORES` | `16` | Cores for KMeans |
| `SIFT_N_FEATURES` | `100` | Max keypoints per image (0 = no limit) |
| `SIFT_CONTRASTH` | `0.03` | SIFT contrast threshold (lower → more keypoints) |
| `SIFT_EDGEH` | `10` | SIFT edge threshold |
| `K_MIN / K_MAX / K_STEP` | `64 / 512 / 64` | Search range for optimal K |
| `K_SAMPLE` | `50 000` | Descriptors used to evaluate clustering metrics |
| `NGRAMS` | `(1, 4)` | N-gram range for TF-IDF |
| `BATCH_SIZE` | `128` | Batch size for training |
| `EPOCHS` | `60` | Training epochs |
| `LR` | `1e-3` | Learning rate (Adam) |
| `DROPOUT` | `0.3` | Dropout rate |
| `HIDDEN` | `[256, 128]` | Neurons per hidden layer |

---

## Usage

```bash
python pipeline.py
```

The script prints progress for each step. If a GPU is available, training runs on CUDA automatically.

---

## Pipeline step by step

### Step 1 — Image collection
Walks through `CLASES/` and collects paths and labels for all images, excluding `OUTLIERS/`.

### Step 2 — SIFT descriptor extraction
For each image:
1. Resize to 225×225 px if needed.
2. Convert to grayscale.
3. Extract up to `SIFT_N_FEATURES` keypoints with `cv2.SIFT_create`.
4. Each keypoint produces a descriptor of **128 dimensions**.

Images with no detected keypoints are skipped.

### Step 3 — Optimal K search (KMeans)
Evaluates `MiniBatchKMeans` for each `k` in `[K_MIN, K_MAX]` with step `K_STEP`, measuring:

- **Inertia** → elbow method (2nd derivative)
- **Silhouette score** ↑
- **Davies-Bouldin score** ↓

The chosen K is the one at the **inertia elbow**. Selection plots and the final model are saved to `kmeans_bovw.pkl`.

### Step 4 — BoVW encoding with spatial ordering
For each image:
1. Detect keypoints and their `(x, y)` coordinates.
2. Compute each keypoint's distance to the **image center** (112.5, 112.5).
3. Sort keypoints by ascending distance (center → edges).
4. Assign each descriptor to its nearest cluster → generates a token sequence such as `"w45 w12 w200 ..."`.

This sequence is the image's "visual document".

### Step 5 — TF-IDF + Neural Network
1. **TF-IDF** with n-grams `(1,4)` over the token sequences → one numeric vector per image.
2. Split: **70% train / 15% val / 15% test** (stratified).
3. **Neural network** `Linear → ReLU → Dropout` × 2 hidden layers + output layer.
4. Adam optimizer with `ReduceLROnPlateau` (patience=5, factor=0.5).
5. The checkpoint with the **best validation accuracy** is saved.

---

## Neural network architecture

```
Input (TF-IDF dim)
    ↓
Linear(input, 256) → ReLU → Dropout(0.3)
    ↓
Linear(256, 128) → ReLU → Dropout(0.3)
    ↓
Linear(128, n_classes)
    ↓
Prediction (CrossEntropyLoss)
```

---

## Generated outputs

```
experiment_noreduction_sift_sorted/
├── kmeans_bovw.pkl                    # Trained KMeans model
├── tfidf_vectorizer.pkl               # Trained TF-IDF vectorizer
└── metrics/
    ├── kmeans_k_selection.png         # K selection plots
    ├── training_curves.png            # Training loss and validation accuracy
    ├── confusion_matrix.png           # Confusion matrix on test set
    ├── classification_report.txt      # Per-class report (precision, recall, F1)
    ├── metrics_per_class.csv          # Per-class metrics in CSV format
    └── visual_sentences.csv           # Token sequences per image
```

---

## Reference results (Puyucunapi dataset, 1 050 images, 5 classes)

| Class | Precision | Recall | F1 | n |
|---|---|---|---|---|
| AVE_GRANDE           | 0.7966 | 0.8876 | **0.8396** | 525  |
| AVE_PEQUEÑA          | 0.6933 | 0.6445 | 0.6680     | 256  |
| MAMIFERO_GRANDE      | 0.8077 | 0.7975 | **0.8025** | 79   |
| MAMIFERO_MEDIANO     | 0.6324 | 0.5658 | 0.5972     | 152  |
| **MAMIFERO_PEQUEÑO** | 0.3077 | 0.1053 | **0.1569** | 38   |
| Weighted average     | 0.7308 | 0.7467 | 0.7352     | 1050 |
| Macro average        | 0.6475 | 0.6001 | 0.6129     | 1050 |

> **Overall accuracy: 74.67%** — A significant improvement over the previous version (+8.67 pp). The low F1 for MAMIFERO_PEQUEÑO persists due to class imbalance (only 3.6% of the dataset).

---

## Notes

- Class imbalance is the main bottleneck. To address it: use weighted loss (`class_weight`), oversampling (SMOTE), or data augmentation for minority classes.
- The `.pkl` models can be reused for inference on new images without rerunning the full pipeline.
- To change the visual vocabulary, adjust the `K_MIN`/`K_MAX` range and rerun from Step 3.
