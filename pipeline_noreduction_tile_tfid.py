"""
Pipeline: CLASES → tiles 8x8 → KMeans (K=256) → BoVW → TF-IDF → NN classifier
Imágenes de entrada: 225x225 px → tiles de 8x8 con stride 8 (sin overlap)
Todo se guarda en: experimen_noreduction/
"""

import os
import cv2
import numpy as np
import pickle
import pandas as pd
from tqdm import tqdm
from threadpoolctl import threadpool_limits

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt
import seaborn as sns

# ─────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────
INPUT_ROOT   = "CLASES"
EXCLUDE_DIR  = "OUTLIERS"
TILE_SIZE    = 8
IMG_SIZE     = 225
BEST_K       = 256
N_CORES      = 16
NGRAMS       = (1, 4)

OUTPUT_ROOT  = "experiment_noreduction_puyucunapi_ngram1-4"
METRICS_DIR  = os.path.join(OUTPUT_ROOT, "metrics")
PKL_KMEANS   = os.path.join(OUTPUT_ROOT, "kmeans_bovw.pkl")
PKL_TFIDF    = os.path.join(OUTPUT_ROOT, "tfidf_vectorizer.pkl")

BATCH_SIZE   = 128
EPOCHS       = 60
LR           = 1e-3
DROPOUT      = 0.3
HIDDEN       = [256, 128]
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(METRICS_DIR, exist_ok=True)
print(f"Dispositivo: {DEVICE}")

# ─────────────────────────────────────────────
# PASO 1: RECOLECTAR IMÁGENES
# ─────────────────────────────────────────────
def collect_images(root, exclude):
    paths, labels = [], []
    for cls in sorted(os.listdir(root)):
        if cls == exclude:
            continue
        cls_path = os.path.join(root, cls)
        if not os.path.isdir(cls_path):
            continue
        for f in os.listdir(cls_path):
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                paths.append(os.path.join(cls_path, f))
                labels.append(cls)
    return paths, labels

print("\n[1/5] Recolectando imágenes...")
img_paths, img_labels = collect_images(INPUT_ROOT, EXCLUDE_DIR)
print(f"  Total imágenes: {len(img_paths)} | Clases: {sorted(set(img_labels))}")

# ─────────────────────────────────────────────
# PASO 2: EXTRACCIÓN DE TILES (8x8 → vectores de 64 dims)
# ─────────────────────────────────────────────
def extract_tiles(img_bgr, tile_size=TILE_SIZE):
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w  = gray.shape
    tiles = []
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            tile = gray[y:y+tile_size, x:x+tile_size].astype(np.float32) / 255.0
            tiles.append(tile.flatten())  # (64,)
    return tiles

print("\n[2/5] Extrayendo tiles de todas las imágenes...")
all_tiles    = []
img_tile_map = []

for path, label in tqdm(zip(img_paths, img_labels), total=len(img_paths)):
    img = cv2.imread(path)
    if img is None:
        continue
    if img.shape[:2] != (IMG_SIZE, IMG_SIZE):
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    tiles = extract_tiles(img)
    if tiles:
        img_tile_map.append((path, label, tiles))
        all_tiles.extend(tiles)

all_tiles_matrix = np.vstack(all_tiles).astype(np.float32)
print(f"  Total tiles: {all_tiles_matrix.shape[0]} | Dims: {all_tiles_matrix.shape[1]}")

# ─────────────────────────────────────────────
# PASO 3: KMEANS con K=256
# ─────────────────────────────────────────────
print(f"\n[3/5] Entrenando KMeans con K={BEST_K}...")
kmeans = MiniBatchKMeans(n_clusters=BEST_K, batch_size=2048,
                         init="k-means++", n_init=5,
                         random_state=42, verbose=1)
with threadpool_limits(limits=N_CORES):
    kmeans.fit(all_tiles_matrix)

with open(PKL_KMEANS, "wb") as f:
    pickle.dump(kmeans, f)
print(f"  Guardado: {PKL_KMEANS}")

# ─────────────────────────────────────────────
# PASO 4: CODIFICACIÓN BoVW → secuencia de tokens
# ─────────────────────────────────────────────
print("\n[4/5] Codificando imágenes como secuencias de clusters...")

def id_to_token(cid):
    return f"w{cid}"

rows = []
for path, label, tiles in tqdm(img_tile_map):
    tile_matrix = np.vstack(tiles).astype(np.float32)
    cluster_ids = kmeans.predict(tile_matrix)
    sentence    = " ".join(id_to_token(c) for c in cluster_ids)
    rows.append({"image": os.path.basename(path),
                 "label": label,
                 "sentence": sentence})

df = pd.DataFrame(rows)
df.to_csv(os.path.join(METRICS_DIR, "visual_sentences.csv"), index=False)
print(f"  Ejemplos de secuencias:")
for _, r in df.head(3).iterrows():
    print(f"    [{r['label']}] {r['sentence']}")

# ─────────────────────────────────────────────
# PASO 5: TF-IDF (1,1)-gramas → NN classifier
# ─────────────────────────────────────────────
print("\n[5/5] Aplicando TF-IDF y entrenando red neuronal...")

MIN_SAMPLES = 7
class_counts = df["label"].value_counts()
valid_classes = class_counts[class_counts >= MIN_SAMPLES].index
dropped = class_counts[class_counts < MIN_SAMPLES]
if len(dropped) > 0:
    print(f"  Clases omitidas por pocas muestras (<{MIN_SAMPLES}):")
    for cls, cnt in dropped.items():
        print(f"    {cls}: {cnt} muestras")
df = df[df["label"].isin(valid_classes)].reset_index(drop=True)
print(f"  Imágenes tras filtrado: {len(df)}  |  Clases restantes: {sorted(valid_classes.tolist())}")

tfidf = TfidfVectorizer(
    token_pattern=r"w\d+",
    ngram_range=NGRAMS,
    analyzer="word"
)
X_all = tfidf.fit_transform(df["sentence"].values).toarray().astype(np.float32)

with open(PKL_TFIDF, "wb") as f:
    pickle.dump(tfidf, f)
print(f"  TF-IDF shape: {X_all.shape}  | Guardado: {PKL_TFIDF}")

le    = LabelEncoder()
y_all = le.fit_transform(df["label"].values)
n_cls = len(le.classes_)

X_train, X_tmp, y_train, y_tmp = train_test_split(
    X_all, y_all, test_size=0.30, stratify=y_all, random_state=42)
X_val, X_test, y_val, y_test   = train_test_split(
    X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42)

print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

def to_tensor_ds(X, y):
    return TensorDataset(torch.tensor(X, dtype=torch.float32),
                         torch.tensor(y, dtype=torch.long))

train_dl = DataLoader(to_tensor_ds(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(to_tensor_ds(X_val,   y_val),   batch_size=BATCH_SIZE)
test_dl  = DataLoader(to_tensor_ds(X_test,  y_test),  batch_size=BATCH_SIZE)

class BoVW_NN(nn.Module):
    def __init__(self, in_dim, hidden_dims, n_classes, dropout):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model     = BoVW_NN(X_all.shape[1], HIDDEN, n_cls, DROPOUT).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

def evaluate(loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds   = model(xb).argmax(1)
            correct += (preds == yb).sum().item()
            total   += len(yb)
    return correct / total

history      = {"train_loss": [], "val_acc": []}
best_val_acc = 0
best_state   = None

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for xb, yb in train_dl:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)

    avg_loss = total_loss / len(X_train)
    val_acc  = evaluate(val_dl)
    scheduler.step(1 - val_acc)

    history["train_loss"].append(avg_loss)
    history["val_acc"].append(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state   = {k: v.clone() for k, v in model.state_dict().items()}

    if epoch % 10 == 0 or epoch == 1:
        print(f"  Epoch {epoch:3d}/{EPOCHS}  Loss={avg_loss:.4f}  ValAcc={val_acc:.4f}")

# ─────────────────────────────────────────────
# MÉTRICAS FINALES
# ─────────────────────────────────────────────
model.load_state_dict(best_state)
test_acc = evaluate(test_dl)
print(f"\n  Test Accuracy: {test_acc:.4f}")

model.eval()
all_preds, all_true = [], []
with torch.no_grad():
    for xb, yb in test_dl:
        preds = model(xb.to(DEVICE)).argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_true.extend(yb.numpy())

report = classification_report(all_true, all_preds,
                                target_names=le.classes_, digits=4)
print("\n" + report)

with open(os.path.join(METRICS_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Test Accuracy: {test_acc:.4f}\n\n")
    f.write(report)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], color="steelblue")
ax1.set_title("Train Loss"); ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
ax2.plot(history["val_acc"], color="seagreen")
ax2.axhline(test_acc, color="red", linestyle="--", label=f"Test={test_acc:.3f}")
ax2.set_title("Validation Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "training_curves.png"), dpi=120)
plt.close()

cm = confusion_matrix(all_true, all_preds)
fig, ax = plt.subplots(figsize=(max(6, n_cls), max(5, n_cls - 1)))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
ax.set_xlabel("Predicho"); ax.set_ylabel("Real")
ax.set_title(f"Matriz de Confusión  (Test Acc={test_acc:.3f})")
plt.tight_layout()
plt.savefig(os.path.join(METRICS_DIR, "confusion_matrix.png"), dpi=120)
plt.close()

report_dict = classification_report(all_true, all_preds,
                                     target_names=le.classes_,
                                     output_dict=True)
pd.DataFrame(report_dict).T.to_csv(os.path.join(METRICS_DIR, "metrics_per_class.csv"))

print(f"\n✓ Pipeline completo. Todo guardado en '{OUTPUT_ROOT}/'")
print(f"  metrics/training_curves.png")
print(f"  metrics/confusion_matrix.png")
print(f"  metrics/classification_report.txt")
print(f"  metrics/metrics_per_class.csv")
print(f"  metrics/visual_sentences.csv")
print(f"  kmeans_bovw.pkl")
print(f"  tfidf_vectorizer.pkl")