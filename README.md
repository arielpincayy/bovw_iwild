# NLP-Inspired Visual Tokenization Pipeline
## CLASES → SIFT → KMeans (K óptimo) → BoVW → TF-IDF → Red Neuronal

Pipeline de clasificación de especies en imágenes de trampas cámara, inspirado en procesamiento de lenguaje natural. Trata cada imagen como un "documento" donde los descriptores visuales son "palabras".

---

## Requisitos

```bash
pip install opencv-python numpy scikit-learn torch pandas tqdm matplotlib seaborn threadpoolctl
```

---

## Estructura de directorios esperada

```
CLASES/
├── AVE_GRANDE/
│   ├── imagen1.jpg
│   └── ...
├── AVE_PEQUEÑA/
├── MAMIFERO_GRANDE/
├── MAMIFERO_MEDIANO/
├── MAMIFERO_PEQUEÑO/
└── OUTLIERS/          ← excluido automáticamente
```

Las imágenes deben ser `.jpg`, `.jpeg` o `.png`. El tamaño objetivo es **225×225 px** (se redimensionan automáticamente).

---

## Configuración principal

| Parámetro | Valor por defecto | Descripción |
|---|---|---|
| `INPUT_ROOT` | `"CLASES"` | Carpeta raíz con subdirectorios por clase |
| `EXCLUDE_DIR` | `"OUTLIERS"` | Subdirectorio a ignorar |
| `IMG_SIZE` | `225` | Tamaño de imagen en píxeles |
| `N_CORES` | `16` | Núcleos para KMeans |
| `SIFT_N_FEATURES` | `100` | Keypoints máximos por imagen (0 = sin límite) |
| `SIFT_CONTRASTH` | `0.03` | Umbral de contraste SIFT (menor → más keypoints) |
| `SIFT_EDGEH` | `10` | Umbral de bordes SIFT |
| `K_MIN / K_MAX / K_STEP` | `64 / 512 / 64` | Rango de búsqueda del K óptimo |
| `K_SAMPLE` | `50 000` | Descriptores usados para evaluar métricas de clustering |
| `NGRAMS` | `(1, 4)` | Rango de n-gramas para TF-IDF |
| `BATCH_SIZE` | `128` | Tamaño de batch para entrenamiento |
| `EPOCHS` | `60` | Épocas de entrenamiento |
| `LR` | `1e-3` | Tasa de aprendizaje (Adam) |
| `DROPOUT` | `0.3` | Tasa de dropout |
| `HIDDEN` | `[256, 128]` | Neuronas por capa oculta |

---

## Ejecución

```bash
python pipeline.py
```

El script imprime el progreso de cada paso. Si hay GPU disponible, el entrenamiento se ejecuta en CUDA automáticamente.

---

## Pipeline paso a paso

### Paso 1 — Recolección de imágenes
Recorre `CLASES/` y recopila rutas y etiquetas de todas las imágenes, excluyendo `OUTLIERS/`.

### Paso 2 — Extracción de descriptores SIFT
Para cada imagen:
1. Redimensiona a 225×225 px si es necesario.
2. Convierte a escala de grises.
3. Extrae hasta `SIFT_N_FEATURES` keypoints con `cv2.SIFT_create`.
4. Cada keypoint produce un descriptor de **128 dimensiones**.

Las imágenes sin keypoints detectados se omiten.

### Paso 3 — Búsqueda del K óptimo (KMeans)
Evalúa `MiniBatchKMeans` para cada `k` en `[K_MIN, K_MAX]` con paso `K_STEP`, midiendo:

- **Inercia** → método del codo (2ª derivada)
- **Silhouette score** ↑
- **Davies-Bouldin score** ↓

El K elegido es el del **codo de la inercia**. Se guardan gráficas de selección y el modelo final en `kmeans_bovw.pkl`.

### Paso 4 — Codificación BoVW con orden espacial
Para cada imagen:
1. Detecta keypoints y sus coordenadas `(x, y)`.
2. Calcula la distancia de cada keypoint al **centro de la imagen** (112.5, 112.5).
3. Ordena los keypoints por distancia ascendente (centro → bordes).
4. Asigna cada descriptor al cluster más cercano → genera una secuencia de tokens como `"w45 w12 w200 ..."`.

Esta secuencia es el "documento visual" de la imagen.

### Paso 5 — TF-IDF + Red Neuronal
1. **TF-IDF** con n-gramas `(1,4)` sobre las secuencias de tokens → vector numérico por imagen.
2. División: **70% train / 15% val / 15% test** (estratificada).
3. **Red neuronal** `Linear → ReLU → Dropout` × 2 capas ocultas + capa de salida.
4. Optimizador Adam con `ReduceLROnPlateau` (patience=5, factor=0.5).
5. Se guarda el estado con la **mejor accuracy de validación**.

---

## Arquitectura de la red neuronal

```
Input (dim TF-IDF)
    ↓
Linear(input, 256) → ReLU → Dropout(0.3)
    ↓
Linear(256, 128) → ReLU → Dropout(0.3)
    ↓
Linear(128, n_clases)
    ↓
Predicción (CrossEntropyLoss)
```

---

## Salidas generadas

```
experiment_noreduction_sift_sorted/
├── kmeans_bovw.pkl                    # Modelo KMeans entrenado
├── tfidf_vectorizer.pkl               # Vectorizador TF-IDF entrenado
└── metrics/
    ├── kmeans_k_selection.png         # Gráficas de selección de K
    ├── training_curves.png            # Loss de entrenamiento y accuracy de validación
    ├── confusion_matrix.png           # Matriz de confusión en test
    ├── classification_report.txt      # Reporte por clase (precision, recall, F1)
    ├── metrics_per_class.csv          # Métricas por clase en formato CSV
    └── visual_sentences.csv           # Secuencias de tokens por imagen
```

---

## Resultados de referencia (dataset Puyucunapi, 1 050 imágenes, 5 clases)

| Clase | Precisión | Recall | F1 | n |
|---|---|---|---|---|
| Ave Grande | 0.74 | 0.75 | **0.75** | 525 |
| Ave Pequeña | 0.60 | 0.64 | 0.62 | 256 |
| Mamífero Grande | 0.72 | 0.72 | **0.72** | 79 |
| Mamífero Mediano | 0.51 | 0.52 | 0.52 | 152 |
| **Mamífero Pequeño** | 0.18 | 0.05 | **0.08** | 38 |
| Promedio ponderado | 0.65 | 0.66 | 0.66 | 1050 |
| Promedio macro | 0.55 | 0.54 | 0.54 | 1050 |

> **Accuracy global: 66%** — El bajo F1 en Mamífero Pequeño refleja el desbalance de clases (solo 3.6% del dataset).

---

## Notas

- El desbalance de clases es el principal cuello de botella. Para mejorarlo: usar pérdida ponderada (`class_weight`), oversampling (SMOTE), o aumentación de datos para clases minoritarias.
- Los modelos `.pkl` pueden reutilizarse para inferencia en nuevas imágenes sin reentrenar el pipeline completo.
- Para cambiar el vocabulario visual, ajustar el rango `K_MIN`/`K_MAX` y reejecutar desde el Paso 3.
