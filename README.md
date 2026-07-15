
# Skin Lesion Classifier

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-FF6F00?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)
![Tests](https://img.shields.io/badge/Tests-7%20passed-brightgreen)
![License](https://img.shields.io/badge/License-Research%20Only-lightgrey)

## Project Overview

This project trains and serves a binary image classifier that distinguishes **benign from malignant** skin lesions in dermoscopic images. Early and accurate melanoma detection is critical — melanoma accounts for the majority of skin cancer deaths despite being one of the most treatable cancers when caught early. An automated screening tool can triage high-risk lesions for priority dermatologist review.

The dataset is an ISIC 2019/2020-derived collection of **11,400 resized dermoscopic images** split across three folds. The overall class distribution (55% benign, 45% malignant) is mild — a 1.24:1 ratio — but was still addressed with class-weighted loss to avoid the model silently favouring the majority class.

> ⚠️ **Research prototype only.** Not a certified medical device. Do not use for real diagnostic decisions. Always consult a qualified dermatologist.

---

## Repository Structure

```
skin-cancer-deployment/
├── .gitignore
├── README.md
├── docker-compose.yml            ← one-command full-stack launch
│
├── backend/
│   ├── src/                      ← all application source (PYTHONPATH=src in Docker)
│   │   ├── api.py                #   FastAPI entrypoint, routes & middleware
│   │   ├── schemas.py            #   Pydantic response models (typed OpenAPI contract)
│   │   ├── config.py             #   All settings — fully env-overridable, type-annotated
│   │   ├── model.py              #   Thread-safe singleton loader + startup validation
│   │   ├── inference.py          #   Preprocessing pipeline — mirrors training exactly
│   │   └── monitoring.py        #   Structured logging + in-memory rolling metrics
│   ├── models/                   ← model artefacts (gitignored; mounted as volume in Docker)
│   │   └── Custom_CNN.keras
│   ├── tests/
│   │   ├── conftest.py           #   Session-scoped app + client fixtures
│   │   └── test_api.py          #   7 tests, 100% endpoint coverage
│   ├── pyproject.toml            ← packaging + ruff + mypy + pytest config
│   ├── requirements.txt
│   ├── Dockerfile                ← multi-stage, non-root, Python 3.12-slim
│   ├── .dockerignore
│   ├── .env.example
│   └── start.sh                  ← `./start.sh` to run locally
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx               #   Upload UI — health poll, backend status banner
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
└── Notebooks/
    ├── EDA.ipynb                 ← dataset audit, class distribution
    ├── Training.ipynb            ← all 3 model architectures, training, evaluation
    ├── Error_analysis.ipynb      ← per-class error breakdown, failure mode inspection
    └── threshold_tuning.ipynb    ← precision-recall sweep → threshold selection (0.34)
```

---

## Dataset

| Split | Benign | Malignant | Total |
|-------|--------|-----------|-------|
| Train | 5,200  | 4,000     | 9,200 |
| Val   | 550    | 550       | 1,100 |
| Test  | 550    | 550       | 1,100 |
| **Total** | **6,300** | **5,100** | **11,400** |

**Key EDA findings:**
- Class imbalance ratio: **1.24:1** (benign:malignant) — mild, treated with class-weighted loss only, no oversampling.
- Image dimensions vary (up to 19 distinct resolutions); all images resized to **224×224** for training.
- Benign images are measurably brighter (mean brightness 157.66 vs 132.22 for malignant) — colour is diagnostically meaningful in dermoscopy, so colour jitter augmentation was deliberately excluded to prevent the model learning from distorted colour signals.
- 0 corrupted images found across all 11,400 files.

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| **Custom CNN** ✅ | **0.8145** | 0.8240 | 0.8000 | **0.8118** | 0.8966 |
| EfficientNetB0 (full fine-tune) | 0.7745 | **0.9467** | 0.5818 | 0.7207 | **0.9356** |
| EfficientNetB4 (transfer learning) | 0.7227 | 0.8490 | 0.5418 | 0.6615 | 0.8662 |

**Train–test accuracy gap (overfitting check):**

| Model | Train Acc | Test Acc | Gap |
|---|---|---|---|
| **Custom CNN** | 0.8990 | 0.8145 | **0.0845** |
| EfficientNetB0 (full fine-tune) | 0.9876 | 0.7745 | 0.2131 |
| EfficientNetB4 (transfer learning) | 0.8923 | 0.7227 | 0.1696 |

EfficientNetB0 memorised the training set almost perfectly (0.9876) but failed to generalise (gap 0.2131), making it unreliable for unseen lesions. EfficientNetB4 (frozen base, head-only) had the worst test accuracy despite the large model — the frozen features did not adapt well enough to dermoscopy at this dataset scale. The Custom CNN, despite having no pretrained knowledge, achieved the best F1 score and the smallest train–test gap, indicating the most reliable generalisation.

---

## Why Custom CNN

The Custom CNN was selected as the production model for the following reasons:

- **Highest F1 score (0.8118):** F1 is the primary metric for imbalanced binary classification — it balances precision and recall rather than rewarding the model for being good at only one.
- **Highest test accuracy (0.8145):** Correct on 81% of unseen lesions.
- **Best generalisation (gap 0.0845):** The smallest train–test gap of all three models — evidence that the regularisation strategy (class-weighted loss, dropout, early stopping) worked as intended.
- **Clinically acceptable recall (0.80 at default threshold, 0.96 at tuned threshold):** The model misses fewer malignant cases than the EfficientNet variants at comparable operating points.
- EfficientNetB0's high AUC (0.9356) is promising but does not translate to usable test performance — a model that correctly identifies only 58% of malignant cases (recall 0.5818) is not acceptable for clinical screening.

---

## Technical Decisions

### Class Imbalance

The 1.24:1 benign-to-malignant ratio was treated with **class-weighted cross-entropy loss** only:

```python
class_weights = {0: 0.8846, 1: 1.15}  # computed from training counts
```

No oversampling or synthetic data augmentation was used for balancing. At this ratio, weighted loss is sufficient and avoids the risk of overfitting to duplicated minority-class images.

### Data Augmentation

Augmentations were chosen with clinical plausibility in mind, applied as model layers (automatically disabled at inference):

- **Random horizontal + vertical flip:** lesion orientation carries no diagnostic meaning.
- **Small random rotation (±20°):** accounts for image acquisition angle variation.
- **Small random zoom (±10%):** simulates variation in dermoscope distance.
- **No colour jitter / hue distortion:** colour is diagnostically meaningful in dermoscopy (ABCDE criteria include colour). Distorting it risks teaching the model to ignore a real signal, or worse, to learn a spurious shortcut from the distorted signal.

### Preprocessing Parity

The inference pipeline in `src/inference.py` mirrors the training pipeline exactly:

```python
img = tf.image.decode_image(image_bytes, channels=3)
img = tf.image.resize(img, (224, 224))   # bilinear — matches training default
img = tf.cast(img, tf.float32)            # values stay in [0, 255]
```

The Custom CNN has an internal `Rescaling(1/255)` layer as its first layer — pre-scaling to [0, 1] externally would rescale twice and silently corrupt predictions.

---

## Model Selection and Training Methodology

### Custom CNN (Baseline → Production Model)

**Why used:** Built from scratch to establish a lower-bound baseline with no pretrained knowledge. Also the most interpretable of the three — if a from-scratch CNN can perform comparably to pretrained architectures on a dataset this size, that tells us the task is learnable from the image signal alone.

**Architecture:** Four convolutional blocks (filters: 32 → 64 → 128 → 256), each with BatchNorm and MaxPooling. Classifier head: `Flatten → Dense(256) → Dropout(0.5) → Dense(1, sigmoid)`. A `Rescaling(1/255)` layer is the first layer (all other models have this built into the EfficientNet base).

**How trained:**
- Random weight initialisation (no pretraining)
- Up to 15 epochs; early stopping with patience 4 (not triggered)
- Learning rate: 1e-3, Adam optimiser, weight decay 5e-4
- Class-weighted CrossEntropyLoss (weights: {0: 0.8846, 1: 1.15})
- ReduceLROnPlateau (patience 2, factor 0.5) on val_loss

### EfficientNetB0 — Full Fine-Tuning

**Why used:** Lightweight (5.3M trainable params in the base) pretrained model, fully unfrozen to assess how well transfer learning adapts to dermoscopy features. Full fine-tuning is the highest-risk approach for this dataset size and was included to quantify that risk.

**Architecture:** Standard EfficientNetB0 base (all layers unfrozen) + new head: `GlobalAveragePooling2D → Dense(128) → Dropout(0.3) → Dense(1, sigmoid)`.

**How trained:**
- Initialised with ImageNet weights, all layers unfrozen
- Up to 15 epochs
- Learning rate: 1e-3
- Same class weighting, early stopping, and LR scheduling as Custom CNN

**Result:** Train accuracy 0.9876 vs test accuracy 0.7745 — gap of 0.2131. The model memorised training-specific features and failed to generalise. The high AUC (0.9356) reflects good ranking ability but the poor recall (0.5818) makes it unsuitable for screening.

### EfficientNetB4 — Transfer Learning (Frozen Base)

**Why used:** Higher-capacity pretrained model (17.7M non-trainable params) with the base frozen — only the classification head is trained. This is the most compute-efficient approach and typically the most overfitting-resistant, but the frozen features may be too generic for the domain shift from ImageNet to dermoscopy.

**Architecture:** EfficientNetB4 base (all layers frozen, 17.67M non-trainable params) + `GlobalAveragePooling2D → Dense(128) → Dropout(0.3) → Dense(1, sigmoid)`.

**How trained:**
- Initialised with ImageNet weights, base fully frozen
- Only head parameters trained (1,793 trainable params)
- Up to 15 epochs
- Learning rate: 1e-3

**Result:** Test accuracy 0.7227, F1 0.6615. The frozen ImageNet features did not transfer well enough to dermoscopy at this dataset scale. The domain gap between natural images and clinical dermoscopy images is large enough that even a large pretrained model underperforms a small trained-from-scratch CNN when the base is not allowed to adapt.

### Training Commonality (All Models)

| Setting | Value |
|---|---|
| Image size | 224 × 224 |
| Batch size | 32 |
| Optimiser | Adam |
| Loss | Class-weighted binary cross-entropy |
| LR scheduler | ReduceLROnPlateau (patience=2, factor=0.5) |
| Early stopping | patience=4, monitor=val_loss, restore_best_weights=True |
| Augmentation | Flip (H+V), Rotation ±20°, Zoom ±10% |
| Framework | TensorFlow / Keras 2.16 |

---

## Threshold Selection — Why 0.34

The default sigmoid threshold of 0.50 treats a missed malignant case and an unnecessary follow-up as equally costly. In a clinical screening context, **they are not**.

A **false negative** (malignant lesion classified as benign) means a patient goes untreated — a potentially life-threatening outcome. A **false positive** (benign lesion flagged as malignant) means an unnecessary biopsy — uncomfortable and costly, but survivable.

### Threshold Sweep Results (from `Notebooks/threshold_tuning.ipynb`)

Three candidate operating points were evaluated on the 1,100-image validation set:

| Candidate | Threshold | Accuracy | Precision | Recall | F1 | FP | FN | Misclassified |
|---|---|---|---|---|---|---|---|---|
| Min-misclassification | 0.48 | 0.8836 | 0.8588 | 0.9182 | 0.8875 | 83 | 45 | 128 |
| F1-optimal | 0.48 | 0.8836 | 0.8588 | 0.9182 | 0.8875 | 83 | 45 | 128 |
| **F2-optimal (recall-priority)** | **0.34** | **0.8718** | **0.8141** | **0.9636** | **0.8826** | 121 | 20 | 141 |

The min-misclassification and F1-optimal thresholds agreed at 0.48 — two different objectives converging on the same value, which indicates a stable property of the model rather than noise. However, "fewest total errors" is not the right objective for a malignancy screener.

**Choosing 0.34 (F2-optimal):**

Moving from 0.48 → 0.34 produces:
- **25 fewer missed malignant cases** (45 FN → 20 FN): recall improves from 91.8% → 96.4%
- **38 more false positives** (83 FP → 121 FP): precision drops from 85.9% → 81.4%
- **13 more total misclassifications** (128 → 141)

In a screening tool, **20 missed malignancies vs 45 missed malignancies is the correct trade-off to make**, even at the cost of 38 additional follow-up referrals. The F2 score (which weights recall twice as heavily as precision) formalises this clinical preference and identified 0.34 as the optimal operating point.

> The 0.34 threshold was selected because it minimises **false negatives** — the clinically costly error — at an acceptable precision cost, as justified by the F2-weighted precision-recall analysis in the threshold tuning notebook.

---

## Quickstart

### Option A — Docker Compose (recommended, zero setup)

You can launch both the frontend and backend services together in a single command using Docker Compose. The configuration in [docker-compose.yml](file:///Volumes/hdd/vs%20code/python/skin-cancer-deployment/docker-compose.yml) handles service building, networking, and environment variables:

1. **Launch the stack:**
   ```bash
   # From the root directory of the repository
   docker compose up --build
   ```

2. **How it works:**
   - **Backend Service:** Builds using [backend/Dockerfile](file:///Volumes/hdd/vs%20code/python/skin-cancer-deployment/backend/Dockerfile). It mounts `./backend/models` as a read-only volume to `/app/models` inside the container (so you can swap models without rebuilding the image) and mounts `./backend/logs` to persist prediction logs.
   - **Frontend Service:** Runs a Node.js development container and proxies requests to the backend container.
   - **Healthcheck Coordination:** The frontend container waits to launch until the backend passes its startup health check (which verifies that the `.keras` model loads and validates successfully).

3. **Access URLs:**
   - **Frontend UI:** [http://localhost:5173](http://localhost:5173)
   - **Backend API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
   - **Backend Health Check:** [http://localhost:8000/health](http://localhost:8000/health)

4. **Stop the stack:**
   ```bash
   docker compose down
   ```

### Option B — Local development

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./start.sh                     # sets PYTHONPATH=src automatically
```

**Frontend (separate terminal):**
```bash
cd frontend
npm install && npm run dev
```

---

## API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | `{status, model_path}` — poll before trusting `/predict` |
| `/metrics` | GET | Rolling stats: total requests, avg latency, recent malignant rate |
| `/predict` | POST | Multipart `file=<image>` → `{label, probability_malignant, confidence, threshold, inference_time_ms}` |

**Example prediction:**
```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@/path/to/lesion.jpg"
# {
#   "label": "malignant",
#   "probability_malignant": 0.7812,
#   "confidence": 0.7812,
#   "threshold": 0.34,
#   "inference_time_ms": 147.3
# }
```

---

## Production & Monitoring

Every prediction is logged to `backend/logs/predictions.log` in structured JSON:

| Field | Description |
|---|---|
| `event` | Always `"prediction"` |
| `filename` | Original uploaded filename |
| `label` | `benign` or `malignant` |
| `probability_malignant` | Raw sigmoid output [0, 1] |
| `confidence` | Probability of the winning class |
| `threshold` | Decision threshold applied |
| `inference_time_ms` | End-to-end latency in milliseconds |

The `/metrics` endpoint exposes a rolling window summary (default: last 500 predictions) including average latency and recent malignant rate — useful for detecting distribution shift without external infrastructure.

**Production upgrade path:** replace the flat log file with InfluxDB or Prometheus for time-series querying; run drift checks on a schedule via Airflow or APScheduler; alert on sustained shifts in the malignant rate.

---

## Future Improvements

- **Larger dataset:** train on the full ISIC 2019 challenge dataset (25,000+ images, 9 diagnostic categories) for broader coverage and more robust feature learning.
- **Partial fine-tuning of EfficientNetB0:** unfreeze only the top N layers rather than all or none — likely to close the train-test gap seen in full fine-tuning while retaining transfer learning benefit.
- **Ensemble:** soft-vote the Custom CNN and EfficientNetB0 outputs — the CNN leads on F1/recall while EB0 leads on precision/AUC; their error patterns are likely complementary.
- **Test-time augmentation (TTA):** average predictions over multiple augmented views of the same input to reduce variance on borderline cases.
- **Confidence calibration:** apply temperature scaling to ensure that `probability_malignant` values are well-calibrated (i.e. a 0.7 output genuinely reflects 70% probability) — important for communicating uncertainty to clinicians.
- **Grad-CAM visualisation:** add a `/explain` endpoint that returns a heatmap showing which image regions drove the classification, increasing interpretability for clinical users.
=======
# Skin-Lesion-Classifier
>>>>>>> bef92bb14d91d0585a86c5bc26c763bff16e4586
