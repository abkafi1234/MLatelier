# MLatelier — Zero-Code ML Prototyping Dashboard

MLatelier is a Streamlit application that lets researchers and domain experts run complete machine-learning experiments — tabular, vision, and NLP — without writing a single line of model-training code. Upload data, pick models, click **Run**, and get a publication-ready diagnostics panel, optimised model export, LIME/SHAP explanations, and an optional Gemini AI assistant.

---

## Table of Contents

1. [Quick Install](#quick-install)
2. [Launching the App](#launching-the-app)
3. [Features at a Glance](#features-at-a-glance)
4. [Tab-by-Tab Guide](#tab-by-tab-guide)
5. [Supported Models](#supported-models)
6. [Scientific Insights Panel](#scientific-insights-panel)
7. [Export and Reproducibility](#export-and-reproducibility)
8. [Project Structure](#project-structure)
9. [Running the Test Suite](#running-the-test-suite)
10. [Publishing to PyPI (GitHub Actions)](#publishing-to-pypi-github-actions)
11. [Benchmark Datasets](#benchmark-datasets)
12. [Requirements](#requirements)
13. [License](#license)

---

## Quick Install

```bash
pip install mlatelier
```

Everything is included in a single install — no optional extras needed.

> **GPU/CUDA note:** For a CUDA-enabled PyTorch build, install the matching torch wheels from [pytorch.org](https://pytorch.org) *before* running `pip install mlatelier`.

---

## Launching the App

```bash
mlatelier
```

Or via Python module:

```bash
python -m mlatelier
```

The dashboard opens automatically in your browser at `http://localhost:8501`.

---

## Features at a Glance

| Tab | Engine | Key capabilities |
|---|---|---|
| **Tabular ML** | scikit-learn, XGBoost, LightGBM, CatBoost | Classification, regression, multi-label; Bayesian BO; SHAP/LIME/PDP; EDA |
| **Vision DL** | PyTorch, torchvision | Transfer learning on 9 architectures; GradCAM; live epoch progress |
| **Ablation Study** | scikit-learn | Per-component sensitivity (scaler, SMOTE, split, folds) |
| **Predict** | joblib / PyTorch | Batch tabular & vision inference; confidence histograms |
| **NLP** | TF-IDF / Sentence Transformers / HuggingFace | ML + DL + fine-tune tracks; BayesSearchCV; LIME text; word cloud |
| **AI Assistant** | Google Gemini | Context-aware ML advisor aware of your leaderboard results |

---

## Tab-by-Tab Guide

### Tab 1 — Tabular ML

**Workflow**

1. Upload a `.csv` or `.xlsx/.xls` file, paste a URL, or enter a local path.
2. Select one **or more** Target Y columns (multiple = multi-label mode).
3. Select Feature X columns (numeric and/or categorical; auto-encoded).
4. Pick models from the grouped model picker.
5. Configure imbalance handling, scaler, test split, CV folds, and BO iterations.
6. Click **Run Tabular Pipeline**.

**Highlights**

- **Task auto-detection** — continuous targets → regression; multi-Y → `MultiOutputClassifier`
- **Imbalance handling** — SMOTE or random oversampling (skipped for regression and multi-label)
- **Feature scaling** — auto-select, RobustScaler, StandardScaler, or MinMaxScaler
- **Live leaderboard** — updates after each model's cross-validation completes
- **Bayesian optimisation** — `BayesSearchCV` from scikit-optimize; winner selected by holdout test score
- **EDA panel** — summary stats, histograms, box plots, correlation heatmap, missing-value heatmap, data quality report
- **All models exported** — download `.joblib` + JSON metadata sidecar for any run model, not just the winner

---

### Tab 2 — Vision DL

**Workflow**

1. Provide an image dataset as a folder path (sub-folders = class names) or a `.zip` archive.
2. Select one or more architectures and configure training settings.
3. Click **Run Vision Pipeline**.

**Highlights**

- **Architectures** — ResNet18/50, EfficientNet-B0/B3, MobileNetV3-S/L, DenseNet121, ViT-B/16, Swin-T
- **Fine-tuning strategy** — frozen backbone (fast prototyping) or full fine-tune
- **Live training curves** — epoch-by-epoch training and validation loss/accuracy
- **Post-run diagnostics** — confusion matrix, class distribution chart, sample image grid
- **GradCAM** — gradient-weighted class activation maps for incorrect predictions
- **Device-agnostic** — routes to CUDA GPU when available, falls back to CPU

---

### Tab 3 — Ablation Study

**Workflow**

1. Upload or load a dataset (same file formats as Tab 1).
2. Choose a single model and a baseline configuration.
3. Check which pipeline components to ablate.
4. Click **Run Ablation Study**.

**Highlights**

- Vary one component at a time: SMOTE, scaler, test split, CV folds
- Delta (%) shows each component's contribution relative to the baseline
- Ranked bar chart with colour-coded positive/negative impact
- SMOTE option is automatically hidden for regression tasks

---

### Tab 4 — Predict / Inference

**Tabular inference**

1. Upload a saved `.joblib` tabular model.
2. Supply new data — paste rows directly, upload CSV/Excel, or provide a URL.
3. Download a prediction CSV with confidence scores per class.

**Vision inference**

1. Upload a `.pt` vision model and its `_metadata.json` sidecar.
2. Upload image files or specify a folder path.
3. Download a batch prediction CSV.

**Highlights**

- Confidence histogram with zone shading (red < 70 %, amber 70–90 %, green ≥ 90 %)
- Uncertain prediction flagging with expandable low-confidence rows
- Missing input columns filled with 0 and flagged with a warning

---

### Tab 5 — NLP

**Workflow**

1. Upload a CSV/Excel file with a **text column** and a **label column**.
2. Select a track:
   - **ML track** — TF-IDF or Count vectorizer → sklearn classifiers → BayesSearchCV
   - **DL track (Sentence Transformers)** — frozen embeddings → sklearn classifier head
   - **DL track (HuggingFace Fine-tune)** — end-to-end fine-tuning of any HF model
3. Pick models (grouped by family) and configure preprocessing / embedding settings.
4. Click **Run NLP Pipeline**.

**ML Track details**

- **Preprocessing** — TF-IDF or Count vectorizer; stopword removal; Porter stemming; n-gram range (1–3); min/max document frequency; sublinear TF scaling
- **Classifiers** — Logistic Regression, Linear SVC, SGD, Multinomial NB, Complement NB, Random Forest, XGBoost
- **Bayesian optimisation** — BayesSearchCV over regularisation and tree hyperparameters
- **LIME text explanations** — per-sample word-level attribution for the winning model
- **Top feature chart** — top positive/negative coefficient words per class (linear models)

**DL Track (Sentence Transformers) details**

- **Models** — `all-MiniLM-L6-v2`, `all-mpnet-base-v2`, `paraphrase-MiniLM-L6-v2`
- Embeds the entire corpus once; classifiers are trained on the fixed embeddings (fast, no GPU fine-tuning loop)

**DL Track (HuggingFace Fine-tune) details**

- **Any HuggingFace model** — type any model ID (DistilBERT, BERT, RoBERTa, XLM-RoBERTa, etc.) or choose a preset
- Configurable: epochs, batch size, learning rate, max sequence length, warmup ratio
- Head-only or full fine-tune; model saved with `save_pretrained()` for reuse

**NLP Visualisations**

- Word cloud (corpus-wide or per-class)
- Text length and character count distributions
- Per-class word count box plot

---

### Tab 6 — AI Assistant

Enter a **Google Gemini API key** in the sidebar to activate a context-aware ML assistant.

The assistant is automatically briefed on:
- The current leaderboard (model names, accuracy scores)
- The best hyperparameters found by Bayesian optimisation
- The active tab and task type

Useful for: interpreting results, suggesting next steps, explaining model behaviour, and generating code snippets.

---

## Supported Models

### Tabular Classification
Random Forest, XGBoost, Gradient Boosting, Extra Trees, Hist Gradient Boosting, SVM, Logistic Regression, Ridge Classifier, MLP, Decision Tree, KNN, Bernoulli NB, Gaussian NB, LDA, QDA, AdaBoost, SGD, Linear SVC, Passive Aggressive, LightGBM, CatBoost

### Tabular Regression
Random Forest, XGBoost, Gradient Boosting, Extra Trees, Hist Gradient Boosting, SVR, Linear Regression, Ridge, MLP, Decision Tree, KNN, AdaBoost, SGD, Linear SVR, Passive Aggressive, LightGBM, CatBoost

### Vision Architectures
ResNet18, ResNet50, EfficientNet-B0, EfficientNet-B3, MobileNetV3-Small, MobileNetV3-Large, DenseNet121, ViT-B/16, Swin-T

### NLP Classifiers (ML and DL tracks)
Logistic Regression, Linear SVC, SGD Classifier, Multinomial NB, Complement NB, Random Forest, XGBoost

### NLP Fine-tune (HuggingFace track)
Any model on HuggingFace Hub — presets: DistilBERT, BERT, RoBERTa, DistilRoBERTa, XLM-RoBERTa, ALBERT

---

## Scientific Insights Panel

Shown after every completed run:

| Section | Contents |
|---|---|
| **Leaderboard** | Accuracy, Macro F1, Weighted F1 per model; timing; CSV export |
| **Classification Report** | Per-class precision / recall / F1 / support |
| **Confusion Matrix** | Standard (single-label) or per-label 2×2 grid (multi-label) |
| **ROC Curves** | Per-class AUC curves (classification only) |
| **Feature Importance** | Top-15 features by importance or coefficient magnitude |
| **SHAP Global** | Mean \|SHAP\| per feature (multi-class: class-average) |
| **SHAP Waterfall** | Single-sample SHAP breakdown |
| **PDP** | Partial dependence plots for top numeric features |
| **LIME** | Single-sample LIME tabular or text explanation |
| **Calibration** | Reliability diagram + Brier score (classification only) |
| **Convergence** | Bayesian optimisation score history |
| **Learning Curve** | Train-set size vs CV score (on-demand) |
| **Feature Ablation** | Drop-one feature impact table and chart (on-demand) |
| **Regression Report** | Residuals vs Fitted, Q-Q, histogram + Normal PDF, Scale-Location; Shapiro-Wilk and KS tests |
| **Training Curve** | Epoch-level training / validation curves (Vision DL) |
| **Word Cloud** | Corpus-wide and per-class word cloud (NLP) |
| **Text Stats** | Word / char length distributions and per-class box plot (NLP) |
| **NLP LIME** | Word-level attribution bar chart for winning NLP model |
| **Top Features** | Per-class positive/negative coefficient words (linear NLP models) |

---

## Export and Reproducibility

- Per-model `.joblib` download buttons for every trained tabular model
- JSON metadata sidecar alongside each export (feature list, hyperparameters, metric)
- Leaderboard CSV download
- Full self-contained HTML experiment report
- LaTeX `booktabs` table for direct inclusion in papers
- Experiment archive saved under `~/MLatelier/experiments/`

---

## Project Structure

```text
Proto-ML/
├── pyproject.toml               # pip packaging config and entry point
├── requirements.txt             # plain dependency list
├── README.md
│
├── mlatelier/                   # installed Python package
│   ├── __init__.py              # exposes __version__ = "1.0.0"
│   ├── __main__.py              # CLI entry point — runs `streamlit run app.py`
│   ├── app.py                   # Streamlit UI: 6-tab layout, session state, result rendering
│   ├── tabular_engine.py        # Baseline CV, Bayesian BO, SHAP/LIME/PDP, multi-label
│   ├── vision_engine.py         # Vision training loop, GradCAM, dataset summary
│   ├── nlp_engine.py            # TF-IDF + ST + HF fine-tune pipelines, BayesOpt, LIME, visualisations
│   ├── inference_engine.py      # Tabular and vision model loading and batch inference
│   ├── reporting.py             # Every chart, table, and export function
│   ├── file_utils.py            # CSV / Excel reading helpers (Streamlit-free, fully testable)
│   ├── tracker.py               # Experiment saving and HTML report generation
│   └── utils.py                 # Page styling and session-state helpers
│
├── tests/
│   ├── test_tabular_engine.py   # 40+ tests: baseline, BO, multi-label, PDP, SHAP, LIME
│   ├── test_nlp_engine.py       # 136 tests: every model, every config flag, DL track, LIME
│   ├── test_nlp_enhancements.py # 43 tests: rich preprocessing, HF fine-tune, NLP visualisations
│   ├── test_reporting.py        # 70+ tests: every render function including NLP renderers
│   ├── test_inference_engine.py # 12 tests: tabular and vision inference
│   ├── test_vision_analysis.py  # 14 tests: dataset summary, device detection, model loading
│   └── test_file_reading.py     # 20+ tests: CSV/Excel reading, sheet selection
│
└── .github/
    └── workflows/
        └── publish.yml          # CI: test → build → publish to PyPI on version tag
```

---

## Running the Test Suite

```bash
pip install -e "."
pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run only fast tests (skip vision and DL-track tests)
python -m pytest tests/ -v --ignore=tests/test_vision_analysis.py -k "not DlTrack"
```

Test coverage at a glance:

| File | Tests | Notes |
|---|---|---|
| `test_nlp_engine.py` | 136 (125 pass, 11 skip) | LIME tests skip when `lime` not installed |
| `test_nlp_enhancements.py` | 43 | Rich preprocessing, HF fine-tune (mocked), NLP visualisations |
| `test_reporting.py` | 70+ | All render functions including NLP renderers |
| `test_tabular_engine.py` | 40+ | Baseline, BO, multi-label, PDP, SHAP, LIME |
| `test_inference_engine.py` | 12 | Tabular and vision inference |
| `test_vision_analysis.py` | 14 | Dataset summary, device detection |
| `test_file_reading.py` | 20+ | CSV/Excel formats, sheet selection |

---

## Publishing to PyPI (GitHub Actions)

The included workflow at [.github/workflows/publish.yml](.github/workflows/publish.yml) handles testing, building, and publishing automatically.

### How it works

| Trigger | What runs |
|---|---|
| Push to `main` or any PR | `test` job — runs suite on Python 3.9, 3.11, 3.12 |
| Push of a version tag `v*.*.*` | `test` → `build` → `publish` to PyPI |

### One-time PyPI setup (Trusted Publisher — no token needed)

1. Log in to [pypi.org](https://pypi.org) → **Your projects** → select or create `mlatelier`.
2. Go to **Settings → Publishing → Add a new publisher** → choose **GitHub Actions**.
3. Fill in:
   - Owner: `<your-github-username>`
   - Repository: `<your-repo-name>`
   - Workflow filename: `publish.yml`
   - Environment name: `pypi`
4. Save.

### How to release a new version

```bash
# 1. Bump version in pyproject.toml and mlatelier/__init__.py
# 2. Commit and push
git add pyproject.toml mlatelier/__init__.py
git commit -m "chore: bump version to v1.1.0"
git push origin main

# 3. Tag and push the tag — this triggers the publish job
git tag v1.1.0
git push origin v1.1.0
```

The workflow will run tests, build `dist/mlatelier-1.1.0-py3-none-any.whl` and the sdist, then upload both to PyPI. The new version is live within seconds.

> **Alternative: API token** — If you prefer a token over Trusted Publisher, create one at pypi.org → Account settings → API tokens, store it as a GitHub Actions secret named `PYPI_API_TOKEN`, and follow the token-based instructions in the comments inside `publish.yml`.

---

## Benchmark Datasets

| Task | Dataset | Rows | Source |
|---|---|---|---|
| Binary classification | Breast Cancer Wisconsin | 569 | `sklearn.datasets.load_breast_cancer()` → CSV |
| Regression | Diabetes | 442 | `sklearn.datasets.load_diabetes()` → CSV |
| Multi-label classification | Emotions | ~1,000 | Kaggle; 6 binary label columns |
| Image classification | Rock Paper Scissors | 2,520 | Kaggle; 3 balanced classes; folder-per-class |
| Text classification | 20 Newsgroups | 18,846 | `sklearn.datasets.fetch_20newsgroups()` → CSV |
| Sentiment analysis | IMDB Reviews | 50,000 | Kaggle; binary positive/negative labels |

---

## Requirements

- Python 3.9 or newer
- CPU is sufficient for all tabular and NLP experiments
- CUDA GPU is recommended (but not required) for Vision DL and NLP DL tracks

All dependencies are installed automatically with `pip install mlatelier`:

```
streamlit, pandas, numpy, matplotlib, seaborn, scipy, scikit-learn, scikit-optimize,
joblib, imbalanced-learn, xgboost, lightgbm, catboost, shap, lime, torch, torchvision,
nltk, sentence-transformers, transformers, wordcloud, google-genai
```

---

## Notes

- **URL ingestion** requires network access (`requests`).
- **ZIP extraction** includes a path-traversal safety check before unpacking.
- **Multi-label mode** wraps every classifier in `MultiOutputClassifier`; SMOTE, SHAP, LIME, and PDP are skipped automatically.
- **SHAP waterfall** values are averaged across classes for multi-class models.
- **Missing inference columns** are filled with 0 and flagged with a warning.
- **skopt tuple bug** — scikit-optimize 0.10.x crashes when a `Categorical` space contains tuple values; the NLP engine works around this by excluding `ngram_range` from BayesSearchCV and controlling it at pipeline build time instead.

---

## License

MIT License. See `LICENSE` for details.
