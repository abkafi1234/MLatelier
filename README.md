# MLatelier — Zero-Code ML Prototyping Dashboard

[![Test, Build & Publish](https://github.com/abkafi1234/MLatelier/actions/workflows/publish.yml/badge.svg)](https://github.com/abkafi1234/MLatelier/actions/workflows/publish.yml)
[![PyPI](https://img.shields.io/pypi/v/mlatelier.svg)](https://pypi.org/project/mlatelier/)
[![Python](https://img.shields.io/pypi/pyversions/mlatelier.svg)](https://pypi.org/project/mlatelier/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MLatelier is a Streamlit application that lets researchers and domain experts run complete machine-learning experiments — tabular, vision, and NLP — without writing a single line of model-training code. Upload data, pick models, click **Run**, and get a publication-ready diagnostics panel, optimised model export, LIME/SHAP explanations, and an optional Gemini AI assistant.

## Documentation

| Guide | |
|---|---|
| [Installation](docs/installation.md) | Install, extras, GPU setup, troubleshooting |
| [Quickstart](docs/quickstart.md) | Your first experiment in five minutes |
| [Tabular ML](docs/tutorial-tabular.md) | Classification, regression, multi-label, XAI, ablation |
| [NLP](docs/tutorial-nlp.md) | The three text tracks and when to use each |
| [Vision](docs/tutorial-vision.md) | Transfer learning and Grad-CAM |
| [API Reference](docs/api-reference.md) | Every public function |
| [Examples](docs/examples/) | Runnable scripts using the engines as a library |
| [Contributing](CONTRIBUTING.md) | Dev setup, tests, conventions |

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

Everything is included in that single install — every feature in the dashboard
works without a second `pip` command. MLatelier is driven from a browser by
people who do not write code, so a button that answers "install another package
first" is a broken button.

Contributors additionally want the test and build tooling:

```bash
pip install "mlatelier[dev]"    # pytest, pytest-cov, build, onnxruntime
```

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
| **Vision DL** | PyTorch, torchvision | Transfer learning on 12 architectures; GradCAM (9 conv backbones); live epoch progress |
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
- **Bayesian optimisation** — `BayesSearchCV` from scikit-optimize; winner selected by inner CV score, so holdout performance (reported for every model) never influences the choice
- **EDA panel** — summary stats, histograms, box plots, correlation heatmap, missing-value heatmap, data quality report
- **All models exported** — download `.joblib` + JSON metadata sidecar for any run model, not just the winner

---

### Tab 2 — Vision DL

**Workflow**

1. Provide an image dataset as a folder path (sub-folders = class names) or a `.zip` archive.
2. Select one or more architectures and configure training settings.
3. Click **Run Vision Pipeline**.

**Highlights**

- **Architectures (12)** — `ResNet18`, `ResNet50`, `VGG16`, `EfficientNet_B0`, `EfficientNet_V2_S`, `MobileNet_v3`, `DenseNet121`, `ConvNeXt_T`, `ConvNeXt_S`, `ViT_B_16`, `Swin_T`, `Swin_S`
- **Data split** — 60/20/20 train/validation/test, stratified by class
- **Fine-tuning strategy** — frozen backbone (fast prototyping) or full fine-tune
- **Live training curves** — epoch-by-epoch training and validation loss/accuracy
- **Post-run diagnostics** — confusion matrix, class distribution chart, sample image grid
- **GradCAM** — gradient-weighted class activation maps for incorrect predictions, on the 9 convolutional backbones (not ViT/Swin)
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
`ResNet18`, `ResNet50`, `VGG16`, `EfficientNet_B0`, `EfficientNet_V2_S`, `MobileNet_v3`, `DenseNet121`, `ConvNeXt_T`, `ConvNeXt_S`, `ViT_B_16`, `Swin_T`, `Swin_S`

These are the exact registry keys — an unrecognised name raises `ValueError`. Grad-CAM is available for the nine convolutional architectures; `ViT_B_16`, `Swin_T`, and `Swin_S` have no final convolutional feature map to attribute over.

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

Two locations, and it is worth knowing which is which:

| Written to | What lands there |
|---|---|
| `~/MLatelier/experiments/EXP_<timestamp>/` | Self-contained HTML report and generated curve images, archived per run |
| `~/MLatelier/models/` | A `.joblib` per trained model plus a JSON metadata sidecar (feature list, hyperparameters, task type, scores), and a `.onnx` file for every convertible model |

Offered as download buttons in the UI, not written to disk: the leaderboard CSV
and the LaTeX `booktabs` table.

### Checkpointing and resume

Vision transfer learning and HuggingFace fine-tuning are the only operations
that run for tens of minutes, and they are the ones worth protecting. After
every epoch MLatelier writes model, optimiser, and scheduler state to
`~/MLatelier/checkpoints/`. If training is interrupted — closed tab, OOM, power
cut — re-running the **same configuration** picks up from the last completed
epoch instead of starting over. Checkpoints are deleted automatically once
training finishes.

- **Vision DL** — toggle *Checkpoint each epoch* in the training panel (on by default)
- **NLP HuggingFace track** — always on; it is the longest operation in the framework

A checkpoint records a fingerprint of the run configuration. Change the learning
rate, architecture, epoch count, or seed and it will **not** resume — you get a
clean run rather than a silent continuation of a different experiment. Loss and
validation-score history carries across the resume, so the training curves are
continuous.

What is *not* resumable: progress through Bayesian-optimisation trials. Each
trial is short; the long full-training runs after the search are the ones that
checkpoint.

### ONNX export

Every convertible model gets an `.onnx` file alongside its `.joblib`, downloadable
from the **Download All Models → Download as ONNX** panel. The whole inference
pipeline — imputation, scaling, encoding, the estimator — is baked into one
graph, so you do not reimplement preprocessing on the target platform. The
training-only oversampling step is excluded.

Converters for XGBoost and LightGBM register automatically, and multi-label
(`MultiOutputClassifier`) models convert too. **CatBoost has no converter** and
remains `.joblib`-only; the UI names which models were skipped. Classifier
graphs emit a probability tensor rather than a ZipMap, so non-Python runtimes
read them directly.

The graph takes **one named input per feature column**, mirroring the
`ColumnTransformer` the pipeline is built around:

```python
import onnxruntime as ort, numpy as np

sess = ort.InferenceSession("MLatelier_Random_Forest.onnx")
feed = {i.name: X[i.name].to_numpy().astype(np.float32).reshape(-1, 1)
        for i in sess.get_inputs()}
label, proba = sess.run(None, feed)
```

> **The models directory is flat and filenames are keyed by model name**, so a
> later run overwrites an earlier run's export of the same model. Copy out
> anything you need to keep before re-running.

Every model that was trained is exported, not only the winner.

---

## Project Structure

```text
MLatelier/
├── pyproject.toml               # pip packaging config, extras, entry point
├── requirements.txt             # plain dependency list
├── README.md
├── LICENSE                      # MIT
├── CONTRIBUTING.md              # dev setup, conventions, PR process
├── CITATION.cff                 # "Cite this repository" metadata
├── .gitignore
│
├── docs/                        # documentation
│   ├── index.md                 # documentation home
│   ├── installation.md          # install, extras, GPU, troubleshooting
│   ├── quickstart.md            # first experiment in five minutes
│   ├── tutorial-tabular.md      # tabular tasks, XAI, ablation
│   ├── tutorial-nlp.md          # the three NLP tracks
│   ├── tutorial-vision.md       # transfer learning and Grad-CAM
│   ├── api-reference.md         # every public function
│   └── examples/                # runnable library-usage scripts
│
├── src/mlatelier/               # installed Python package
│   ├── __init__.py              # exposes __version__ = "1.1.0"
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
├── tests/                       # 390 tests total
│   ├── conftest.py              # shared Streamlit stub (installed before any test module)
│   ├── test_nlp_engine.py       # 136: every model, every config flag, DL track, LIME
│   ├── test_reporting.py        #  68: every render function including NLP renderers
│   ├── test_nlp_enhancements.py #  44: rich preprocessing, HF fine-tune, NLP visualisations
│   ├── test_tabular_engine.py   #  38: baseline, BO, multi-label, PDP, SHAP, LIME
│   ├── test_ai_context.py       #  27: AI tutor diagnostic signals and dataset health
│   ├── test_file_reading.py     #  20: CSV/Excel reading, sheet selection
│   ├── test_checkpoint.py       #  16: checkpoint save/load/fingerprint semantics
│   ├── test_vision_analysis.py  #  15: dataset summary, device detection, model loading
│   ├── test_inference_engine.py #  12: tabular and vision inference
│   ├── test_onnx_export.py      #  10: ONNX round-trip and converter registration
│   └── test_checkpoint_resume.py#   4: crash mid-training, resume from last epoch
│
└── .github/
    ├── ISSUE_TEMPLATE/          # bug report and feature request forms
    ├── PULL_REQUEST_TEMPLATE.md
    └── workflows/
        └── publish.yml          # CI: test (+coverage) → build → publish on version tag
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

**390 tests** in total. CI runs the full suite on Python 3.9, 3.11, 3.12, 3.13,
and 3.14, and reports coverage in the workflow summary. Measured on Python
3.14: **390 passed, 4 skipped, 0 failed, 52% line coverage**.

Coverage by module — the Streamlit UI layer (`app.py`) is driven through the
browser and is not unit-tested, which is most of the uncovered total:

| Module | Coverage |
|---|---|
| `file_utils.py` | 93% |
| `nlp_engine.py` | 82% |
| `checkpoint.py` | 78% |
| `reporting.py` | 76% |
| `inference_engine.py` | 72% |
| `tabular_engine.py` | 72% |
| `vision_engine.py` | 37% |
| `app.py`, `tracker.py`, `utils.py` | 0% |

Excluding `app.py`, engine coverage is 68%.

| File | Tests | Notes |
|---|---|---|
| `test_nlp_engine.py` | 136 | Every model, every config flag, DL track, LIME. LIME tests skip when `lime` is not installed |
| `test_reporting.py` | 68 | All render functions including NLP renderers |
| `test_nlp_enhancements.py` | 44 | Rich preprocessing, HF fine-tune (mocked), NLP visualisations |
| `test_tabular_engine.py` | 38 | Baseline, BO, multi-label, PDP, SHAP, LIME |
| `test_ai_context.py` | 27 | AI tutor diagnostic signals, dataset health, degenerate state |
| `test_file_reading.py` | 20 | CSV/Excel formats, sheet selection |
| `test_checkpoint.py` | 16 | Save/load, fingerprint mismatch, corrupt files, optimiser state |
| `test_vision_analysis.py` | 15 | Dataset summary, device detection, model loading |
| `test_inference_engine.py` | 12 | Tabular and vision inference |
| `test_onnx_export.py` | 10 | ONNX round-trip, XGBoost/LightGBM converters, graceful fallback |
| `test_checkpoint_resume.py` | 4 | Crash mid-training, resume from last epoch |
| **Total** | **390** | |

Run with coverage locally:

```bash
pip install -e ".[dev]"
pytest --cov=mlatelier --cov-report=term-missing
```

---

## Publishing to PyPI (GitHub Actions)

The included workflow at [.github/workflows/publish.yml](.github/workflows/publish.yml) handles testing, building, and publishing automatically.

### How it works

| Trigger | What runs |
|---|---|
| Push to `main` or any PR | `test` job — runs suite on Python 3.9, 3.11, 3.12, 3.13, 3.14 |
| Push of a version tag `v*.*.*` | `test` → `build` → `publish` to PyPI |

### One-time PyPI setup (Trusted Publisher — no token needed)

1. Log in to [pypi.org](https://pypi.org) → **Your projects** → select or create `mlatelier`.
2. Go to **Settings → Publishing → Add a new publisher** → choose **GitHub Actions**.
3. Fill in:
   - Owner: `abkafi1234`
   - Repository: `MLatelier`
   - Workflow filename: `publish.yml`
   - Environment name: `pypi`
4. Save.

### How to release a new version

Keep the version in step across all three files — `pyproject.toml`,
`src/mlatelier/__init__.py`, and `CITATION.cff`.

```bash
# 1. Bump the version in all three files (example: 1.1.0 -> 1.2.0)
# 2. Commit and push
git add pyproject.toml src/mlatelier/__init__.py CITATION.cff
git commit -m "chore: bump version to v1.2.0"
git push origin main

# 3. Tag and push the tag — this triggers the publish job
git tag v1.2.0
git push origin v1.2.0
```

The workflow will run tests, build `dist/mlatelier-1.2.0-py3-none-any.whl` and the sdist, then upload both to PyPI. The new version is live within seconds.

> PyPI refuses to overwrite an existing version, so the tag must be a version that has never been published.

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
