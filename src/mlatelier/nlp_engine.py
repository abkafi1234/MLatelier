"""
nlp_engine.py — NLP text-classification engine for MLatelier.

ML track  : TF-IDF  +  curated sklearn classifiers  +  BayesSearchCV
DL track  : Sentence-Transformer embeddings  +  sklearn classifiers  (no fine-tuning)
"""

from __future__ import annotations

import os
import time
import warnings
import numpy as np
from typing import Callable, Optional

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (
    cross_val_score, StratifiedKFold, train_test_split,
)
from sklearn.metrics import (
    classification_report, f1_score, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import ttest_rel

# ── Optional heavy dependencies ───────────────────────────────────────────────

try:
    from xgboost import XGBClassifier as _XGB
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from skopt import BayesSearchCV
    from skopt.space import Integer, Real, Categorical
    _HAS_SKOPT = True
except ImportError:
    _HAS_SKOPT = False
    # Placeholders so type annotations don't break
    class Integer:  # type: ignore
        def __init__(self, *a, **k): pass
    class Real:  # type: ignore
        def __init__(self, *a, **k): pass
    class Categorical:  # type: ignore
        def __init__(self, *a, **k): pass

try:
    import nltk as _nltk
    from nltk.corpus import stopwords as _sw_corpus
    from nltk.stem import PorterStemmer as _PS
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False

try:
    from lime.lime_text import LimeTextExplainer
    _HAS_LIME = True
except ImportError:
    _HAS_LIME = False

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False

try:
    import transformers as _transformers_mod  # noqa: F401
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


# ── Device detection ───────────────────────────────────────────────────────────

def get_nlp_device() -> dict:
    """Return device-info dict matching vision_engine.get_device_info() shape."""
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "type": "cuda",
                "name": props.name,
                "memory_gb": round(props.total_memory / 1e9, 1),
                "cuda_version": torch.version.cuda or "?",
            }
        try:
            if torch.backends.mps.is_available():
                return {"type": "mps"}
        except AttributeError:
            pass
    except ImportError:
        pass
    return {"type": "cpu"}


# ── NLTK helpers ───────────────────────────────────────────────────────────────

def _ensure_stopwords() -> list[str]:
    if not _HAS_NLTK:
        return []
    try:
        return list(_sw_corpus.words("english"))
    except LookupError:
        try:
            _nltk.download("stopwords", quiet=True)
            return list(_sw_corpus.words("english"))
        except Exception:
            return []


class _StemTokenizer:
    """Porter-stemmer tokenizer compatible with sklearn's TF-IDF."""

    def __init__(self):
        self._stemmer = _PS()

    def __call__(self, doc: str) -> list[str]:
        from sklearn.feature_extraction.text import CountVectorizer as _CV
        analyzer = _CV().build_analyzer()
        return [self._stemmer.stem(w) for w in analyzer(doc)]

    def __getstate__(self):
        return {"_stemmer": self._stemmer}

    def __setstate__(self, state):
        self._stemmer = state["_stemmer"]


# ── Model registries ───────────────────────────────────────────────────────────

def get_nlp_ml_models() -> dict[str, tuple]:
    """Return {name: (class, default_kwargs)} for the ML track."""
    models = {
        "Logistic Regression": (
            LogisticRegression,
            {"max_iter": 1000, "solver": "lbfgs"},
        ),
        "Linear SVC": (LinearSVC, {"max_iter": 2000}),
        "SGD Classifier": (SGDClassifier, {"max_iter": 1000, "n_jobs": -1}),
        "Multinomial NB": (MultinomialNB, {}),
        "Complement NB": (ComplementNB, {}),
        "Random Forest": (RandomForestClassifier, {"n_estimators": 100, "n_jobs": -1}),
    }
    if _HAS_XGB:
        models["XGBoost"] = (
            _XGB,
            {"n_estimators": 100, "eval_metric": "logloss", "verbosity": 0,
             "device": "cpu"},
        )
    return models


def get_nlp_dl_classifiers() -> dict[str, object]:
    """Classifiers for the DL track (operate on numeric ST embeddings)."""
    clfs = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="lbfgs"),
        "Linear SVC":          LinearSVC(max_iter=2000),
        "SGD Classifier":      SGDClassifier(max_iter=1000, n_jobs=-1),
        "Random Forest":       RandomForestClassifier(n_estimators=100, n_jobs=-1),
    }
    if _HAS_XGB:
        clfs["XGBoost"] = _XGB(n_estimators=100, eval_metric="logloss",
                                verbosity=0, device="cpu")
    return clfs


NLP_ST_MODELS: list[str] = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "paraphrase-MiniLM-L6-v2",
]

HF_PRESET_MODELS: list[str] = [
    "distilbert-base-uncased",
    "bert-base-uncased",
    "roberta-base",
    "distilroberta-base",
    "xlm-roberta-base",
    "albert-base-v2",
]


# ── Pipeline builder (ML track) ────────────────────────────────────────────────

def build_tfidf_pipeline(
    model_name: str,
    *,
    use_stopwords: bool = True,
    use_stemming: bool = False,
    ngram_max: int = 2,
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 1.0,
    sublinear_tf: Optional[bool] = None,
    vectorizer_type: str = "tfidf",
) -> Pipeline:
    """Build a vectorizer → classifier Pipeline for the ML track.

    vectorizer_type: 'tfidf' (TfidfVectorizer) or 'count' (CountVectorizer).
    sublinear_tf: None = auto (True for non-NB TF-IDF; ignored for Count).
    """
    stop_words = None
    tokenizer = None

    if use_stopwords:
        sw = _ensure_stopwords()
        stop_words = sw if sw else "english"

    if use_stemming and _HAS_NLTK:
        tokenizer = _StemTokenizer()

    is_nb = model_name in ("Multinomial NB", "Complement NB")

    shared_kw = dict(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        stop_words=stop_words,
        tokenizer=tokenizer,
        strip_accents="unicode",
        min_df=min_df,
        max_df=max_df,
    )

    if vectorizer_type == "count":
        vec = CountVectorizer(**shared_kw)
        step_name = "count"
    else:
        _sublinear = (not is_nb) if sublinear_tf is None else (sublinear_tf and not is_nb)
        vec = TfidfVectorizer(**shared_kw, sublinear_tf=_sublinear)
        step_name = "tfidf"

    registry = get_nlp_ml_models()
    cls_class, default_kw = registry[model_name]
    classifier = cls_class(**default_kw)

    return Pipeline([(step_name, vec), ("clf", classifier)])


# ── Sentence-Transformer encoder ───────────────────────────────────────────────

def encode_with_transformer(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> np.ndarray:
    """Encode texts → (N, D) float32 embedding matrix."""
    if not _HAS_ST:
        raise ImportError(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers"
        )
    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return embeddings.astype(np.float32)


# ── BayesOpt search spaces ─────────────────────────────────────────────────────

def _ml_spaces(vec_step: str = "tfidf") -> dict:
    if not _HAS_SKOPT:
        return {}
    _mf = f"{vec_step}__max_features"
    spaces = {
        "Logistic Regression": {
            "clf__C": Real(1e-3, 100.0, prior="log-uniform"),
            _mf:      Integer(1000, 20000),
            # ngram_range excluded: skopt Categorical with tuples is buggy in v0.10
        },
        "Linear SVC": {
            "clf__C": Real(1e-3, 100.0, prior="log-uniform"),
            _mf:      Integer(1000, 20000),
        },
        "SGD Classifier": {
            "clf__alpha": Real(1e-5, 1e-1, prior="log-uniform"),
            "clf__loss":  Categorical(["hinge", "log_loss", "modified_huber"]),
            _mf:          Integer(1000, 20000),
        },
        "Multinomial NB": {
            "clf__alpha": Real(1e-3, 10.0, prior="log-uniform"),
            _mf:          Integer(1000, 20000),
        },
        "Complement NB": {
            "clf__alpha": Real(1e-3, 10.0, prior="log-uniform"),
            _mf:          Integer(1000, 20000),
        },
        "Random Forest": {
            "clf__n_estimators": Integer(50, 300),
            "clf__max_depth":    Integer(3, 20),
            _mf:                 Integer(1000, 10000),
        },
        "XGBoost": {
            "clf__n_estimators":  Integer(50, 300),
            "clf__max_depth":     Integer(3, 10),
            "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
            _mf:                  Integer(1000, 10000),
        },
    }
    return spaces


def _dl_spaces() -> dict:
    if not _HAS_SKOPT:
        return {}
    return {
        "Logistic Regression": {"clf__C": Real(1e-3, 100.0, prior="log-uniform")},
        "Linear SVC":          {"clf__C": Real(1e-3, 100.0, prior="log-uniform")},
        "SGD Classifier":      {"clf__alpha": Real(1e-5, 1e-1, prior="log-uniform")},
        "Random Forest": {
            "clf__n_estimators": Integer(50, 300),
            "clf__max_depth":    Integer(3, 20),
        },
        "XGBoost": {
            "clf__n_estimators":  Integer(50, 300),
            "clf__max_depth":     Integer(3, 10),
            "clf__learning_rate": Real(0.01, 0.3, prior="log-uniform"),
        },
    }


# ── Baseline cross-validation ──────────────────────────────────────────────────

def run_nlp_baseline(
    texts: list[str],
    labels,
    model_names: list[str],
    *,
    track: str = "ml",
    use_stopwords: bool = True,
    use_stemming: bool = False,
    ngram_max: int = 2,
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 1.0,
    sublinear_tf: Optional[bool] = None,
    vectorizer_type: str = "tfidf",
    st_model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    cv_folds: int = 5,
    random_state: int = 42,
    progress_bar=None,
    status_text=None,
    model_callback: Optional[Callable] = None,
) -> tuple[dict, list[str]]:
    """5-fold CV baselines for every selected model. Returns (results_dict, class_names)."""
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names: list[str] = le.classes_.tolist()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    X_embed: Optional[np.ndarray] = None
    if track == "dl":
        if status_text:
            status_text.info(f"Encoding texts with {st_model_name}…")
        X_embed = encode_with_transformer(texts, st_model_name, device=device)

    n = len(model_names)
    results: dict = {}

    for i, name in enumerate(model_names):
        t0 = time.perf_counter()
        try:
            if track == "ml":
                pipe = build_tfidf_pipeline(
                    name, use_stopwords=use_stopwords,
                    use_stemming=use_stemming,
                    ngram_max=ngram_max, max_features=max_features,
                    min_df=min_df, max_df=max_df,
                    sublinear_tf=sublinear_tf,
                    vectorizer_type=vectorizer_type,
                )
                X = texts
            else:
                clf = get_nlp_dl_classifiers()[name]
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
                X = X_embed

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fold_f1s = cross_val_score(
                    pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=1)

            mean_f1 = float(np.mean(fold_f1s))
            std_f1  = float(np.std(fold_f1s))
            elapsed = round(time.perf_counter() - t0, 2)

            results[name] = {
                "mean": mean_f1, "std": std_f1,
                "fold_scores": fold_f1s.tolist(),
                "time_s": elapsed, "class_names": class_names,
            }
            if model_callback:
                model_callback(name, mean_f1, std_f1, elapsed, fold_f1s.tolist())

        except Exception as exc:
            results[name] = {
                "mean": float("nan"), "std": 0.0,
                "fold_scores": [], "time_s": 0.0,
                "error": str(exc), "class_names": class_names,
            }

        if progress_bar:
            progress_bar.progress((i + 1) / n * 0.4)

    return results, class_names


# ── Bayesian Optimisation ──────────────────────────────────────────────────────

def run_nlp_optimization(
    texts: list[str],
    labels,
    model_names: list[str],
    baseline_results: dict,
    *,
    track: str = "ml",
    use_stopwords: bool = True,
    use_stemming: bool = False,
    ngram_max: int = 2,
    max_features: int = 5000,
    min_df: int = 2,
    max_df: float = 1.0,
    sublinear_tf: Optional[bool] = None,
    vectorizer_type: str = "tfidf",
    st_model_name: str = "all-MiniLM-L6-v2",
    device: str = "cpu",
    cv_folds: int = 5,
    n_iter: int = 20,
    random_state: int = 42,
    export_model: bool = True,
    export_dir: Optional[str] = None,
    progress_bar=None,
    status_text=None,
) -> tuple:
    """BayesSearchCV per model; winner selected by inner-CV score (best_score_).

    Returns:
        (results_list, best_acc, avg_acc, improvement_pct, p_val_str,
         winner_params, class_report, class_names, winner_curves)
    """
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names: list[str] = le.classes_.tolist()

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    X_embed: Optional[np.ndarray] = None
    if track == "dl":
        if status_text:
            status_text.info(f"Encoding texts with {st_model_name}…")
        X_embed = encode_with_transformer(texts, st_model_name, device=device)

    ml_spaces = _ml_spaces(vec_step="count" if vectorizer_type == "count" else "tfidf")
    dl_spaces = _dl_spaces()

    n = len(model_names)
    results_list: list[dict] = []
    best_cv_score = -float("inf")
    winner_name: Optional[str] = None
    winner_pipe = None
    winner_params: dict = {}
    bo_scores_all: list[float] = []
    fold_scores_all: dict = {}
    timing_data: dict = {}

    best_baseline_name: Optional[str] = max(
        (m for m in model_names
         if m in baseline_results
         and not np.isnan(baseline_results[m].get("mean", float("nan")))),
        key=lambda m: baseline_results[m].get("mean", -1.0),
        default=None,
    )

    for i, name in enumerate(model_names):
        if status_text:
            status_text.info(f"Optimising {name} ({i + 1}/{n})…")
        t0 = time.perf_counter()

        try:
            if track == "ml":
                pipe = build_tfidf_pipeline(
                    name, use_stopwords=use_stopwords,
                    use_stemming=use_stemming,
                    ngram_max=ngram_max, max_features=max_features,
                    min_df=min_df, max_df=max_df,
                    sublinear_tf=sublinear_tf,
                    vectorizer_type=vectorizer_type,
                )
                space = ml_spaces.get(name, {})
                X = texts
            else:
                clf = get_nlp_dl_classifiers()[name]
                pipe = Pipeline([("scaler", StandardScaler()), ("clf", clf)])
                space = dl_spaces.get(name, {})
                X = X_embed

            bl_mean = baseline_results.get(name, {}).get("mean", float("nan"))

            if _HAS_SKOPT and space:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    opt = BayesSearchCV(
                        pipe, space,
                        n_iter=max(n_iter, 5),
                        cv=cv, scoring="f1_macro",
                        n_jobs=1, random_state=random_state,
                        refit=True, return_train_score=False,
                    )
                    opt.fit(X, y)
                sel_score = float(opt.best_score_)
                best_params = dict(opt.best_params_)
                fitted_pipe = opt.best_estimator_
                bo_scores_all.append(sel_score)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    pipe.fit(X, y)
                sel_score = bl_mean
                best_params = {}
                fitted_pipe = pipe

            elapsed = round(time.perf_counter() - t0, 2)
            timing_data[name] = elapsed

            # Final fold scores with fitted pipeline
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                final_folds = cross_val_score(
                    fitted_pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=1)
            fold_scores_all[name] = final_folds.tolist()

            _cmp = sel_score if not np.isnan(sel_score) else -float("inf")
            if _cmp > best_cv_score:
                best_cv_score = _cmp
                winner_name = name
                winner_pipe = fitted_pipe
                winner_params = best_params

            results_list.append({
                "Algorithm": name,
                "Track":    track.upper(),
                "Baseline Macro F1":        bl_mean,
                "Selection CV Macro F1":    sel_score,
                "Optimized (CV mean)":      float(np.mean(final_folds)),
                "Training Time (s)":        elapsed,
            })

        except Exception as exc:
            bl_mean = baseline_results.get(name, {}).get("mean", float("nan"))
            results_list.append({
                "Algorithm": name, "Track": track.upper(),
                "Baseline Macro F1":     bl_mean,
                "Selection CV Macro F1": float("nan"),
                "Optimized (CV mean)":   float("nan"),
                "Training Time (s)":     0.0,
                "Error":                 str(exc),
            })

        if progress_bar:
            progress_bar.progress(0.4 + (i + 1) / n * 0.5)

    # ── Evaluate winner on held-out split (D_test, consulted exactly once) ─────
    class_report: dict = {}
    winner_curves: dict = {}
    best_acc = float("nan")
    avg_acc = float("nan")
    imp = 0.0
    p_val = "N/A"
    exported_path: Optional[str] = None

    if winner_pipe is not None:
        X_winner = texts if track == "ml" else X_embed
        X_tr, X_te, y_tr, y_te = train_test_split(
            X_winner, y, test_size=0.2,
            random_state=random_state, stratify=y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            winner_pipe.fit(X_tr, y_tr)
            y_pred = winner_pipe.predict(X_te)

        best_acc = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
        all_cv_means = [np.mean(v) for v in fold_scores_all.values() if v]
        avg_acc = float(np.mean(all_cv_means)) if all_cv_means else best_acc

        # Improvement over best baseline
        if best_baseline_name and best_baseline_name in baseline_results:
            bl_b = baseline_results[best_baseline_name].get("mean", best_acc)
            imp = max(0.0, (best_acc - bl_b) * 100)

        # Paired t-test: winner folds vs best-baseline folds
        wf = fold_scores_all.get(winner_name, [])
        bf = baseline_results.get(best_baseline_name, {}).get("fold_scores", [])
        if len(wf) == len(bf) and len(wf) > 1:
            try:
                _, pv = ttest_rel(wf, bf)
                p_val = f"{pv:.4f}"
            except Exception:
                pass

        cm = confusion_matrix(y_te, y_pred)
        cr = classification_report(
            y_te, y_pred, target_names=class_names,
            output_dict=True, zero_division=0)
        class_report = cr
        class_report["confusion_matrix"] = cm.tolist()
        class_report["confusion_matrix_labels"] = class_names

        winner_curves = {
            "bo_scores":       bo_scores_all or None,
            "fold_scores":     fold_scores_all.get(winner_name, []),
            "timing_data":     timing_data,
            "confusion_matrix": {
                "matrix": cm.tolist(),
                "labels": class_names,
            },
            "winner_name":     winner_name,
            "track":           track,
        }

        if export_model and export_dir:
            os.makedirs(export_dir, exist_ok=True)
            import joblib
            fname = f"MLatelier_NLP_{winner_name.replace(' ', '_')}.joblib"
            exported_path = os.path.join(export_dir, fname)
            joblib.dump(winner_pipe, exported_path)
            winner_curves["exported_model_path"] = exported_path

    if progress_bar:
        progress_bar.progress(1.0)

    return (
        results_list, best_acc, avg_acc, imp, p_val,
        winner_params, class_report, class_names, winner_curves,
    )


# ── LIME text explanation ─────────────────────────────────────────────────────

def compute_nlp_lime(
    pipeline,
    texts: list[str],
    class_names: list[str],
    *,
    sample_idx: int = 0,
    num_features: int = 15,
    num_samples: int = 300,
) -> dict:
    """Compute LIME explanation for a single text sample (ML track only)."""
    if not _HAS_LIME:
        return {"error": "lime not installed — run: pip install lime"}

    text = texts[sample_idx]

    # Ensure the pipeline supports predict_proba (wrap LinearSVC if needed)
    predict_fn: Optional[Callable] = None
    try:
        _ = pipeline.predict_proba([text])

        def predict_fn(batch):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return pipeline.predict_proba(batch)

    except AttributeError:
        # LinearSVC / SGD(hinge): wrap with calibrated classifier
        # LIME needs probability output — we calibrate on the text data
        try:
            calib = CalibratedClassifierCV(pipeline, cv="prefit")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                calib.fit(texts[:1], [0])  # dummy — pipeline is prefit
        except Exception:
            return {
                "error": (
                    "Model does not support probability output. "
                    "Use Logistic Regression, Multinomial NB, Complement NB, "
                    "Random Forest, or XGBoost for LIME explanations."
                )
            }

        def predict_fn(batch):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return calib.predict_proba(batch)

    try:
        explainer = LimeTextExplainer(class_names=class_names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            exp = explainer.explain_instance(
                text, predict_fn,
                num_features=num_features,
                num_samples=num_samples,
            )

        proba = predict_fn([text])[0]
        pred_class_idx = int(np.argmax(proba))
        pred_class = (class_names[pred_class_idx]
                      if pred_class_idx < len(class_names)
                      else str(pred_class_idx))

        explanation = exp.as_list(label=pred_class_idx)
        return {
            "text":            text,
            "predicted_class": pred_class,
            "explanation":     explanation,  # [(word, weight), …]
            "class_names":     class_names,
        }

    except Exception as exc:
        return {"error": f"LIME failed: {exc}"}


# ── Top TF-IDF features (linear models) ──────────────────────────────────────

def get_top_tfidf_features(pipeline, class_names: list[str], n: int = 20) -> dict:
    """Extract top positive/negative vectorizer features for linear classifiers."""
    try:
        tfidf = pipeline.named_steps.get("tfidf") or pipeline.named_steps.get("count")
        clf   = pipeline.named_steps.get("clf")
        if tfidf is None or clf is None:
            return {}

        vocab = np.array(tfidf.get_feature_names_out())

        # Logistic Regression, LinearSVC, SGD → .coef_
        if not hasattr(clf, "coef_"):
            return {}

        coef = np.array(clf.coef_)
        if coef.ndim == 1:
            coef = coef[np.newaxis, :]

        result = {}
        for i, row in enumerate(coef):
            label = class_names[i] if i < len(class_names) else f"Class {i}"
            top_pos = np.argsort(row)[-n:][::-1]
            top_neg = np.argsort(row)[:n]
            result[label] = {
                "positive": [(vocab[j], float(row[j])) for j in top_pos],
                "negative": [(vocab[j], float(row[j])) for j in top_neg],
            }
        return result
    except Exception:
        return {}


# ── HuggingFace Transformer fine-tuning ───────────────────────────────────────

def run_transformer_finetune(
    texts: list[str],
    labels,
    model_id: str = "distilbert-base-uncased",
    *,
    epochs: int = 3,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    max_seq_len: int = 128,
    warmup_ratio: float = 0.1,
    freeze_backbone: bool = False,
    random_state: int = 42,
    export_dir: Optional[str] = None,
    progress_bar=None,
    status_text=None,
) -> tuple:
    """Fine-tune any HuggingFace sequence-classification model end-to-end.

    Returns the same 9-tuple as run_nlp_optimization() for drop-in compatibility:
        (results_list, best_acc, avg_acc, improvement_pct, p_val_str,
         winner_params, class_report, class_names, winner_curves)
    """
    if not _HAS_TRANSFORMERS:
        raise ImportError(
            "transformers is not installed. "
            "Run: pip install transformers"
        )

    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim import AdamW
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup,
    )

    t0 = time.perf_counter()

    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_names: list[str] = le.classes_.tolist()
    n_classes = len(class_names)

    dev_info = get_nlp_device()
    _dev_type = dev_info.get("type", "cpu")
    device = torch.device(_dev_type if _dev_type in ("cuda", "mps") else "cpu")

    if status_text:
        status_text.info(f"Loading tokenizer: {model_id} …")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    X_tr, X_te, y_tr, y_te = train_test_split(
        texts, y, test_size=0.2, random_state=random_state,
        stratify=y,
    )

    def _encode(txts: list[str]) -> dict:
        return tokenizer(
            list(txts), truncation=True, padding=True,
            max_length=max_seq_len, return_tensors="pt",
        )

    if status_text:
        status_text.info("Tokenising texts …")
    train_enc = _encode(X_tr)
    test_enc  = _encode(X_te)

    train_ds = TensorDataset(
        train_enc["input_ids"],
        train_enc["attention_mask"],
        torch.tensor(list(y_tr), dtype=torch.long),
    )
    test_ds = TensorDataset(
        test_enc["input_ids"],
        test_enc["attention_mask"],
        torch.tensor(list(y_te), dtype=torch.long),
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    if status_text:
        status_text.info(f"Loading model: {model_id} …")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=n_classes, ignore_mismatched_sizes=True,
    )

    if freeze_backbone:
        for name, param in model.named_parameters():
            if "classifier" not in name and "pooler" not in name:
                param.requires_grad = False

    model = model.to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        # Freeze left nothing — fall back to all parameters
        trainable = list(model.parameters())
    optimizer = AdamW(trainable, lr=learning_rate, eps=1e-8)

    total_steps  = len(train_loader) * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    train_losses: list[float] = []

    for ep in range(epochs):
        if status_text:
            status_text.info(f"Fine-tuning epoch {ep + 1}/{epochs} …")
        model.train()
        ep_loss = 0.0
        for step, (input_ids, attn_mask, batch_y) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attn_mask = attn_mask.to(device)
            batch_y   = batch_y.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask,
                labels=batch_y,
            )
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            ep_loss += loss.item()

            if progress_bar:
                frac = (ep * len(train_loader) + step + 1) / (epochs * len(train_loader))
                progress_bar.progress(min(frac * 0.9, 0.9))

        train_losses.append(ep_loss / max(len(train_loader), 1))

    # ── Evaluation ────────────────────────────────────────────────────────────
    if status_text:
        status_text.info("Evaluating on held-out test set …")
    model.eval()
    all_preds: list[int] = []
    all_true:  list[int] = []

    with torch.no_grad():
        for input_ids, attn_mask, batch_y in test_loader:
            logits = model(
                input_ids=input_ids.to(device),
                attention_mask=attn_mask.to(device),
            ).logits
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_true.extend(batch_y.tolist())

    best_acc = float(
        f1_score(all_true, all_preds, average="macro", zero_division=0))

    cm = confusion_matrix(all_true, all_preds)
    cr = classification_report(
        all_true, all_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cr["confusion_matrix"]        = cm.tolist()
    cr["confusion_matrix_labels"] = class_names

    elapsed = round(time.perf_counter() - t0, 2)

    results_list = [{
        "Algorithm":             model_id,
        "Track":                 "HF-FINETUNE",
        "Baseline Macro F1":     float("nan"),
        "Selection CV Macro F1": best_acc,
        "Optimized (CV mean)":   best_acc,
        "Training Time (s)":     elapsed,
    }]

    winner_params = {
        "model_id":        model_id,
        "epochs":          epochs,
        "batch_size":      batch_size,
        "learning_rate":   learning_rate,
        "max_seq_len":     max_seq_len,
        "freeze_backbone": freeze_backbone,
        "warmup_ratio":    warmup_ratio,
    }

    winner_curves = {
        "bo_scores":        None,
        "fold_scores":      [],
        "timing_data":      {model_id: elapsed},
        "confusion_matrix": {"matrix": cm.tolist(), "labels": class_names},
        "winner_name":      model_id,
        "track":            "hf-finetune",
        "train_losses":     train_losses,
    }

    if export_dir:
        os.makedirs(export_dir, exist_ok=True)
        safe_name = model_id.replace("/", "_")
        out_path = os.path.join(export_dir, f"MLatelier_HF_{safe_name}")
        model.save_pretrained(out_path)
        tokenizer.save_pretrained(out_path)
        winner_curves["exported_model_path"] = out_path

    if progress_bar:
        progress_bar.progress(1.0)

    return (
        results_list, best_acc, best_acc, 0.0, "N/A",
        winner_params, cr, class_names, winner_curves,
    )

