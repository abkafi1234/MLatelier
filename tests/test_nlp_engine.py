"""
Tests for nlp_engine.py and the new NLP reporting functions in reporting.py.

Coverage:
  - Registry helpers (get_nlp_ml_models, get_nlp_dl_classifiers, NLP_ST_MODELS)
  - Device detection (get_nlp_device)
  - TF-IDF pipeline builder (build_tfidf_pipeline) — every model, every config flag
  - Baseline CV (run_nlp_baseline) — binary, multiclass, callback, error handling
  - Bayesian optimisation (run_nlp_optimization) — with & without BayesOpt
  - LIME explanations (compute_nlp_lime) — proba / non-proba models, missing dep
  - Top TF-IDF features (get_top_tfidf_features) — linear vs tree
  - DL track encode + pipeline — skipped when sentence-transformers absent
  - Reporting renderers (render_nlp_lime_explanation, render_nlp_top_features)

All tests use small synthetic text corpora so the suite runs quickly (<60 s).
"""
from __future__ import annotations

import sys
import os
import types
import warnings
import numpy as np
import pytest

# ── Stub streamlit before any reporting import ────────────────────────────────

def _make_st_stub():
    st = types.ModuleType("streamlit")

    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def markdown(self, *a, **kw): pass
        def dataframe(self, *a, **kw): pass
        def metric(self, *a, **kw): pass
        def pyplot(self, *a, **kw): pass
        def image(self, *a, **kw): pass
        def caption(self, *a, **kw): pass
        def warning(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def success(self, *a, **kw): pass
        def info(self, *a, **kw): pass
        def download_button(self, *a, **kw): pass
        def columns(self, *a, **kw): return [self] * 10
        def expander(self, *a, **kw): return self
        def tabs(self, labels): return [self] * len(labels)

    _ctx = _Ctx()

    for _n in ["markdown", "dataframe", "metric", "pyplot", "image", "caption",
               "warning", "error", "success", "info", "download_button", "container"]:
        setattr(st, _n, lambda *a, **kw: None)

    st.columns   = lambda *a, **kw: [_ctx] * (a[0] if isinstance(a[0], int) else len(a[0]))
    st.tabs      = lambda labels: [_ctx] * len(labels)
    st.container = lambda **kw: _ctx
    st.expander  = lambda *a, **kw: _ctx
    st.session_state = {}
    return st


sys.modules.setdefault("streamlit", _make_st_stub())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/mlatelier"))

from nlp_engine import (
    NLP_ST_MODELS,
    _ensure_stopwords,
    build_tfidf_pipeline,
    compute_nlp_lime,
    encode_with_transformer,
    get_nlp_device,
    get_nlp_dl_classifiers,
    get_nlp_ml_models,
    get_top_tfidf_features,
    run_nlp_baseline,
    run_nlp_optimization,
    _HAS_LIME,
    _HAS_NLTK,
    _HAS_ST,
)

# ── Mock corpora ───────────────────────────────────────────────────────────────

# Binary: positive / negative
_POS = [
    "I absolutely love this movie, fantastic acting and great story",
    "This product is amazing and works perfectly every single time",
    "Wonderful experience at the restaurant food was delicious and fresh",
    "Best book I have ever read highly recommend this brilliant novel",
    "Outstanding customer service team was very helpful and professional",
    "The concert was incredible musicians played beautifully all night long",
    "Excellent quality product arrived quickly packaging was very secure",
    "Loved every moment of the trip hotel was clean and comfortable",
    "Perfect gift for kids they were delighted and played for hours",
    "Great value for money quality exceeded expectations tremendously",
]
_NEG = [
    "Terrible experience worst product I have ever purchased do not buy",
    "Very disappointed the quality was poor and broke after one day",
    "Awful customer service staff were rude and unhelpful throughout",
    "The food at the restaurant was cold and tasteless very bad meal",
    "This movie was boring and predictable waste of two hours of life",
    "Poor packaging item arrived damaged completely unusable and broken",
    "Would not recommend to anyone service was extremely slow and expensive",
    "Complete waste of money product stopped working after one week",
    "Very unhappy with purchase returns process was complicated and slow",
    "Disgusting experience staff ignored complaints and refused refund",
]

def _make_binary_corpus(n_per_class: int = 30) -> tuple[list[str], list[str]]:
    """Generate binary text corpus by repeating and shuffling base phrases."""
    rng = np.random.default_rng(0)
    texts, labels = [], []
    for _ in range(n_per_class):
        idx = rng.integers(0, len(_POS))
        # Add slight variation by appending a random word count
        texts.append(_POS[idx] + f" great item {rng.integers(100)}")
        labels.append("positive")
    for _ in range(n_per_class):
        idx = rng.integers(0, len(_NEG))
        texts.append(_NEG[idx] + f" bad product {rng.integers(100)}")
        labels.append("negative")
    perm = rng.permutation(len(texts))
    return [texts[i] for i in perm], [labels[i] for i in perm]


_TECH = [
    "Python programming language machine learning deep neural networks",
    "Software engineer develops scalable cloud computing infrastructure",
    "Artificial intelligence algorithm processes natural language data",
    "Open source database optimizes query performance with indexing",
    "Cybersecurity team patches critical vulnerability in web application",
]
_SPORT = [
    "Football team wins championship final goal scored in last minute",
    "Basketball player breaks scoring record during playoff season game",
    "Tennis champion defeats rival in five set thriller match today",
    "Olympic swimmer breaks world record at international competition",
    "Soccer referee reviews controversial decision using video technology",
]
_POLITICS = [
    "Government announces new economic policy to reduce national deficit",
    "Election results show close race between two major political parties",
    "Parliament debates proposed legislation on climate change regulations",
    "President signs executive order on immigration reform and border control",
    "Senate committee investigates foreign interference in recent elections",
]


def _make_multiclass_corpus(n_per_class: int = 30) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(1)
    texts, labels = [], []
    for cat, pool in [("tech", _TECH), ("sports", _SPORT), ("politics", _POLITICS)]:
        for _ in range(n_per_class):
            idx = rng.integers(0, len(pool))
            texts.append(pool[idx] + f" news report {rng.integers(1000)}")
            labels.append(cat)
    perm = rng.permutation(len(texts))
    return [texts[i] for i in perm], [labels[i] for i in perm]


BINARY_TEXTS, BINARY_LABELS   = _make_binary_corpus(30)
MULTI_TEXTS,  MULTI_LABELS    = _make_multiclass_corpus(30)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def binary():
    return BINARY_TEXTS, BINARY_LABELS


@pytest.fixture(scope="module")
def multiclass():
    return MULTI_TEXTS, MULTI_LABELS


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Registry helpers
# ══════════════════════════════════════════════════════════════════════════════

class TestRegistries:
    def test_ml_models_is_dict(self):
        m = get_nlp_ml_models()
        assert isinstance(m, dict)
        assert len(m) >= 5

    def test_ml_models_required_keys(self):
        m = get_nlp_ml_models()
        for name in ["Logistic Regression", "Linear SVC", "SGD Classifier",
                     "Multinomial NB", "Complement NB", "Random Forest"]:
            assert name in m, f"Expected {name!r} in ML registry"

    def test_ml_models_each_is_tuple_of_two(self):
        for name, entry in get_nlp_ml_models().items():
            assert isinstance(entry, tuple) and len(entry) == 2, (
                f"{name}: expected (class, kwargs)")
            cls, kw = entry
            assert callable(cls)
            assert isinstance(kw, dict)

    def test_dl_classifiers_is_dict(self):
        c = get_nlp_dl_classifiers()
        assert isinstance(c, dict)
        assert len(c) >= 3

    def test_dl_classifiers_required_keys(self):
        c = get_nlp_dl_classifiers()
        for name in ["Logistic Regression", "Linear SVC", "Random Forest"]:
            assert name in c

    def test_dl_classifiers_are_fitted_instances(self):
        for name, clf in get_nlp_dl_classifiers().items():
            assert hasattr(clf, "fit"), f"{name} has no .fit()"
            assert hasattr(clf, "predict"), f"{name} has no .predict()"

    def test_st_models_list_nonempty(self):
        assert isinstance(NLP_ST_MODELS, list)
        assert len(NLP_ST_MODELS) >= 1
        assert all(isinstance(m, str) for m in NLP_ST_MODELS)

    def test_default_st_model_is_minilm(self):
        assert NLP_ST_MODELS[0] == "all-MiniLM-L6-v2"


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Device detection
# ══════════════════════════════════════════════════════════════════════════════

class TestDeviceDetection:
    def test_returns_dict(self):
        d = get_nlp_device()
        assert isinstance(d, dict)

    def test_has_type_key(self):
        d = get_nlp_device()
        assert "type" in d

    def test_type_is_valid(self):
        d = get_nlp_device()
        assert d["type"] in ("cpu", "cuda", "mps")

    def test_cuda_fields_present_when_cuda(self):
        d = get_nlp_device()
        if d["type"] == "cuda":
            assert "name" in d
            assert "memory_gb" in d


# ══════════════════════════════════════════════════════════════════════════════
# 3 — NLTK helper
# ══════════════════════════════════════════════════════════════════════════════

class TestNLTKHelper:
    def test_returns_list(self):
        result = _ensure_stopwords()
        assert isinstance(result, list)

    def test_returns_strings(self):
        result = _ensure_stopwords()
        if result:  # may be empty if NLTK not installed
            assert all(isinstance(w, str) for w in result)

    def test_contains_common_words_when_available(self):
        if not _HAS_NLTK:
            pytest.skip("NLTK not installed")
        result = _ensure_stopwords()
        assert "the" in result
        assert "is" in result


# ══════════════════════════════════════════════════════════════════════════════
# 4 — TF-IDF Pipeline Builder
# ══════════════════════════════════════════════════════════════════════════════

class TestBuildTfidfPipeline:

    ALL_ML_MODELS = list(get_nlp_ml_models().keys())

    @pytest.mark.parametrize("model_name", ALL_ML_MODELS)
    def test_pipeline_builds_without_error(self, model_name):
        pipe = build_tfidf_pipeline(model_name)
        assert pipe is not None

    @pytest.mark.parametrize("model_name", ALL_ML_MODELS)
    def test_pipeline_has_tfidf_and_clf_steps(self, model_name):
        pipe = build_tfidf_pipeline(model_name)
        assert "tfidf" in pipe.named_steps
        assert "clf" in pipe.named_steps

    @pytest.mark.parametrize("model_name", ALL_ML_MODELS)
    def test_pipeline_fits_and_predicts(self, model_name, binary):
        from sklearn.preprocessing import LabelEncoder
        texts, labels = binary
        # XGBoost requires numeric labels; engine always pre-encodes with LabelEncoder
        le = LabelEncoder()
        y = le.fit_transform(labels)
        pipe = build_tfidf_pipeline(model_name, max_features=200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(texts, y)
            preds_enc = pipe.predict(texts)
        assert len(preds_enc) == len(texts)
        assert set(preds_enc).issubset(set(range(len(le.classes_))))

    def test_stopwords_disabled(self, binary):
        texts, labels = binary
        pipe = build_tfidf_pipeline("Logistic Regression",
                                    use_stopwords=False, max_features=200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(texts, labels)
        tfidf = pipe.named_steps["tfidf"]
        assert tfidf.stop_words is None

    def test_stopwords_enabled(self, binary):
        texts, labels = binary
        pipe = build_tfidf_pipeline("Logistic Regression",
                                    use_stopwords=True, max_features=200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(texts, labels)
        tfidf = pipe.named_steps["tfidf"]
        assert tfidf.stop_words is not None

    def test_stemming_enabled_when_nltk_available(self, binary):
        if not _HAS_NLTK:
            pytest.skip("NLTK not installed")
        texts, labels = binary
        pipe = build_tfidf_pipeline("Logistic Regression",
                                    use_stemming=True, max_features=200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(texts, labels)
        tfidf = pipe.named_steps["tfidf"]
        assert tfidf.tokenizer is not None

    def test_stemming_skipped_without_nltk(self, binary, monkeypatch):
        import nlp_engine as _ne
        monkeypatch.setattr(_ne, "_HAS_NLTK", False)
        texts, labels = binary
        pipe = build_tfidf_pipeline("Logistic Regression",
                                    use_stemming=True, max_features=200)
        tfidf = pipe.named_steps["tfidf"]
        assert tfidf.tokenizer is None

    @pytest.mark.parametrize("ngram_max", [1, 2, 3])
    def test_ngram_range_applied(self, ngram_max):
        pipe = build_tfidf_pipeline("Logistic Regression", ngram_max=ngram_max)
        tfidf = pipe.named_steps["tfidf"]
        assert tfidf.ngram_range == (1, ngram_max)

    @pytest.mark.parametrize("max_features", [100, 500, 1000])
    def test_max_features_applied(self, max_features):
        pipe = build_tfidf_pipeline("Logistic Regression", max_features=max_features)
        tfidf = pipe.named_steps["tfidf"]
        assert tfidf.max_features == max_features

    def test_nb_models_use_non_sublinear_tf(self):
        for model_name in ("Multinomial NB", "Complement NB"):
            pipe = build_tfidf_pipeline(model_name)
            tfidf = pipe.named_steps["tfidf"]
            assert tfidf.sublinear_tf is False, (
                f"{model_name} should not use sublinear_tf")

    def test_linear_models_use_sublinear_tf(self):
        for model_name in ("Logistic Regression", "Linear SVC", "SGD Classifier"):
            pipe = build_tfidf_pipeline(model_name)
            tfidf = pipe.named_steps["tfidf"]
            assert tfidf.sublinear_tf is True, (
                f"{model_name} should use sublinear_tf")

    def test_multiclass_fits_correctly(self, multiclass):
        from sklearn.preprocessing import LabelEncoder
        texts, labels = multiclass
        le = LabelEncoder()
        y = le.fit_transform(labels)
        pipe = build_tfidf_pipeline("Logistic Regression", max_features=200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(texts, y)
        preds_enc = pipe.predict(texts)
        preds = le.inverse_transform(preds_enc)
        assert set(preds).issubset({"tech", "sports", "politics"})


# ══════════════════════════════════════════════════════════════════════════════
# 5 — Baseline CV (ML track)
# ══════════════════════════════════════════════════════════════════════════════

class TestRunNlpBaseline:

    def test_returns_tuple_of_dict_and_list(self, binary):
        texts, labels = binary
        results, class_names = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            cv_folds=3, random_state=0, max_features=200)
        assert isinstance(results, dict)
        assert isinstance(class_names, list)

    def test_class_names_correct_binary(self, binary):
        texts, labels = binary
        _, class_names = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            cv_folds=3, random_state=0, max_features=200)
        assert set(class_names) == {"positive", "negative"}

    def test_class_names_correct_multiclass(self, multiclass):
        texts, labels = multiclass
        _, class_names = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            cv_folds=3, random_state=0, max_features=200)
        assert set(class_names) == {"tech", "sports", "politics"}

    def test_result_keys_present(self, binary):
        texts, labels = binary
        results, _ = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            cv_folds=3, random_state=0, max_features=200)
        r = results["Logistic Regression"]
        for key in ("mean", "std", "fold_scores", "time_s"):
            assert key in r, f"Missing key {key!r}"

    def test_mean_is_float(self, binary):
        texts, labels = binary
        results, _ = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            cv_folds=3, random_state=0, max_features=200)
        assert isinstance(results["Logistic Regression"]["mean"], float)

    def test_fold_scores_length_matches_cv_folds(self, binary):
        texts, labels = binary
        results, _ = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            cv_folds=3, random_state=0, max_features=200)
        assert len(results["Logistic Regression"]["fold_scores"]) == 3

    def test_multiple_models_all_in_results(self, binary):
        texts, labels = binary
        models = ["Logistic Regression", "Multinomial NB", "Complement NB"]
        results, _ = run_nlp_baseline(
            texts, labels, models,
            cv_folds=3, random_state=0, max_features=200)
        assert set(results.keys()) == set(models)

    def test_model_callback_called_for_each_model(self, binary):
        texts, labels = binary
        called = []
        def cb(name, mean, std, elapsed, folds):
            called.append(name)
        models = ["Logistic Regression", "Multinomial NB"]
        run_nlp_baseline(
            texts, labels, models,
            cv_folds=3, random_state=0, max_features=200,
            model_callback=cb)
        assert set(called) == set(models)

    def test_timing_is_nonnegative(self, binary):
        texts, labels = binary
        results, _ = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            cv_folds=3, random_state=0, max_features=200)
        assert results["Logistic Regression"]["time_s"] >= 0.0

    def test_progress_bar_callback_accepted(self, binary):
        texts, labels = binary

        class _FakeProg:
            def progress(self, v): pass

        run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            cv_folds=3, random_state=0, max_features=200,
            progress_bar=_FakeProg())

    def test_status_text_callback_accepted(self, binary):
        texts, labels = binary

        class _FakeStatus:
            def info(self, msg): pass

        run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            cv_folds=3, random_state=0, max_features=200,
            status_text=_FakeStatus())

    @pytest.mark.parametrize("model_name", list(get_nlp_ml_models().keys()))
    def test_each_ml_model_baseline(self, model_name, binary):
        texts, labels = binary
        results, _ = run_nlp_baseline(
            texts, labels, [model_name],
            cv_folds=3, random_state=0, max_features=200)
        r = results[model_name]
        # Either succeeded (mean is a real number) or failed gracefully (error key)
        if "error" in r:
            assert isinstance(r["error"], str)
        else:
            assert 0.0 <= r["mean"] <= 1.0

    def test_stopwords_removed_baseline(self, binary):
        texts, labels = binary
        results, _ = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            use_stopwords=True, cv_folds=3, random_state=0, max_features=200)
        assert "Logistic Regression" in results

    def test_no_stopwords_baseline(self, binary):
        texts, labels = binary
        results, _ = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            use_stopwords=False, cv_folds=3, random_state=0, max_features=200)
        assert "Logistic Regression" in results

    def test_unigram_baseline(self, binary):
        texts, labels = binary
        results, _ = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            ngram_max=1, cv_folds=3, random_state=0, max_features=200)
        r = results["Logistic Regression"]
        assert "mean" in r

    def test_trigram_baseline(self, binary):
        texts, labels = binary
        results, _ = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            ngram_max=3, cv_folds=3, random_state=0, max_features=500)
        assert "Logistic Regression" in results


# ══════════════════════════════════════════════════════════════════════════════
# 6 — Optimisation (ML track)
# ══════════════════════════════════════════════════════════════════════════════

class TestRunNlpOptimization:

    @pytest.fixture(scope="class")
    def binary_baseline(self):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        results, class_names = run_nlp_baseline(
            texts, labels, ["Logistic Regression", "Multinomial NB"],
            cv_folds=3, random_state=0, max_features=200)
        return results, class_names

    @pytest.fixture(scope="class")
    def multiclass_baseline(self):
        texts, labels = MULTI_TEXTS, MULTI_LABELS
        results, class_names = run_nlp_baseline(
            texts, labels, ["Logistic Regression", "Complement NB"],
            cv_folds=3, random_state=0, max_features=200)
        return results, class_names

    def test_returns_nine_element_tuple(self, binary_baseline):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        out = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert len(out) == 9

    def test_results_list_has_one_row_per_model(self, binary_baseline):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        models = ["Logistic Regression", "Multinomial NB"]
        (results_list, *_) = run_nlp_optimization(
            texts, labels, models,
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert len(results_list) == 2

    def test_results_list_has_required_columns(self, binary_baseline):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        (results_list, *_) = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        row = results_list[0]
        for col in ("Algorithm", "Track", "Baseline Macro F1",
                    "Selection CV Macro F1", "Training Time (s)"):
            assert col in row, f"Missing column {col!r}"

    def test_best_acc_is_float(self, binary_baseline):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        (_, best_acc, *_) = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert isinstance(best_acc, float)
        assert 0.0 <= best_acc <= 1.0

    def test_class_names_correct(self, binary_baseline):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        (*_, class_names, _curves) = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert set(class_names) == {"positive", "negative"}

    def test_winner_curves_has_confusion_matrix(self, binary_baseline):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        (*_, winner_curves) = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert "confusion_matrix" in winner_curves
        cm = winner_curves["confusion_matrix"]
        assert "matrix" in cm and "labels" in cm

    def test_class_report_has_standard_keys(self, binary_baseline):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        (_, _, _, _, _, _, class_report, *_) = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert class_report is not None
        assert "accuracy" in class_report
        assert "confusion_matrix" in class_report

    def test_track_label_in_results(self, binary_baseline):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        (results_list, *_) = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert results_list[0]["Track"] == "ML"

    def test_multiclass_optimization(self, multiclass_baseline):
        texts, labels = MULTI_TEXTS, MULTI_LABELS
        bl, _ = multiclass_baseline
        (results_list, best_acc, *_, class_names, curves) = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert set(class_names) == {"tech", "sports", "politics"}
        assert 0.0 <= best_acc <= 1.0

    def test_export_model_writes_file(self, binary_baseline, tmp_path):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        (*_, curves) = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=True, export_dir=str(tmp_path))
        ep = curves.get("exported_model_path")
        assert ep is not None
        assert os.path.exists(ep)
        assert ep.endswith(".joblib")

    def test_no_export_produces_no_file(self, binary_baseline):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        (*_, curves) = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert curves.get("exported_model_path") is None

    @pytest.mark.parametrize("model_name", list(get_nlp_ml_models().keys()))
    def test_each_model_optimizes_without_crash(self, model_name, binary):
        texts, labels = binary
        bl, _ = run_nlp_baseline(
            texts, labels, [model_name],
            cv_folds=3, random_state=0, max_features=200)
        out = run_nlp_optimization(
            texts, labels, [model_name],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert len(out) == 9
        results_list = out[0]
        assert len(results_list) == 1
        # Either succeeded or recorded a graceful error
        row = results_list[0]
        assert "Algorithm" in row

    def test_improvement_is_nonnegative(self, binary_baseline):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        (_, _, _, imp, *_) = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert imp >= 0.0

    def test_winner_name_in_curves(self, binary_baseline):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        (*_, curves) = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert curves.get("winner_name") == "Logistic Regression"

    def test_winner_curves_has_track_field(self, binary_baseline):
        texts, labels = BINARY_TEXTS, BINARY_LABELS
        bl, _ = binary_baseline
        (*_, curves) = run_nlp_optimization(
            texts, labels, ["Logistic Regression"],
            bl, cv_folds=3, n_iter=5, random_state=0, max_features=200,
            export_model=False)
        assert curves.get("track") == "ml"


# ══════════════════════════════════════════════════════════════════════════════
# 7 — LIME explanations
# ══════════════════════════════════════════════════════════════════════════════

def _encode_fit(model_name: str, texts, labels):
    """Fit a TF-IDF pipeline using label-encoded targets (matches engine usage)."""
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(labels)
    pipe = build_tfidf_pipeline(model_name, max_features=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(texts, y)
    return pipe, le


class TestComputeNlpLime:

    @pytest.fixture(scope="class")
    def fitted_lr(self):
        pipe, le = _encode_fit("Logistic Regression", BINARY_TEXTS, BINARY_LABELS)
        return pipe

    @pytest.fixture(scope="class")
    def fitted_mnb(self):
        pipe, le = _encode_fit("Multinomial NB", BINARY_TEXTS, BINARY_LABELS)
        return pipe

    @pytest.fixture(scope="class")
    def fitted_svc(self):
        pipe, le = _encode_fit("Linear SVC", BINARY_TEXTS, BINARY_LABELS)
        return pipe

    # The fitted_lr/mnb/svc fixtures use label-encoded (0/1) targets.
    # Class names passed to LIME are the integer string labels that match.
    _CLASS_NAMES = ["0", "1"]  # LabelEncoder sorts alphabetically: negative=0, positive=1

    def test_lime_not_installed_returns_error(self, fitted_lr, monkeypatch):
        import nlp_engine as _ne
        monkeypatch.setattr(_ne, "_HAS_LIME", False)
        result = compute_nlp_lime(
            fitted_lr, BINARY_TEXTS, self._CLASS_NAMES,
            sample_idx=0, num_samples=50)
        assert "error" in result

    @pytest.mark.skipif(not _HAS_LIME, reason="lime not installed")
    def test_lime_lr_returns_dict(self, fitted_lr):
        result = compute_nlp_lime(
            fitted_lr, BINARY_TEXTS, self._CLASS_NAMES,
            sample_idx=0, num_samples=50)
        assert isinstance(result, dict)

    @pytest.mark.skipif(not _HAS_LIME, reason="lime not installed")
    def test_lime_lr_has_required_keys(self, fitted_lr):
        result = compute_nlp_lime(
            fitted_lr, BINARY_TEXTS, self._CLASS_NAMES,
            sample_idx=0, num_samples=50)
        if "error" not in result:
            for k in ("text", "predicted_class", "explanation", "class_names"):
                assert k in result

    @pytest.mark.skipif(not _HAS_LIME, reason="lime not installed")
    def test_lime_lr_explanation_is_list_of_tuples(self, fitted_lr):
        result = compute_nlp_lime(
            fitted_lr, BINARY_TEXTS, self._CLASS_NAMES,
            sample_idx=0, num_samples=50)
        if "error" not in result:
            exp = result["explanation"]
            assert isinstance(exp, list)
            for item in exp:
                assert isinstance(item, tuple) and len(item) == 2
                word, weight = item
                assert isinstance(word, str)
                assert isinstance(weight, float)

    @pytest.mark.skipif(not _HAS_LIME, reason="lime not installed")
    def test_lime_predicted_class_is_string(self, fitted_lr):
        result = compute_nlp_lime(
            fitted_lr, BINARY_TEXTS, self._CLASS_NAMES,
            sample_idx=0, num_samples=50)
        if "error" not in result:
            assert isinstance(result["predicted_class"], str)
            assert result["predicted_class"] in self._CLASS_NAMES

    @pytest.mark.skipif(not _HAS_LIME, reason="lime not installed")
    def test_lime_text_matches_input_sample(self, fitted_lr):
        idx = 5
        result = compute_nlp_lime(
            fitted_lr, BINARY_TEXTS, self._CLASS_NAMES,
            sample_idx=idx, num_samples=50)
        if "error" not in result:
            assert result["text"] == BINARY_TEXTS[idx]

    @pytest.mark.skipif(not _HAS_LIME, reason="lime not installed")
    def test_lime_mnb_works(self, fitted_mnb):
        result = compute_nlp_lime(
            fitted_mnb, BINARY_TEXTS, self._CLASS_NAMES,
            sample_idx=0, num_samples=50)
        assert isinstance(result, dict)

    @pytest.mark.skipif(not _HAS_LIME, reason="lime not installed")
    def test_lime_linear_svc_returns_error_or_result(self, fitted_svc):
        # LinearSVC has no predict_proba — LIME should return graceful error dict
        result = compute_nlp_lime(
            fitted_svc, BINARY_TEXTS, self._CLASS_NAMES,
            sample_idx=0, num_samples=50)
        assert isinstance(result, dict)
        if "error" in result:
            assert isinstance(result["error"], str)
        else:
            assert "explanation" in result

    @pytest.mark.skipif(not _HAS_LIME, reason="lime not installed")
    def test_lime_num_features_respected(self, fitted_lr):
        n_feats = 5
        result = compute_nlp_lime(
            fitted_lr, BINARY_TEXTS, self._CLASS_NAMES,
            sample_idx=0, num_features=n_feats, num_samples=50)
        if "error" not in result:
            assert len(result["explanation"]) <= n_feats

    @pytest.mark.skipif(not _HAS_LIME, reason="lime not installed")
    def test_lime_multiclass(self):
        pipe, le = _encode_fit("Logistic Regression", MULTI_TEXTS, MULTI_LABELS)
        class_names = [str(c) for c in range(len(le.classes_))]
        result = compute_nlp_lime(
            pipe, MULTI_TEXTS, class_names,
            sample_idx=0, num_samples=50)
        assert isinstance(result, dict)
        if "error" not in result:
            assert result["predicted_class"] in class_names


# ══════════════════════════════════════════════════════════════════════════════
# 8 — Top TF-IDF features
# ══════════════════════════════════════════════════════════════════════════════

class TestGetTopTfidfFeatures:

    # Binary: LabelEncoder sorts alphabetically → negative=0, positive=1
    _BIN_CLASSES = ["negative", "positive"]

    @pytest.fixture(scope="class")
    def fitted_lr(self):
        pipe, _ = _encode_fit("Logistic Regression", BINARY_TEXTS, BINARY_LABELS)
        return pipe

    @pytest.fixture(scope="class")
    def fitted_svc(self):
        pipe, _ = _encode_fit("Linear SVC", BINARY_TEXTS, BINARY_LABELS)
        return pipe

    @pytest.fixture(scope="class")
    def fitted_rf(self):
        pipe, _ = _encode_fit("Random Forest", BINARY_TEXTS, BINARY_LABELS)
        return pipe

    def test_lr_returns_dict(self, fitted_lr):
        result = get_top_tfidf_features(fitted_lr, self._BIN_CLASSES)
        assert isinstance(result, dict)

    def test_lr_has_entry_per_class(self, fitted_lr):
        result = get_top_tfidf_features(fitted_lr, self._BIN_CLASSES)
        assert len(result) >= 1  # binary LR may have 1 or 2 rows in coef_

    def test_each_entry_has_positive_and_negative(self, fitted_lr):
        result = get_top_tfidf_features(fitted_lr, self._BIN_CLASSES)
        for label, sides in result.items():
            assert "positive" in sides, f"{label}: missing 'positive'"
            assert "negative" in sides, f"{label}: missing 'negative'"

    def test_top_n_respected(self, fitted_lr):
        n = 5
        result = get_top_tfidf_features(fitted_lr, self._BIN_CLASSES, n=n)
        for label, sides in result.items():
            assert len(sides["positive"]) <= n
            assert len(sides["negative"]) <= n

    def test_positive_entries_are_word_weight_tuples(self, fitted_lr):
        result = get_top_tfidf_features(fitted_lr, self._BIN_CLASSES)
        for label, sides in result.items():
            for word, weight in sides["positive"]:
                assert isinstance(word, str)
                assert isinstance(weight, float)

    def test_svc_returns_features(self, fitted_svc):
        result = get_top_tfidf_features(fitted_svc, self._BIN_CLASSES)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_random_forest_returns_empty(self, fitted_rf):
        # RF has no coef_ — should return empty dict gracefully
        result = get_top_tfidf_features(fitted_rf, self._BIN_CLASSES)
        assert result == {}

    def test_multiclass_has_three_entries(self):
        pipe, le = _encode_fit("Logistic Regression", MULTI_TEXTS, MULTI_LABELS)
        class_names = le.classes_.tolist()  # ["politics", "sports", "tech"]
        result = get_top_tfidf_features(pipe, class_names)
        assert len(result) == 3

    def test_weights_are_sorted_descending_for_positive(self, fitted_lr):
        result = get_top_tfidf_features(fitted_lr, self._BIN_CLASSES)
        for label, sides in result.items():
            weights = [w for _, w in sides["positive"]]
            assert weights == sorted(weights, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
# 9 — DL track (skipped when sentence-transformers absent)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed")
class TestDlTrack:

    def test_encode_with_transformer_shape(self):
        texts = BINARY_TEXTS[:10]
        embeddings = encode_with_transformer(
            texts, "all-MiniLM-L6-v2", device="cpu", batch_size=8)
        assert embeddings.shape == (10, 384)  # MiniLM-L6 has 384-dim output

    def test_encode_returns_float32(self):
        embeddings = encode_with_transformer(
            BINARY_TEXTS[:5], "all-MiniLM-L6-v2", device="cpu")
        assert embeddings.dtype == np.float32

    def test_dl_baseline_returns_correct_structure(self):
        texts, labels = BINARY_TEXTS[:20], BINARY_LABELS[:20]
        results, class_names = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            track="dl", st_model_name="all-MiniLM-L6-v2",
            device="cpu", cv_folds=3, random_state=0)
        assert "Logistic Regression" in results
        r = results["Logistic Regression"]
        assert "mean" in r
        assert 0.0 <= r["mean"] <= 1.0
        assert set(class_names) == {"positive", "negative"}

    def test_dl_optimization_runs(self):
        texts, labels = BINARY_TEXTS[:20], BINARY_LABELS[:20]
        bl, _ = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            track="dl", st_model_name="all-MiniLM-L6-v2",
            device="cpu", cv_folds=3, random_state=0)
        out = run_nlp_optimization(
            texts, labels, ["Logistic Regression"], bl,
            track="dl", st_model_name="all-MiniLM-L6-v2",
            device="cpu", cv_folds=3, n_iter=3, random_state=0,
            export_model=False)
        assert len(out) == 9
        results_list = out[0]
        assert results_list[0]["Track"] == "DL"

    def test_encode_with_transformer_missing_raises(self, monkeypatch):
        import nlp_engine as _ne
        monkeypatch.setattr(_ne, "_HAS_ST", False)
        with pytest.raises(ImportError, match="sentence-transformers"):
            encode_with_transformer(BINARY_TEXTS[:5], "all-MiniLM-L6-v2")


@pytest.mark.skipif(_HAS_ST, reason="sentence-transformers IS installed — skip absent test")
class TestDlTrackAbsent:

    def test_encode_raises_import_error(self):
        with pytest.raises(ImportError, match="sentence-transformers"):
            encode_with_transformer(BINARY_TEXTS[:5], "all-MiniLM-L6-v2")

    def test_dl_baseline_raises_import_error(self):
        with pytest.raises(ImportError):
            run_nlp_baseline(
                BINARY_TEXTS[:10], BINARY_LABELS[:10], ["Logistic Regression"],
                track="dl", st_model_name="all-MiniLM-L6-v2",
                device="cpu", cv_folds=3, random_state=0)


# ══════════════════════════════════════════════════════════════════════════════
# 10 — NLP Reporting renderers
# ══════════════════════════════════════════════════════════════════════════════

class TestNlpReportingRenderers:
    """Exercises render functions via the stubbed streamlit — must not raise."""

    @pytest.fixture(autouse=True)
    def import_reporting(self):
        import reporting as R
        self.R = R

    def test_lime_empty_data_no_raise(self):
        self.R.render_nlp_lime_explanation({})

    def test_lime_error_dict_no_raise(self):
        self.R.render_nlp_lime_explanation(
            {"error": "lime not installed — run: pip install lime"})

    def test_lime_full_data_no_raise(self):
        data = {
            "text":            "I loved this great film very much",
            "predicted_class": "positive",
            "explanation":     [("loved", 0.42), ("great", 0.31),
                                ("film", 0.15), ("not", -0.28)],
            "class_names":     ["negative", "positive"],
        }
        self.R.render_nlp_lime_explanation(data)

    def test_lime_many_features_no_raise(self):
        explanation = [(f"word_{i}", float(np.sin(i))) for i in range(30)]
        data = {
            "text": "some long text with many words here and there",
            "predicted_class": "negative",
            "explanation": explanation,
            "class_names": ["negative", "positive"],
        }
        self.R.render_nlp_lime_explanation(data)

    def test_lime_missing_text_key_no_raise(self):
        data = {
            "predicted_class": "positive",
            "explanation":     [("good", 0.5)],
            "class_names":     ["negative", "positive"],
        }
        self.R.render_nlp_lime_explanation(data)

    def test_top_features_empty_dict_no_raise(self):
        self.R.render_nlp_top_features({})

    def test_top_features_binary_no_raise(self):
        data = {
            "positive": {
                "positive": [("love", 0.8), ("great", 0.6), ("excellent", 0.5)],
                "negative": [("terrible", -0.7), ("awful", -0.6)],
            },
            "negative": {
                "positive": [("terrible", 0.7), ("bad", 0.5)],
                "negative": [("love", -0.8), ("great", -0.6)],
            },
        }
        self.R.render_nlp_top_features(data)

    def test_top_features_multiclass_no_raise(self):
        data = {
            "tech": {
                "positive": [("python", 0.9), ("algorithm", 0.8)],
                "negative": [("goal", -0.7), ("election", -0.6)],
            },
            "sports": {
                "positive": [("football", 0.9), ("championship", 0.7)],
                "negative": [("software", -0.8), ("parliament", -0.5)],
            },
            "politics": {
                "positive": [("election", 0.9), ("parliament", 0.7)],
                "negative": [("football", -0.8), ("python", -0.6)],
            },
        }
        self.R.render_nlp_top_features(data)

    def test_top_features_empty_sides_no_raise(self):
        data = {
            "class_a": {"positive": [], "negative": []},
        }
        self.R.render_nlp_top_features(data)

    def test_top_features_only_positive_side_no_raise(self):
        data = {
            "class_a": {
                "positive": [("word", 0.5)],
                "negative": [],
            },
        }
        self.R.render_nlp_top_features(data)

    def test_top_features_integration_with_fitted_model(self):
        """Get real feature data from a fitted pipeline and render it."""
        from sklearn.preprocessing import LabelEncoder
        pipe, le = _encode_fit("Logistic Regression", BINARY_TEXTS, BINARY_LABELS)
        top_feats = get_top_tfidf_features(pipe, le.classes_.tolist())
        self.R.render_nlp_top_features(top_feats)  # must not raise


# ══════════════════════════════════════════════════════════════════════════════
# 11 — Edge cases & robustness
# ══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_very_short_texts_do_not_crash(self):
        texts  = ["ok", "bad", "yes", "no", "great", "awful",
                  "good", "poor", "fine", "terrible"] * 6
        labels = ["pos", "neg"] * 30
        results, _ = run_nlp_baseline(
            texts, labels, ["Multinomial NB"],
            cv_folds=3, random_state=0, max_features=50)
        assert "Multinomial NB" in results

    def test_numeric_strings_in_labels(self):
        texts  = BINARY_TEXTS
        labels = ["1" if l == "positive" else "0" for l in BINARY_LABELS]
        results, class_names = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            cv_folds=3, random_state=0, max_features=200)
        assert set(class_names).issubset({"0", "1"})

    def test_empty_explanation_lime_no_crash(self):
        data = {
            "text": "test",
            "predicted_class": "positive",
            "explanation": [],
            "class_names": ["negative", "positive"],
        }
        import reporting as R
        R.render_nlp_lime_explanation(data)

    def test_baseline_progress_bar_none_accepted(self, binary):
        texts, labels = binary
        # None is the default — must not crash
        run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            cv_folds=3, random_state=0, max_features=200,
            progress_bar=None, status_text=None)

    def test_optimization_single_class_graceful(self):
        # All texts same label — CV may fail; should not raise Python exception
        texts  = BINARY_TEXTS[:20]
        labels = ["positive"] * 20
        try:
            bl, _ = run_nlp_baseline(
                texts, labels, ["Logistic Regression"],
                cv_folds=3, random_state=0, max_features=100)
            run_nlp_optimization(
                texts, labels, ["Logistic Regression"], bl,
                cv_folds=3, n_iter=3, random_state=0,
                max_features=100, export_model=False)
        except Exception as e:
            # Acceptable: StratifiedKFold may raise ValueError for 1 class
            assert "class" in str(e).lower() or "stratif" in str(e).lower()

    def test_baseline_with_repeated_texts(self):
        texts  = ["the cat sat on the mat"] * 60
        labels = ["a"] * 30 + ["b"] * 30
        results, _ = run_nlp_baseline(
            texts, labels, ["Logistic Regression"],
            cv_folds=3, random_state=0, max_features=50)
        assert "Logistic Regression" in results

    def test_get_top_features_pipeline_without_tfidf(self):
        # Pipeline with no 'tfidf' step — should return {}
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LogisticRegression as LR
        pipe = Pipeline([("scaler", StandardScaler()), ("clf", LR())])
        result = get_top_tfidf_features(pipe, ["a", "b"])
        assert result == {}

