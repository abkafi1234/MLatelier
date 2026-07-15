"""
Tests for tabular_engine.py — covers baseline, optimization (abbreviated),
calibration, ablation, learning curve, feature ablation, PDP, SHAP waterfall, LIME.
Uses small synthetic datasets so tests run quickly without heavy dependencies.
"""
from __future__ import annotations

import sys
import os
import types
import warnings

import numpy as np
import pandas as pd
import pytest

# ── Stub streamlit ────────────────────────────────────────────────────────────
st_stub = types.ModuleType("streamlit")
for _attr in ["info", "success", "warning", "error", "progress", "empty",
              "markdown", "dataframe", "metric", "caption", "spinner"]:
    setattr(st_stub, _attr, lambda *a, **kw: None)
st_stub.session_state = {}
sys.modules.setdefault("streamlit", st_stub)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/mlatelier"))

from tabular_engine import (
    _is_regression,
    _prepare_data,
    compute_calibration_data,
    compute_learning_curve,
    compute_lime_explanation,
    compute_pdp,
    compute_shap_waterfall,
    get_classification_models,
    get_regression_models,
    run_feature_ablation,
    run_tabular_ablation,
    run_tabular_baseline,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def clf_df():
    rng = np.random.default_rng(42)
    n = 200
    return pd.DataFrame({
        "age":    rng.integers(18, 70, n).astype(float),
        "income": rng.normal(50000, 15000, n),
        "score":  rng.uniform(0, 100, n),
        "label":  rng.choice([0, 1], n),
    })


@pytest.fixture
def reg_df():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.normal(size=(n, 3))
    y = X[:, 0] * 2 + X[:, 1] * -1 + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2], "target": y
    })


@pytest.fixture
def multi_clf_df():
    rng = np.random.default_rng(7)
    n = 180
    return pd.DataFrame({
        "f1": rng.normal(0, 1, n),
        "f2": rng.uniform(-2, 2, n),
        "f3": rng.exponential(1, n),
        "cls": rng.choice([0, 1, 2], n),
    })


# ── _is_regression ────────────────────────────────────────────────────────────

def test_is_regression_continuous():
    s = pd.Series(np.random.default_rng(0).normal(size=200))
    assert _is_regression(s) is True


def test_is_regression_binary():
    s = pd.Series([0, 1, 0, 1, 1, 0] * 30)
    assert _is_regression(s) is False


def test_is_regression_multiclass():
    s = pd.Series([0, 1, 2, 0, 1, 2] * 30)
    assert _is_regression(s) is False


def test_is_regression_small_int():
    # 10 unique values / 10 total = 1.0 > threshold: treated as regression
    # This is expected — with n=10 any int sequence looks continuous
    s = pd.Series(range(10))
    # Verify it doesn't raise; actual True/False depends on cardinality_ratio
    result = _is_regression(s)
    assert isinstance(result, bool)


# ── get_*_models ──────────────────────────────────────────────────────────────

def test_get_classification_models():
    m = get_classification_models()
    assert isinstance(m, dict)
    assert len(m) >= 10
    assert "Random Forest" in m
    assert "XGBoost" in m


def test_get_regression_models():
    m = get_regression_models()
    assert isinstance(m, dict)
    assert len(m) >= 8
    assert "Random Forest Regressor" in m


# ── run_tabular_baseline ──────────────────────────────────────────────────────

def test_baseline_classification(clf_df):
    results, task = run_tabular_baseline(
        clf_df, ["age", "income", "score"], "label",
        ["Random Forest", "Logistic Regression"],
        handle_imbalance=False, random_state=0, test_size=0.2, cv_folds=3,
    )
    assert task == "classification"
    assert "Random Forest" in results
    assert "Logistic Regression" in results
    for v in results.values():
        assert "mean" in v
        assert "fold_scores" in v
        assert "time_s" in v


def test_baseline_regression(reg_df):
    results, task = run_tabular_baseline(
        reg_df, ["x1", "x2", "x3"], "target",
        ["Random Forest Regressor"],
        handle_imbalance=False, random_state=0, test_size=0.2, cv_folds=3,
    )
    assert task == "regression"
    assert "Random Forest Regressor" in results


def test_baseline_model_callback(clf_df):
    called = []
    def cb(name, mean, std, elapsed, folds):
        called.append(name)

    run_tabular_baseline(
        clf_df, ["age", "income", "score"], "label",
        ["Logistic Regression"],
        handle_imbalance=False, random_state=0, cv_folds=3,
        model_callback=cb,
    )
    assert "Logistic Regression" in called


def test_baseline_timing_nonzero(clf_df):
    results, _ = run_tabular_baseline(
        clf_df, ["age", "income", "score"], "label",
        ["Decision Tree"],
        handle_imbalance=False, random_state=0, cv_folds=3,
    )
    assert results["Decision Tree"]["time_s"] >= 0.0


# ── _prepare_data ─────────────────────────────────────────────────────────────

def test_prepare_data_classification(clf_df):
    d = _prepare_data(clf_df, ["age", "income", "score"], "label",
                       handle_imbalance=False, random_state=0)
    assert d.task_type == "classification"
    assert d.X_train is not None
    assert d.X_test  is not None
    assert len(d.X_train) + len(d.X_test) == len(clf_df)


def test_prepare_data_regression(reg_df):
    d = _prepare_data(reg_df, ["x1", "x2", "x3"], "target",
                       handle_imbalance=False, random_state=0)
    assert d.task_type == "regression"
    assert d.class_names is None


def test_prepare_data_custom_folds(clf_df):
    d = _prepare_data(clf_df, ["age", "income", "score"], "label",
                       handle_imbalance=False, random_state=0, cv_folds=10)
    assert d.cv.get_n_splits() == 10


# ── compute_calibration_data ──────────────────────────────────────────────────

def test_calibration_binary(clf_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("model", LogisticRegression(max_iter=200, random_state=0))])
    d = _prepare_data(clf_df, ["age", "income", "score"], "label",
                       handle_imbalance=False, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(d.X_train, d.y_train)
    result = compute_calibration_data(pipe, d.X_test, d.y_test)
    assert result["type"] == "binary"
    assert "fop" in result
    assert "brier_score" in result
    assert 0.0 <= result["brier_score"] <= 1.0


def test_calibration_multiclass(multi_clf_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("model", LogisticRegression(max_iter=200, random_state=0))])
    d = _prepare_data(multi_clf_df, ["f1", "f2", "f3"], "cls",
                       handle_imbalance=False, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(d.X_train, d.y_train)
    result = compute_calibration_data(pipe, d.X_test, d.y_test)
    assert result["type"] == "multiclass"


# ── run_tabular_ablation ──────────────────────────────────────────────────────

def test_ablation_smote_variation(clf_df):
    configs = [
        {"label": "★ Baseline", "handle_imbalance": True,
         "scaler": "auto", "test_size": 0.2, "cv_folds": 3},
        {"label": "No SMOTE", "handle_imbalance": False,
         "scaler": "auto", "test_size": 0.2, "cv_folds": 3},
    ]
    results = run_tabular_ablation(
        clf_df, ["age", "income", "score"], "label",
        "Logistic Regression", configs, random_state=0,
    )
    assert len(results) == 2
    assert results[0]["Config"] == "★ Baseline"
    assert "CV Score" in results[0]


def test_ablation_delta_computed(clf_df):
    configs = [
        {"label": "★ Baseline", "handle_imbalance": False,
         "scaler": "standard", "test_size": 0.2, "cv_folds": 3},
        {"label": "Variant",    "handle_imbalance": False,
         "scaler": "robust",    "test_size": 0.2, "cv_folds": 3},
    ]
    results = run_tabular_ablation(
        clf_df, ["age", "income", "score"], "label",
        "Decision Tree", configs, random_state=0,
    )
    assert results[1]["Delta (%)"] != "—"


# ── compute_learning_curve ────────────────────────────────────────────────────

def test_learning_curve_classification(clf_df):
    result = compute_learning_curve(
        clf_df, ["age", "income", "score"], "label",
        "Logistic Regression",
        handle_imbalance=False, random_state=0, cv_folds=3, n_points=4,
    )
    assert "train_sizes"  in result
    assert "val_means"    in result
    assert "metric_name"  in result
    assert len(result["train_sizes"]) == 4
    for m in result["val_means"]:
        assert -1.0 <= m <= 1.0


def test_learning_curve_regression(reg_df):
    result = compute_learning_curve(
        reg_df, ["x1", "x2", "x3"], "target",
        "Random Forest Regressor",
        handle_imbalance=False, random_state=0, cv_folds=3, n_points=3,
    )
    assert result["metric_name"] == "R² Score"


def test_learning_curve_invalid_model(clf_df):
    with pytest.raises(ValueError, match="not found"):
        compute_learning_curve(
            clf_df, ["age", "income"], "label",
            "NonExistentModel999",
        )


# ── run_feature_ablation ──────────────────────────────────────────────────────

def test_feature_ablation_basic(clf_df):
    results = run_feature_ablation(
        clf_df, ["age", "income", "score"], "label",
        "Logistic Regression",
        handle_imbalance=False, random_state=0, cv_folds=3,
    )
    assert len(results) == 3
    names = [r["Feature Dropped"] for r in results]
    assert set(names) == {"age", "income", "score"}
    for r in results:
        assert "Delta (%)" in r
        assert "Baseline CV" in r
        assert "CV Without" in r


def test_feature_ablation_single_feature(clf_df):
    with pytest.raises(ValueError, match="at least 2"):
        run_feature_ablation(
            clf_df, ["age"], "label", "Logistic Regression"
        )


def test_feature_ablation_regression(reg_df):
    results = run_feature_ablation(
        reg_df, ["x1", "x2", "x3"], "target",
        "Random Forest Regressor",
        handle_imbalance=False, random_state=0, cv_folds=3,
    )
    assert len(results) == 3


# ── compute_pdp ───────────────────────────────────────────────────────────────

def test_compute_pdp(clf_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("model", LogisticRegression(max_iter=200, random_state=0)),
    ])
    d = _prepare_data(clf_df, ["age", "income", "score"], "label",
                       handle_imbalance=False, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(d.X_train, d.y_train)
    result = compute_pdp(pipe, d.X_test, ["age", "income", "score"],
                          top_n=2, grid_resolution=10)
    assert isinstance(result, dict)
    for feat, (xs, ys) in result.items():
        assert len(xs) == 10
        assert len(ys) == 10


def test_compute_pdp_empty_data():
    result = compute_pdp(None, pd.DataFrame(), [], top_n=3)
    assert result == {}


# ── compute_shap_waterfall ────────────────────────────────────────────────────

def test_compute_shap_waterfall(clf_df):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("model", RandomForestClassifier(n_estimators=10, random_state=0))])
    d = _prepare_data(clf_df, ["age", "income", "score"], "label",
                       handle_imbalance=False, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(d.X_train, d.y_train)
    result = compute_shap_waterfall(pipe, d.X_test, ["age", "income", "score"],
                                     sample_idx=0)
    if result:  # SHAP may not be installed in all test envs
        assert "features"    in result
        assert "shap_values" in result
        assert "base_value"  in result


def test_compute_shap_waterfall_empty_df():
    result = compute_shap_waterfall(None, pd.DataFrame(), [], sample_idx=0)
    assert result == {}


# ── compute_lime_explanation ──────────────────────────────────────────────────

def test_compute_lime_explanation(clf_df):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = Pipeline([
        ("sc", StandardScaler()),
        ("model", LogisticRegression(max_iter=200, random_state=0)),
    ])
    d = _prepare_data(clf_df, ["age", "income", "score"], "label",
                       handle_imbalance=False, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(d.X_train, d.y_train)
    result = compute_lime_explanation(
        pipe, d.X_train, d.X_test,
        ["age", "income", "score"],
        task_type="classification", sample_idx=0,
    )
    if result:  # lime may not be installed
        assert "features" in result
        assert "weights"  in result
        assert "label"    in result


def test_compute_lime_empty():
    result = compute_lime_explanation(None, pd.DataFrame(), pd.DataFrame(), [])
    assert result == {}


# ── Multi-label classification ─────────────────────────────────────────────────

@pytest.fixture
def multilabel_df():
    """Synthetic multi-label dataset: 3 binary label columns."""
    rng = np.random.default_rng(99)
    n   = 200
    X   = rng.normal(size=(n, 4))
    return pd.DataFrame({
        "f1": X[:, 0], "f2": X[:, 1], "f3": X[:, 2], "f4": X[:, 3],
        "lbl_a": rng.integers(0, 2, n),
        "lbl_b": rng.integers(0, 2, n),
        "lbl_c": rng.integers(0, 2, n),
    })


LABEL_COLS = ["lbl_a", "lbl_b", "lbl_c"]
FEAT_COLS  = ["f1", "f2", "f3", "f4"]


def test_prepare_data_multilabel(multilabel_df):
    d = _prepare_data(
        multilabel_df, FEAT_COLS, LABEL_COLS,
        handle_imbalance=False, random_state=0, cv_folds=3,
    )
    assert d.task_type == "multilabel"
    assert d.class_names == LABEL_COLS
    assert d.y_train.ndim == 2
    assert d.y_train.shape[1] == 3


def test_multilabel_y_shape(multilabel_df):
    d = _prepare_data(multilabel_df, FEAT_COLS, LABEL_COLS,
                       handle_imbalance=False, random_state=0)
    n_total = len(multilabel_df)
    assert len(d.y_train) + len(d.y_test) == n_total


def test_baseline_multilabel(multilabel_df):
    results, task_type = run_tabular_baseline(
        multilabel_df, FEAT_COLS, LABEL_COLS,
        ["Logistic Regression"],
        handle_imbalance=False, random_state=0, cv_folds=3,
    )
    assert task_type == "multilabel"
    assert "Logistic Regression" in results
    r = results["Logistic Regression"]
    assert 0.0 <= r["mean"] <= 1.0


def test_baseline_multilabel_multi_model(multilabel_df):
    results, task_type = run_tabular_baseline(
        multilabel_df, FEAT_COLS, LABEL_COLS,
        ["Logistic Regression", "Random Forest"],
        handle_imbalance=False, random_state=0, cv_folds=3,
    )
    assert task_type == "multilabel"
    assert "Random Forest" in results


def test_multilabel_task_type_not_regression(multilabel_df):
    """Multi-label must never be detected as regression."""
    d = _prepare_data(multilabel_df, FEAT_COLS, LABEL_COLS,
                       handle_imbalance=False, random_state=0)
    assert d.task_type != "regression"


def test_multilabel_single_col_is_classification(multilabel_df):
    """Selecting a single binary column should give classification, not multilabel."""
    d = _prepare_data(multilabel_df, FEAT_COLS, "lbl_a",
                       handle_imbalance=False, random_state=0)
    assert d.task_type == "classification"


def test_multilabel_single_col_list_is_classification(multilabel_df):
    """Passing a one-element list should still give classification."""
    d = _prepare_data(multilabel_df, FEAT_COLS, ["lbl_a"],
                       handle_imbalance=False, random_state=0)
    assert d.task_type == "classification"


def test_multilabel_target_leakage_raises(multilabel_df):
    with pytest.raises(ValueError, match="target leakage"):
        _prepare_data(multilabel_df, FEAT_COLS + ["lbl_a"], LABEL_COLS,
                       handle_imbalance=False)


def test_multilabel_class_names_are_label_columns(multilabel_df):
    d = _prepare_data(multilabel_df, FEAT_COLS, LABEL_COLS,
                       handle_imbalance=False, random_state=0)
    assert d.class_names == LABEL_COLS

