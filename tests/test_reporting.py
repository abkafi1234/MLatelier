"""
Tests for reporting.py — every render function is exercised with synthetic data.
These tests run headlessly (no browser) using streamlit's testing utilities.
Each function must complete without raising an exception; visual correctness
is validated separately by human review.
"""
from __future__ import annotations

import sys
import os
import types
import importlib

import numpy as np
import pandas as pd
import pytest

# ── Stub streamlit before import so tests run outside a Streamlit server ─────

def _make_st_stub():
    """Return a minimal streamlit stub that accepts any call without rendering."""
    st = types.ModuleType("streamlit")

    class _FigStub:
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
        def text_area(self, *a, **kw): pass
        def download_button(self, *a, **kw): pass
        def columns(self, *a, **kw): return [self] * 10
        def expander(self, *a, **kw): return self
        def tabs(self, labels): return [self] * len(labels)

    _stub = _FigStub()

    for name in ["markdown", "dataframe", "metric", "pyplot", "image", "caption",
                 "warning", "error", "success", "info", "text_area", "download_button",
                 "container", "expander"]:
        setattr(st, name, lambda *a, _n=name, **kw: None)

    st.columns   = lambda *a, **kw: [_stub] * (a[0] if isinstance(a[0], int) else len(a[0]))
    st.tabs      = lambda labels: [_stub] * len(labels)
    st.container = lambda **kw: _stub
    st.expander  = lambda *a, **kw: _stub
    st.session_state = {}
    st.chat_input = lambda *a, **kw: None
    st.chat_message = lambda *a, **kw: _stub
    return st


sys.modules["streamlit"] = _make_st_stub()

# ── Now import reporting ───────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/mlatelier"))
import reporting as R


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def df_clf():
    rng = np.random.default_rng(0)
    n = 150
    return pd.DataFrame({
        "feat_a": rng.normal(0, 1, n),
        "feat_b": rng.normal(2, 0.5, n),
        "feat_c": rng.exponential(1, n),
        "feat_d": rng.uniform(0, 10, n),
        "label":  rng.choice([0, 1, 2], n),
    })


@pytest.fixture
def df_reg():
    rng = np.random.default_rng(1)
    n = 200
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.uniform(-5, 5, n),
        "x3": rng.exponential(2, n),
        "y":  rng.normal(10, 3, n),
    })


@pytest.fixture
def regression_report():
    rng = np.random.default_rng(2)
    n   = 100
    y_true = rng.normal(10, 2, n)
    y_pred = y_true + rng.normal(0, 0.5, n)
    return {
        "R-Squared (R2)":           "0.9200",
        "Mean Squared Error (MSE)": "0.2500",
        "Root MSE (RMSE)":          "0.5000",
        "Mean Absolute Error (MAE)":"0.4100",
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
    }


@pytest.fixture
def clf_report():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report, confusion_matrix
    X, y = make_classification(n_samples=120, n_features=5, n_classes=3,
                                n_informative=3, n_redundant=1, random_state=0)
    lr = LogisticRegression(max_iter=200, random_state=0).fit(X, y)
    preds = lr.predict(X)
    report = classification_report(y, preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, preds)
    report["confusion_matrix"] = cm.tolist()
    report["confusion_matrix_labels"] = ["A", "B", "C"]
    return report


@pytest.fixture
def df_preds():
    rng = np.random.default_rng(3)
    n = 80
    return pd.DataFrame({
        "input_a":    rng.normal(0, 1, n),
        "Prediction": rng.choice(["cat", "dog", "bird"], n),
        "Confidence": rng.uniform(0.5, 1.0, n),
        "P(cat)":     rng.dirichlet([1, 1, 1], n)[:, 0],
        "P(dog)":     rng.dirichlet([1, 1, 1], n)[:, 1],
        "P(bird)":    rng.dirichlet([1, 1, 1], n)[:, 2],
    })


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_render_top_metrics():
    R.render_top_metrics("92.1%", "88.4%", "+4.2%", "0.031 (Significant)")


def test_render_data_preview(df_clf):
    R.render_data_preview(df_clf)


def test_render_data_preview_none():
    R.render_data_preview(None)


def test_render_classification_report(clf_report):
    R.render_classification_report(clf_report, ["A", "B", "C"])


def test_render_classification_report_no_classes(clf_report):
    R.render_classification_report(clf_report, None)


def test_render_confusion_matrix(clf_report):
    R.render_confusion_matrix({
        "matrix": clf_report["confusion_matrix"],
        "labels": clf_report["confusion_matrix_labels"],
    })


def test_render_regression_report(regression_report):
    R.render_regression_report(regression_report)


def test_render_regression_residuals(regression_report):
    R.render_regression_residuals(regression_report)


def test_render_regression_residuals_empty():
    R.render_regression_residuals({})


def test_render_convergence_plot():
    scores = [0.71, 0.74, 0.76, 0.77, 0.78, 0.79, 0.80]
    R.render_convergence_plot(scores)


def test_render_convergence_plot_empty():
    R.render_convergence_plot([])


def test_render_training_curve():
    losses = [1.2, 0.9, 0.7, 0.6, 0.55]
    f1s    = [0.5, 0.65, 0.72, 0.76, 0.79]
    R.render_training_curve(losses, f1s)


def test_render_feature_importance():
    R.render_feature_importance({"feat_a": 0.35, "feat_b": 0.25, "feat_c": 0.2})


def test_render_feature_importance_empty():
    R.render_feature_importance({})


def test_render_shap_plot():
    # render_shap_plot expects {feature_name: mean_abs_shap} flat dict
    shap_data = {"feat_a": 0.30, "feat_b": 0.20, "feat_c": 0.10}
    R.render_shap_plot(shap_data)


def test_render_roc_curves():
    from sklearn.datasets import make_classification
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_curve, auc
    X, y = make_classification(n_samples=100, n_features=4, random_state=0)
    lr   = LogisticRegression(random_state=0).fit(X, y)
    proba = lr.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    roc_data = {
        "type": "binary",
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "auc": float(auc(fpr, tpr)),
    }
    R.render_roc_curves(roc_data)


def test_render_timing_chart():
    R.render_timing_chart({"RandomForest": 3.5, "XGBoost": 2.1, "LogReg": 0.4})


def test_render_timing_chart_empty():
    R.render_timing_chart({})


def test_render_cv_boxplot():
    R.render_cv_boxplot({
        "RandomForest": [0.82, 0.80, 0.84, 0.81, 0.83],
        "XGBoost":      [0.85, 0.83, 0.86, 0.84, 0.87],
    })


def test_render_cv_boxplot_empty():
    R.render_cv_boxplot({})


def test_render_calibration_curve():
    R.render_calibration_curve({
        "type": "binary",
        "fop":  [0.05, 0.20, 0.40, 0.60, 0.80, 0.95],
        "mpv":  [0.10, 0.22, 0.38, 0.62, 0.79, 0.92],
        "brier_score": 0.042,
    })


def test_render_calibration_curve_empty():
    R.render_calibration_curve({})


def test_render_correlation_heatmap(df_clf):
    R.render_correlation_heatmap(df_clf, ["feat_a", "feat_b", "feat_c", "feat_d"])


def test_render_correlation_heatmap_single_col(df_clf):
    R.render_correlation_heatmap(df_clf, ["feat_a"])


def test_render_data_quality_report(df_clf):
    R.render_data_quality_report(df_clf, ["feat_a", "feat_b", "feat_c"], "label")


def test_render_data_quality_report_no_missing(df_clf):
    R.render_data_quality_report(df_clf, ["feat_a", "feat_b"], None)


def test_render_ablation_chart():
    results = [
        {"Config": "★ Baseline", "CV Score": 0.82, "Std (±)": 0.02, "Time (s)": 1.1, "Delta (%)": "—"},
        {"Config": "Without SMOTE", "CV Score": 0.79, "Std (±)": 0.03, "Time (s)": 0.9, "Delta (%)": "-3.66%"},
        {"Config": "StandardScaler", "CV Score": 0.81, "Std (±)": 0.02, "Time (s)": 1.0, "Delta (%)": "-1.22%"},
    ]
    R.render_ablation_chart(results)


def test_render_prediction_results(df_preds):
    R.render_prediction_results(df_preds)


def test_render_prediction_results_no_confidence():
    df = pd.DataFrame({"Prediction": ["cat", "dog", "cat"]})
    R.render_prediction_results(df)


def test_render_prediction_results_empty():
    R.render_prediction_results(pd.DataFrame())


# ── New render functions ───────────────────────────────────────────────────────

def test_render_feature_distributions(df_clf):
    R.render_feature_distributions(df_clf, ["feat_a", "feat_b", "feat_c"])


def test_render_feature_distributions_empty():
    R.render_feature_distributions(pd.DataFrame(), [])


def test_render_feature_distributions_non_numeric():
    df = pd.DataFrame({"cat": ["a", "b", "c"]})
    R.render_feature_distributions(df, ["cat"])


def test_render_feature_boxplots(df_clf):
    R.render_feature_boxplots(df_clf, ["feat_a", "feat_b", "feat_c", "feat_d"])


def test_render_feature_boxplots_single(df_clf):
    R.render_feature_boxplots(df_clf, ["feat_a"])


def test_render_target_distribution_clf(df_clf):
    R.render_target_distribution(df_clf, "label", "classification")


def test_render_target_distribution_reg(df_reg):
    R.render_target_distribution(df_reg, "y", "regression")


def test_render_target_distribution_missing_col(df_clf):
    R.render_target_distribution(df_clf, "nonexistent", "classification")


def test_render_statistical_summary(df_clf):
    R.render_statistical_summary(df_clf, ["feat_a", "feat_b", "feat_c"])


def test_render_statistical_summary_empty():
    R.render_statistical_summary(pd.DataFrame(), [])


def test_render_missing_heatmap():
    rng = np.random.default_rng(4)
    df = pd.DataFrame({
        "a": rng.normal(size=100),
        "b": [np.nan if rng.random() > 0.7 else v
               for v in rng.normal(size=100)],
        "c": [np.nan if rng.random() > 0.5 else v
               for v in rng.uniform(size=100)],
    })
    R.render_missing_heatmap(df, ["a", "b", "c"])


def test_render_missing_heatmap_no_missing(df_clf):
    R.render_missing_heatmap(df_clf, ["feat_a", "feat_b"])


def test_render_vision_class_distribution():
    R.render_vision_class_distribution({"cat": 120, "dog": 95, "bird": 43})


def test_render_vision_class_distribution_balanced():
    R.render_vision_class_distribution({"A": 100, "B": 100, "C": 100})


def test_render_vision_class_distribution_empty():
    R.render_vision_class_distribution({})


def test_render_pdp():
    xs = list(np.linspace(0, 1, 20))
    R.render_pdp({
        "feat_a": (xs, [0.5 + 0.3 * x for x in xs]),
        "feat_b": (xs, [0.7 - 0.2 * x for x in xs]),
    })


def test_render_pdp_empty():
    R.render_pdp({})


def test_render_shap_waterfall():
    R.render_shap_waterfall({
        "features":    ["feat_a", "feat_b", "feat_c"],
        "shap_values": [0.12, -0.08, 0.05],
        "base_value":  0.60,
    })


def test_render_shap_waterfall_empty():
    R.render_shap_waterfall({})


def test_render_lime_explanation():
    R.render_lime_explanation({
        "features": ["feat_a > 0.5", "feat_b <= 1.2"],
        "weights":  [0.15, -0.09],
        "label":    "cat",
    })


def test_render_lime_explanation_empty():
    R.render_lime_explanation({})


def test_render_learning_curve():
    ts = [50, 100, 150, 200, 250, 300]
    R.render_learning_curve({
        "train_sizes":  ts,
        "train_means":  [0.95, 0.94, 0.93, 0.92, 0.91, 0.90],
        "train_stds":   [0.02] * 6,
        "val_means":    [0.70, 0.73, 0.75, 0.77, 0.78, 0.79],
        "val_stds":     [0.04] * 6,
        "metric_name":  "Macro F1",
    })


def test_render_learning_curve_empty():
    R.render_learning_curve({})


def test_render_feature_ablation_chart():
    results = [
        {"Feature Dropped": "feat_a", "Baseline CV": 0.82, "CV Without": 0.70, "Delta (%)": -14.6, "Time (s)": 0.5},
        {"Feature Dropped": "feat_b", "Baseline CV": 0.82, "CV Without": 0.80, "Delta (%)": -2.4,  "Time (s)": 0.4},
        {"Feature Dropped": "feat_c", "Baseline CV": 0.82, "CV Without": 0.83, "Delta (%)": +1.2,  "Time (s)": 0.5},
    ]
    R.render_feature_ablation_chart(results)


def test_render_feature_ablation_chart_empty():
    R.render_feature_ablation_chart([])


def test_render_confidence_histogram(df_preds):
    R.render_confidence_histogram(df_preds)


def test_render_confidence_histogram_no_col():
    R.render_confidence_histogram(pd.DataFrame({"Prediction": ["cat"]}))


def test_render_per_class_accuracy(clf_report):
    R.render_per_class_accuracy(clf_report, ["A", "B", "C"])


def test_render_per_class_accuracy_empty():
    R.render_per_class_accuracy({})


def test_render_latex_export():
    df = pd.DataFrame({
        "Algorithm": ["RandomForest", "XGBoost"],
        "F1": [0.85, 0.83],
        "Time (s)": [3.2, 2.1],
    })
    R.render_latex_export(df)


def test_render_latex_export_none():
    R.render_latex_export(None)


# ── render_multilabel_report ─────────────────────────────────────────────────

def _multilabel_report():
    return {
        "_task":        "multilabel",
        "label_names":  ["anger", "joy", "sadness"],
        "hamming_loss": "0.2500",
        "micro_f1":     "0.7800",
        "macro_f1":     "0.7600",
        "samples_f1":   "0.7400",
        "anger":   {"precision": 0.80, "recall": 0.75, "f1-score": 0.77, "support": 40},
        "joy":     {"precision": 0.72, "recall": 0.80, "f1-score": 0.76, "support": 35},
        "sadness": {"precision": 0.85, "recall": 0.70, "f1-score": 0.77, "support": 45},
    }


def test_render_multilabel_report_basic():
    R.render_multilabel_report(_multilabel_report())


def test_render_multilabel_report_empty():
    R.render_multilabel_report({})


def test_render_multilabel_report_wrong_task():
    # Should show info and return gracefully
    R.render_multilabel_report({"_task": "classification", "label_names": ["a"]})


def test_render_multilabel_report_no_labels():
    R.render_multilabel_report({"_task": "multilabel", "label_names": []})


# ── render_multilabel_confusion_matrix ──────────────────────────────────────

def test_render_multilabel_confusion_matrix_basic():
    # multilabel_confusion_matrix returns shape (n_labels, 2, 2)
    cm_data = {
        "multilabel": True,
        "labels": ["anger", "joy", "sadness"],
        "matrix": [
            [[30, 5], [8, 27]],   # anger
            [[28, 7], [6, 29]],   # joy
            [[32, 3], [10, 25]],  # sadness
        ],
    }
    R.render_multilabel_confusion_matrix(cm_data)


def test_render_multilabel_confusion_matrix_empty():
    R.render_multilabel_confusion_matrix({})


def test_render_multilabel_confusion_matrix_not_multilabel():
    R.render_multilabel_confusion_matrix({"multilabel": False, "matrix": [], "labels": []})

