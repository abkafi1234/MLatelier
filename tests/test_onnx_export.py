"""ONNX export: conversion, numerical agreement, and graceful degradation.

These tests are the safety net for the converter registration in
tabular_engine._register_onnx_converters(). Without it, any pipeline ending in
XGBoost or LightGBM silently falls back to .joblib-only export — the failure
mode is a missing file, not an exception, so only a test catches it.
"""

import os

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from imblearn.pipeline import Pipeline as ImbPipeline

from mlatelier.tabular_engine import (
    _register_onnx_converters,
    export_tabular_model_onnx,
)

ort = pytest.importorskip("onnxruntime", reason="onnxruntime not installed")
pytest.importorskip("skl2onnx", reason="skl2onnx not installed")

N_FEATURES = 8
SEED = 0


# ── fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clf_data():
    X, y = make_classification(
        n_samples=200, n_features=N_FEATURES, n_informative=5,
        n_classes=3, random_state=SEED)
    cols = [f"f{i}" for i in range(N_FEATURES)]
    return pd.DataFrame(X, columns=cols), y


@pytest.fixture(scope="module")
def reg_data():
    X, y = make_regression(
        n_samples=200, n_features=N_FEATURES, n_informative=5, random_state=SEED)
    cols = [f"f{i}" for i in range(N_FEATURES)]
    return pd.DataFrame(X, columns=cols), y


def _fit(estimator, X, y):
    pipe = ImbPipeline([("preprocess", StandardScaler()), ("model", estimator)])
    pipe.fit(X, y)
    return pipe


def _run_onnx(path, X):
    """Feed a DataFrame to an exported graph.

    The graph takes one named input per feature column, mirroring the
    ColumnTransformer the pipeline is built around.
    """
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    if len(names) == 1 and names[0] == "float_input":
        feed = {"float_input": X.to_numpy().astype(np.float32)}
    else:
        feed = {n: X[n].to_numpy().astype(np.float32).reshape(-1, 1) for n in names}
    return sess.run(None, feed)


def _export(pipe, tmp_path, name, X):
    return export_tabular_model_onnx(
        pipe, str(tmp_path), name, N_FEATURES, X_sample=X)


# ── registration ──────────────────────────────────────────────────────────────

def test_register_converters_is_idempotent():
    """Registration is global state; calling it twice must not raise."""
    _register_onnx_converters()
    _register_onnx_converters()


# ── sklearn baseline ──────────────────────────────────────────────────────────

def test_sklearn_classifier_exports_and_agrees(tmp_path, clf_data):
    X, y = clf_data
    pipe = _fit(RandomForestClassifier(n_estimators=10, random_state=SEED), X, y)

    path = _export(pipe, tmp_path, "RF", X)
    assert path is not None and os.path.exists(path)

    labels, _ = _run_onnx(path, X)
    agreement = (np.asarray(labels).ravel() == pipe.predict(X)).mean()
    assert agreement > 0.99


def test_sklearn_regressor_exports_and_agrees(tmp_path, reg_data):
    X, y = reg_data
    pipe = _fit(RandomForestRegressor(n_estimators=10, random_state=SEED), X, y)

    path = _export(pipe, tmp_path, "RFR", X)
    assert path is not None and os.path.exists(path)

    preds = np.asarray(_run_onnx(path, X)[0]).ravel()
    np.testing.assert_allclose(preds, pipe.predict(X), rtol=1e-3, atol=1e-3)


def test_classifier_output_is_tensor_not_zipmap(tmp_path, clf_data):
    """Probabilities must come back as an array, not skl2onnx's ZipMap.

    Non-Python runtimes generally cannot consume a sequence-of-maps output,
    which is why the exporter asks for zipmap=False.
    """
    X, y = clf_data
    pipe = _fit(RandomForestClassifier(n_estimators=10, random_state=SEED), X, y)
    path = _export(pipe, tmp_path, "RF_zipmap", X)

    outputs = _run_onnx(path, X)
    assert len(outputs) >= 2
    proba = np.asarray(outputs[1])
    assert proba.ndim == 2, f"expected a 2-D probability tensor, got {proba.shape}"
    assert proba.shape[0] == len(X)


# ── gradient-boosting backends (the reason registration exists) ───────────────

def test_xgboost_classifier_exports(tmp_path, clf_data):
    xgb = pytest.importorskip("xgboost")
    X, y = clf_data
    est = xgb.XGBClassifier(n_estimators=10, max_depth=3, verbosity=0,
                            random_state=SEED)
    pipe = _fit(est, X, y)

    path = _export(pipe, tmp_path, "XGB", X)
    assert path is not None, "XGBoost export failed — converter not registered?"

    labels, _ = _run_onnx(path, X)
    agreement = (np.asarray(labels).ravel() == pipe.predict(X)).mean()
    assert agreement > 0.95


def test_lightgbm_classifier_exports(tmp_path, clf_data):
    lgb = pytest.importorskip("lightgbm")
    X, y = clf_data
    est = lgb.LGBMClassifier(n_estimators=10, max_depth=3, verbose=-1,
                             random_state=SEED)
    pipe = _fit(est, X, y)

    path = _export(pipe, tmp_path, "LGBM", X)
    assert path is not None, "LightGBM export failed — converter not registered?"

    labels, _ = _run_onnx(path, X)
    agreement = (np.asarray(labels).ravel() == pipe.predict(X)).mean()
    assert agreement > 0.95


# ── graceful degradation ──────────────────────────────────────────────────────

def test_catboost_returns_none_without_raising(tmp_path, clf_data):
    """CatBoost has no skl2onnx converter; export must degrade, not explode."""
    cb = pytest.importorskip("catboost")
    X, y = clf_data
    pipe = _fit(cb.CatBoostClassifier(iterations=10, verbose=0,
                                      random_seed=SEED), X, y)

    with pytest.warns(UserWarning):
        path = _export(pipe, tmp_path, "CatBoost", X)
    assert path is None


def test_multilabel_exports(tmp_path):
    """MultiOutputClassifier converts under skl2onnx >= 1.20.

    Kept as a regression guard: if a future skl2onnx drops support, this fails
    loudly instead of silently reverting multi-label users to .joblib-only.
    """
    from sklearn.multioutput import MultiOutputClassifier

    rng = np.random.default_rng(SEED)
    X = pd.DataFrame(rng.normal(size=(120, N_FEATURES)),
                     columns=[f"f{i}" for i in range(N_FEATURES)])
    Y = rng.integers(0, 2, size=(120, 3))

    pipe = _fit(MultiOutputClassifier(
        RandomForestClassifier(n_estimators=5, random_state=SEED)), X, Y)

    path = _export(pipe, tmp_path, "MultiLabel", X)
    assert path is not None and os.path.exists(path)

    labels = np.asarray(_run_onnx(path, X)[0])
    assert labels.shape[0] == len(X)


def test_sampler_step_is_excluded_from_graph(tmp_path, clf_data):
    """The oversampler is training-only and must not appear in the ONNX graph."""
    from imblearn.over_sampling import RandomOverSampler

    X, y = clf_data
    pipe = ImbPipeline([
        ("preprocess", StandardScaler()),
        ("sampler", RandomOverSampler(random_state=SEED)),
        ("model", RandomForestClassifier(n_estimators=10, random_state=SEED)),
    ])
    pipe.fit(X, y)

    path = _export(pipe, tmp_path, "WithSampler", X)
    assert path is not None and os.path.exists(path)

    labels, _ = _run_onnx(path, X)
    assert len(np.asarray(labels).ravel()) == len(X)


# ── engine integration ────────────────────────────────────────────────────────

def test_optimization_populates_all_onnx_paths(tmp_path, clf_data):
    """run_tabular_optimization must surface .onnx paths for the UI."""
    from mlatelier.tabular_engine import (
        run_tabular_baseline, run_tabular_optimization,
    )

    X, y = clf_data
    df = X.copy()
    df["target"] = y
    features = list(X.columns)
    models = ["Random Forest"]

    baseline, _ = run_tabular_baseline(
        df, features, "target", models,
        handle_imbalance=False, cv_folds=3, random_state=SEED)

    results = run_tabular_optimization(
        df, features, "target", models, baseline,
        handle_imbalance=False, cv_folds=3, n_iter=1,
        export_model=True, export_dir=str(tmp_path), random_state=SEED)

    winning_curves = results[9]
    onnx_paths = winning_curves.get("all_onnx_paths", {})
    assert "Random Forest" in onnx_paths
    assert os.path.exists(onnx_paths["Random Forest"])
