"""
Tests for inference_engine.py â€" tabular predict, vision model loading,
batch prediction, folder prediction.
"""
from __future__ import annotations

import sys
import os
import json
import types
import tempfile

import numpy as np
import pandas as pd
import pytest

# â"€â"€ Stub streamlit â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
# This file is collected first alphabetically, so its stub wins the
# sys.modules.setdefault race for all other test files.  It must include
# every streamlit attribute used anywhere in the package.
st_stub = types.ModuleType("streamlit")

class _StColStub:
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
    def write(self, *a, **kw): pass
    def subheader(self, *a, **kw): pass
    def header(self, *a, **kw): pass
    def columns(self, *a, **kw): return [self] * 10
    def expander(self, *a, **kw): return self
    def tabs(self, labels): return [self] * len(labels)

_col_stub = _StColStub()

for _attr in ["info", "success", "warning", "error", "progress", "empty",
              "markdown", "spinner", "caption", "pyplot", "image", "dataframe",
              "metric", "download_button", "container", "subheader", "header",
              "write", "text_area"]:
    setattr(st_stub, _attr, lambda *a, **kw: None)

st_stub.columns      = lambda *a, **kw: [_col_stub] * (a[0] if isinstance(a[0], int) else len(a[0]))
st_stub.tabs         = lambda labels: [_col_stub] * len(labels)
st_stub.expander     = lambda *a, **kw: _col_stub
st_stub.container    = lambda **kw: _col_stub
st_stub.chat_input   = lambda *a, **kw: None
st_stub.chat_message = lambda *a, **kw: _col_stub
st_stub.session_state = {}
sys.modules.setdefault("streamlit", st_stub)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/mlatelier"))
import inference_engine as infer


# â"€â"€ Helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def _make_tabular_pipeline(clf=True):
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    estimator = LogisticRegression(max_iter=200, random_state=0) if clf else Ridge()
    pipe = Pipeline([("sc", StandardScaler()), ("model", estimator)])
    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.uniform(size=n)})
    y = rng.choice([0, 1], n) if clf else rng.normal(size=n)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X, y)
    return pipe


# â"€â"€ load_tabular_model â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def test_load_tabular_model_without_sidecar(tmp_path):
    import joblib
    pipe = _make_tabular_pipeline()
    model_path = str(tmp_path / "model.joblib")
    joblib.dump(pipe, model_path)
    loaded_pipe, meta = infer.load_tabular_model(model_path)
    assert loaded_pipe is not None
    assert isinstance(meta, dict)
    assert meta == {}  # no sidecar


def test_load_tabular_model_with_sidecar(tmp_path):
    import joblib
    pipe = _make_tabular_pipeline()
    model_path = str(tmp_path / "model.joblib")
    # Sidecar must match: model_path.replace(".joblib", "_metadata.json") = "model_metadata.json"
    sidecar_path = model_path.replace(".joblib", "_metadata.json")
    joblib.dump(pipe, model_path)
    meta_data = {"feature_names": ["a", "b"], "class_names": ["neg", "pos"]}
    with open(sidecar_path, "w") as f:
        json.dump(meta_data, f)
    loaded_pipe, meta = infer.load_tabular_model(model_path)
    assert meta.get("feature_names") == ["a", "b"]
    assert meta.get("class_names")   == ["neg", "pos"]


# â"€â"€ predict_tabular â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def test_predict_tabular_binary():
    pipe = _make_tabular_pipeline(clf=True)
    rng  = np.random.default_rng(1)
    df   = pd.DataFrame({"a": rng.normal(size=20), "b": rng.uniform(size=20)})
    result = infer.predict_tabular(pipe, df, ["a", "b"], class_names=["neg", "pos"])
    assert "Prediction" in result.columns
    assert "Confidence" in result.columns
    assert set(result["Prediction"].unique()).issubset({"neg", "pos"})
    assert result["Confidence"].between(0, 1).all()


def test_predict_tabular_regression():
    pipe = _make_tabular_pipeline(clf=False)
    rng  = np.random.default_rng(2)
    df   = pd.DataFrame({"a": rng.normal(size=10), "b": rng.uniform(size=10)})
    result = infer.predict_tabular(pipe, df, ["a", "b"], class_names=None)
    assert "Prediction" in result.columns


def test_predict_tabular_missing_columns():
    import warnings as _w
    pipe = _make_tabular_pipeline()
    df   = pd.DataFrame({"a": [1.0, 2.0]})  # missing "b"
    with _w.catch_warnings(record=True) as w_list:
        _w.simplefilter("always")
        result = infer.predict_tabular(pipe, df, ["a", "b"])
    assert "Prediction" in result.columns
    assert any(
        "missing" in str(w.message).lower() or "filled" in str(w.message).lower()
        for w in w_list
    )


def test_predict_tabular_preserves_original_cols():
    pipe = _make_tabular_pipeline()
    rng  = np.random.default_rng(3)
    df   = pd.DataFrame({
        "extra_col": ["x", "y", "z"],
        "a": rng.normal(size=3),
        "b": rng.uniform(size=3),
    })
    result = infer.predict_tabular(pipe, df, ["a", "b"])
    assert "extra_col" in result.columns


def test_predict_tabular_proba_columns():
    pipe = _make_tabular_pipeline()
    rng  = np.random.default_rng(4)
    df   = pd.DataFrame({"a": rng.normal(size=10), "b": rng.uniform(size=10)})
    result = infer.predict_tabular(pipe, df, ["a", "b"], class_names=["neg", "pos"])
    assert "P(neg)" in result.columns
    assert "P(pos)" in result.columns
    # probabilities should sum to ~1 per row
    prob_sum = result["P(neg)"] + result["P(pos)"]
    assert (prob_sum - 1.0).abs().max() < 0.01


# â"€â"€ predict_vision_batch â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def _make_test_images(folder: str, n: int = 3) -> list:
    """Create tiny RGB PNG files and return their paths."""
    from PIL import Image as PILImage
    paths = []
    for i in range(n):
        img  = PILImage.fromarray(
            (np.random.rand(32, 32, 3) * 255).astype(np.uint8))
        path = os.path.join(folder, f"img_{i}.png")
        img.save(path)
        paths.append(path)
    return paths


def test_predict_vision_batch_paths(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image as PILImage
    paths = _make_test_images(str(tmp_path), n=3)
    class_names = ["cat", "dog"]

    # Build a minimal fake model
    import torch
    import torch.nn as nn

    class _Dummy(nn.Module):
        def forward(self, x): return torch.zeros(x.shape[0], 2)

    model  = _Dummy().eval()
    device = torch.device("cpu")
    result = infer.predict_vision_batch(model, paths, class_names, device)
    assert len(result) == 3
    assert "File"       in result.columns
    assert "Prediction" in result.columns
    assert "Confidence" in result.columns


def test_predict_vision_batch_empty():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    class _Dummy(nn.Module):
        def forward(self, x): return torch.zeros(x.shape[0], 2)

    result = infer.predict_vision_batch(
        _Dummy().eval(), [], ["A", "B"], torch.device("cpu"))
    assert result.empty


# â"€â"€ predict_vision_folder â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

def test_predict_vision_folder(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    import torch
    import torch.nn as nn

    _make_test_images(str(tmp_path), n=4)

    class _Dummy(nn.Module):
        def forward(self, x): return torch.zeros(x.shape[0], 2)

    result = infer.predict_vision_folder(
        _Dummy().eval(), str(tmp_path), ["A", "B"], torch.device("cpu"))
    assert len(result) == 4


def test_predict_vision_folder_no_images(tmp_path):
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    class _Dummy(nn.Module):
        def forward(self, x): return torch.zeros(x.shape[0], 2)

    with pytest.raises(ValueError, match="No images found"):
        infer.predict_vision_folder(
            _Dummy().eval(), str(tmp_path), ["A", "B"], torch.device("cpu"))


def test_predict_vision_folder_nonexistent():
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn

    class _Dummy(nn.Module):
        def forward(self, x): return torch.zeros(x.shape[0], 2)

    with pytest.raises((ValueError, FileNotFoundError)):
        infer.predict_vision_folder(
            _Dummy().eval(), "/nonexistent/path", ["A", "B"], torch.device("cpu"))

