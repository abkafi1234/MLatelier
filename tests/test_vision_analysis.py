"""
Tests for vision_engine.py analysis utilities:
  - get_vision_dataset_summary
  - get_device_info / get_device
  - Vision model construction helpers
"""
from __future__ import annotations

import sys
import os
import types
import tempfile

import numpy as np
import pytest

# ── Stub streamlit ────────────────────────────────────────────────────────────
st_stub = types.ModuleType("streamlit")
for _attr in ["info", "success", "warning", "error", "progress",
              "empty", "markdown", "spinner"]:
    setattr(st_stub, _attr, lambda *a, **kw: None)
st_stub.session_state = {}
sys.modules.setdefault("streamlit", st_stub)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/mlatelier"))

# Import vision_engine only after stubs
from vision_engine import get_vision_dataset_summary, get_device_info, get_device


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_image_dataset(root: str, classes: dict[str, int]):
    """Create a folder-per-class image dataset with tiny black PNG files."""
    from PIL import Image as PILImage
    for cls_name, n in classes.items():
        cls_dir = os.path.join(root, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(n):
            img  = PILImage.fromarray(
                (np.random.rand(16, 16, 3) * 255).astype(np.uint8))
            img.save(os.path.join(cls_dir, f"img_{i}.jpg"))


# ── get_vision_dataset_summary ────────────────────────────────────────────────

def test_summary_counts(tmp_path):
    pytest.importorskip("PIL")
    _make_image_dataset(str(tmp_path), {"cat": 10, "dog": 7, "bird": 4})
    result = get_vision_dataset_summary(str(tmp_path))
    counts = result["class_counts"]
    assert counts["cat"]  == 10
    assert counts["dog"]  == 7
    assert counts["bird"] == 4


def test_summary_samples_capped(tmp_path):
    pytest.importorskip("PIL")
    _make_image_dataset(str(tmp_path), {"cat": 20})
    result = get_vision_dataset_summary(str(tmp_path), n_samples_per_class=4)
    assert len(result["samples"]["cat"]) == 4


def test_summary_samples_less_than_n(tmp_path):
    pytest.importorskip("PIL")
    _make_image_dataset(str(tmp_path), {"cat": 2})
    result = get_vision_dataset_summary(str(tmp_path), n_samples_per_class=4)
    assert len(result["samples"]["cat"]) == 2


def test_summary_empty_folder(tmp_path):
    result = get_vision_dataset_summary(str(tmp_path))
    assert result["class_counts"] == {}
    assert result["samples"]      == {}


def test_summary_nonexistent_path():
    result = get_vision_dataset_summary("/path/does/not/exist/xyz")
    assert result["class_counts"] == {}


def test_summary_ignores_non_image_files(tmp_path):
    pytest.importorskip("PIL")
    cls_dir = os.path.join(str(tmp_path), "cat")
    os.makedirs(cls_dir, exist_ok=True)
    # Write one real image and some non-image files
    from PIL import Image as PILImage
    img = PILImage.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    img.save(os.path.join(cls_dir, "real.jpg"))
    open(os.path.join(cls_dir, "readme.txt"), "w").close()
    open(os.path.join(cls_dir, "data.csv"),   "w").close()
    result = get_vision_dataset_summary(str(tmp_path))
    assert result["class_counts"]["cat"] == 1


def test_summary_sample_paths_exist(tmp_path):
    pytest.importorskip("PIL")
    _make_image_dataset(str(tmp_path), {"dog": 5})
    result = get_vision_dataset_summary(str(tmp_path))
    for path in result["samples"]["dog"]:
        assert os.path.exists(path), f"Sample path missing: {path}"


def test_summary_class_ordering(tmp_path):
    pytest.importorskip("PIL")
    _make_image_dataset(str(tmp_path), {"zebra": 3, "alpha": 5, "monkey": 2})
    result = get_vision_dataset_summary(str(tmp_path))
    # Should be sorted alphabetically by class name
    keys = list(result["class_counts"].keys())
    assert keys == sorted(keys)


def test_summary_multiple_image_extensions(tmp_path):
    pytest.importorskip("PIL")
    cls_dir = os.path.join(str(tmp_path), "mixed")
    os.makedirs(cls_dir, exist_ok=True)
    from PIL import Image as PILImage
    for ext, color in [("jpg", "R"), ("png", "G"), ("bmp", "B")]:
        arr  = np.zeros((8, 8, 3), dtype=np.uint8)
        img  = PILImage.fromarray(arr)
        img.save(os.path.join(cls_dir, f"img.{ext}"))
    result = get_vision_dataset_summary(str(tmp_path))
    assert result["class_counts"]["mixed"] == 3


# ── get_device_info ───────────────────────────────────────────────────────────

def test_get_device_info_returns_dict():
    pytest.importorskip("torch")
    info = get_device_info()
    assert isinstance(info, dict)
    assert "type" in info
    assert info["type"] in ("cuda", "mps", "cpu")


def test_get_device_returns_device():
    pytest.importorskip("torch")
    import torch
    dev = get_device()
    assert isinstance(dev, torch.device)


# ── get_vision_model ─────────────────────────────────────────────────────────

def test_get_vision_model_resnet18():
    pytest.importorskip("torchvision")
    from vision_engine import get_vision_model
    model = get_vision_model("ResNet18")
    assert model is not None


def test_get_vision_model_unknown():
    from vision_engine import get_vision_model
    with pytest.raises((KeyError, ValueError, Exception)):
        get_vision_model("NonExistentArch999")


# ── replace_classification_head ───────────────────────────────────────────────

def test_replace_head_resnet18():
    pytest.importorskip("torchvision")
    from vision_engine import get_vision_model, replace_classification_head
    model = get_vision_model("ResNet18")
    model = replace_classification_head(model, "ResNet18", num_classes=5)
    import torch
    x   = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 5)


def test_replace_head_efficientnet():
    pytest.importorskip("torchvision")
    from vision_engine import get_vision_model, replace_classification_head
    model = get_vision_model("EfficientNet_B0")
    model = replace_classification_head(model, "EfficientNet_B0", num_classes=3)
    import torch
    out = model(torch.randn(1, 3, 224, 224))
    assert out.shape == (1, 3)

