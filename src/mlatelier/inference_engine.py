"""
inference_engine.py — Load saved MLatelier models and predict on new data.
Mirrors Weka's "Classify → Use supplied test set" pattern.
"""
from __future__ import annotations

import json
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd


# ── Tabular inference ─────────────────────────────────────────────────────────

def load_tabular_model(model_path: str):
    """Load a saved ImbPipeline (.joblib) and its JSON metadata sidecar.
    Returns (pipeline, metadata_dict).
    """
    import joblib
    pipeline = joblib.load(model_path)

    meta_path = model_path.replace(".joblib", "_metadata.json")
    metadata: dict = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            metadata = json.load(f)

    return pipeline, metadata


def predict_tabular(
    pipeline,
    df: pd.DataFrame,
    feature_names: list,
    class_names: Optional[list] = None,
) -> pd.DataFrame:
    """
    Run pipeline.predict (and predict_proba if available) on df.
    Returns a DataFrame with all original columns + Prediction [+ Confidence
    + per-class probability columns].
    Missing expected feature columns are added and filled with 0, and a
    UserWarning naming them is emitted.
    """
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        warnings.warn(
            f"New data is missing these expected columns (filled with 0): {missing}",
            UserWarning,
        )

    X = df.reindex(columns=feature_names).fillna(0)

    preds = pipeline.predict(X)

    result = df.copy()
    if class_names:
        result["Prediction"] = [
            class_names[int(p)] if 0 <= int(p) < len(class_names) else str(p)
            for p in preds
        ]
    else:
        result["Prediction"] = preds

    if hasattr(pipeline, "predict_proba"):
        try:
            proba = pipeline.predict_proba(X)
            result["Confidence"] = np.max(proba, axis=1).round(4)
            if class_names:
                for i, cn in enumerate(class_names):
                    if i < proba.shape[1]:
                        result[f"P({cn})"] = proba[:, i].round(4)
        except Exception:
            pass
    elif hasattr(pipeline, "decision_function"):
        try:
            df_scores = pipeline.decision_function(X)
            result["Decision Score"] = (
                df_scores.round(4) if df_scores.ndim == 1
                else df_scores[:, 0].round(4)
            )
        except Exception:
            pass

    return result


# ── Vision inference ──────────────────────────────────────────────────────────

def load_vision_model(
    model_path: str,
    metadata_path: Optional[str] = None,
):
    """
    Load a MLatelier vision model (.pt state dict) and its JSON metadata.
    Returns (model, metadata, device).
    """
    import torch
    from vision_engine import (
        get_device,
        get_vision_model,
        replace_classification_head,
    )

    if metadata_path is None:
        metadata_path = model_path.replace(".pt", "_metadata.json")

    metadata: dict = {}
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)

    model_name  = metadata.get("model_name", "ResNet18")
    num_classes = int(metadata.get("num_classes", 2))

    device = get_device()
    net    = get_vision_model(model_name)
    for param in net.parameters():
        param.requires_grad = False
    net = replace_classification_head(net, model_name, num_classes)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net = net.to(device)
    net.eval()

    return net, metadata, device


def _predict_single_image(model, image_source, class_names: list, device) -> dict:
    """Predict one image (file path, bytes IO, or PIL Image)."""
    import torch
    from PIL import Image as PILImage
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if isinstance(image_source, str):
        img = PILImage.open(image_source).convert("RGB")
    elif hasattr(image_source, "read"):
        img = PILImage.open(image_source).convert("RGB")
    else:
        img = image_source.convert("RGB")

    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        proba  = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx   = int(np.argmax(proba))
    pred_class = (class_names[pred_idx]
                  if pred_idx < len(class_names) else str(pred_idx))
    row = {"Prediction": pred_class,
           "Confidence": round(float(proba[pred_idx]), 4)}
    for i, cn in enumerate(class_names):
        row[f"P({cn})"] = round(float(proba[i]), 4)
    return row


def predict_vision_batch(
    model,
    image_items,
    class_names: list,
    device,
) -> pd.DataFrame:
    """
    Predict a list of images.
    image_items: list of (name, file_object_or_path) tuples, or list of file paths.
    Returns DataFrame with File, Prediction, Confidence, P(class) columns.
    """
    rows = []
    for item in image_items:
        if isinstance(item, tuple):
            name, src = item
        else:
            name, src = os.path.basename(str(item)), item
        try:
            row = _predict_single_image(model, src, class_names, device)
            row["File"] = name
            rows.append(row)
        except Exception as e:
            rows.append({"File": name, "Prediction": "ERROR",
                          "Confidence": 0.0, "Error": str(e)})

    if not rows:
        return pd.DataFrame()

    cols = (["File", "Prediction", "Confidence"]
            + [f"P({cn})" for cn in class_names])
    df   = pd.DataFrame(rows)
    ordered = [c for c in cols if c in df.columns] + [
        c for c in df.columns if c not in cols]
    return df[ordered]


def predict_vision_folder(
    model,
    folder_path: str,
    class_names: list,
    device,
) -> pd.DataFrame:
    """Predict all images found recursively inside folder_path."""
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
    image_paths = []
    for root, _, files in os.walk(folder_path):
        for fn in files:
            if os.path.splitext(fn)[1].lower() in _IMG_EXTS:
                image_paths.append(os.path.join(root, fn))

    if not image_paths:
        raise ValueError(f"No images found in folder: {folder_path}")

    items = [(os.path.relpath(p, folder_path).replace("\\", "/"), p)
             for p in sorted(image_paths)]
    return predict_vision_batch(model, items, class_names, device)

