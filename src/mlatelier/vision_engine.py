from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import platform
import random
import time
import warnings
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import chi2
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from skopt import gp_minimize
from skopt.space import Real
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import datasets, models, transforms

try:                       # package import
    from . import checkpoint as _ckpt
except ImportError:        # flat import (tests insert src/mlatelier on sys.path)
    import checkpoint as _ckpt

warnings.filterwarnings("ignore")


# ── Device & seed ─────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> dict:
    dev = get_device()
    info = {"type": dev.type}
    if dev.type == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        info["memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda
    return info


def set_master_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ── Progress helpers ──────────────────────────────────────────────────────────

def _update_progress(progress_bar=None, value: float = 0.0):
    if progress_bar is not None:
        progress_bar.progress(max(0.0, min(1.0, float(value))))


def _update_status(status_text=None, message: str = ""):
    if status_text is not None and message:
        status_text.info(message)


# ── Model registry ────────────────────────────────────────────────────────────

_MODEL_REGISTRY = {
    "ResNet18":         lambda: models.resnet18(weights=models.ResNet18_Weights.DEFAULT),
    "ResNet50":         lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT),
    "VGG16":            lambda: models.vgg16(weights=models.VGG16_Weights.DEFAULT),
    "EfficientNet_B0":  lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT),
    "EfficientNet_V2_S": lambda: models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT),
    "MobileNet_v3":     lambda: models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT),
    "DenseNet121":      lambda: models.densenet121(weights=models.DenseNet121_Weights.DEFAULT),
    "ConvNeXt_T":       lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT),
    "ConvNeXt_S":       lambda: models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT),
    "ViT_B_16":         lambda: models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT),
    "Swin_T":           lambda: models.swin_t(weights=models.Swin_T_Weights.DEFAULT),
    "Swin_S":           lambda: models.swin_s(weights=models.Swin_S_Weights.DEFAULT),
}

# Architectures where Grad-CAM is supported (CNN-based, have conv layers)
_GRADCAM_SUPPORTED = {
    "ResNet18", "ResNet50", "VGG16", "EfficientNet_B0", "EfficientNet_V2_S",
    "MobileNet_v3", "DenseNet121", "ConvNeXt_T", "ConvNeXt_S",
}

# Transformer-based — skip Grad-CAM
_TRANSFORMER_MODELS = {"ViT_B_16", "Swin_T", "Swin_S"}


def get_vision_model(model_name: str) -> nn.Module:
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(_MODEL_REGISTRY)}")
    return _MODEL_REGISTRY[model_name]()


def replace_classification_head(model: nn.Module, model_name: str, num_classes: int) -> nn.Module:
    if model_name in ("ResNet18", "ResNet50"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name in ("VGG16", "EfficientNet_B0", "EfficientNet_V2_S", "MobileNet_v3"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == "DenseNet121":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name in ("ConvNeXt_T", "ConvNeXt_S"):
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif model_name == "ViT_B_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif model_name in ("Swin_T", "Swin_S"):
        model.head = nn.Linear(model.head.in_features, num_classes)
    return model


def _unfreeze_last_block(model: nn.Module, model_name: str) -> nn.Module:
    if model_name in ("ResNet18", "ResNet50"):
        for param in model.layer4.parameters():
            param.requires_grad = True
    elif model_name == "VGG16":
        for param in list(model.features.parameters())[-8:]:
            param.requires_grad = True
    elif model_name in ("EfficientNet_B0", "EfficientNet_V2_S"):
        for param in list(model.features.parameters())[-10:]:
            param.requires_grad = True
    elif model_name == "MobileNet_v3":
        for param in list(model.features.parameters())[-6:]:
            param.requires_grad = True
    elif model_name == "DenseNet121":
        for param in model.features.denseblock4.parameters():
            param.requires_grad = True
    elif model_name in ("ConvNeXt_T", "ConvNeXt_S"):
        for param in list(model.features.parameters())[-4:]:
            param.requires_grad = True
    elif model_name == "ViT_B_16":
        for param in model.encoder.layers[-1].parameters():
            param.requires_grad = True
    elif model_name in ("Swin_T", "Swin_S"):
        for param in model.features[-1].parameters():
            param.requires_grad = True
    return model


# ── Data loading ──────────────────────────────────────────────────────────────

class TransformWrapper(Dataset):
    def __init__(self, subset: Subset, transform=None):
        self.subset    = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def _build_transforms(use_augmentation: bool):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    val_tf = transforms.Compose([transforms.Resize((224, 224)),
                                  transforms.ToTensor(), normalize])
    if use_augmentation:
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_tf = val_tf
    return train_tf, val_tf


def _fingerprint_imagefolder(root: str, samples: list) -> str:
    digest = hashlib.sha256()
    for path, label in sorted(samples):
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        digest.update(rel.encode("utf-8", errors="ignore"))
        digest.update(str(label).encode("utf-8"))
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _class_distribution(labels: list, class_names: list) -> dict:
    counts = np.bincount(labels, minlength=len(class_names))
    return {str(class_names[i]): int(counts[i]) for i in range(len(class_names))}


def _build_vision_loaders(
    data_dir: str,
    use_augmentation: bool = True,
    batch_size: int = 32,
    handle_imbalance: bool = True,
    random_seed: int = 42,
) -> dict:
    train_tf, val_tf = _build_transforms(use_augmentation)
    n_workers = 0 if platform.system() == "Windows" else 2
    device    = get_device()
    pin       = device.type == "cuda"

    dataset_full = datasets.ImageFolder(root=data_dir, transform=None)
    if len(dataset_full.classes) < 2:
        raise ValueError("ZIP must contain at least 2 class subfolders.")
    if len(dataset_full) == 0:
        raise ValueError("No valid images found in the ZIP. Check folder structure.")

    targets     = dataset_full.targets
    num_classes = len(dataset_full.classes)
    class_names = dataset_full.classes

    if np.bincount(targets).min() < 3:
        raise ValueError(
            "Each class needs ≥3 images for train/val/test splitting."
        )

    all_idx = np.arange(len(targets))
    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=0.2, random_state=random_seed, stratify=targets)
    tv_targets = [targets[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.25, random_state=random_seed, stratify=tv_targets)

    train_ds = TransformWrapper(Subset(dataset_full, train_idx), transform=train_tf)
    val_ds   = TransformWrapper(Subset(dataset_full, val_idx),   transform=val_tf)
    test_ds  = TransformWrapper(Subset(dataset_full, test_idx),  transform=val_tf)

    train_targets = [targets[i] for i in train_idx]
    class_counts  = np.bincount(train_targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)

    if handle_imbalance:
        sample_weights = [class_weights[t] for t in train_targets]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                                   num_workers=n_workers, pin_memory=pin)
        loss_weights = None
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=n_workers, pin_memory=pin)
        norm_w = class_weights / class_weights.sum() * len(class_counts)
        loss_weights = torch.FloatTensor(norm_w).to(device)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=n_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=n_workers, pin_memory=pin)

    return {
        "train_loader":      train_loader,
        "val_loader":        val_loader,
        "test_loader":       test_loader,
        "num_classes":       num_classes,
        "class_names":       class_names,
        "loss_weight_tensor": loss_weights,
        "device":            device,
        "metadata": {
            "random_seed": random_seed,
            "dataset_fingerprint_sha256": _fingerprint_imagefolder(
                data_dir, dataset_full.samples),
            "task_type": "classification",
            "preprocessing": {
                "image_size": "224x224",
                "normalization": "ImageNet mean/std",
                "augmentation_enabled": bool(use_augmentation),
                "imbalance_handling": (
                    "WeightedRandomSampler" if handle_imbalance
                    else "CrossEntropyLoss class weights"),
            },
            "split": {
                "train_rows": int(len(train_idx)),
                "validation_rows": int(len(val_idx)),
                "test_rows": int(len(test_idx)),
                "train_class_distribution": _class_distribution(train_targets, class_names),
                "validation_class_distribution": _class_distribution(
                    [targets[i] for i in val_idx], class_names),
                "test_class_distribution": _class_distribution(
                    [targets[i] for i in test_idx], class_names),
            },
        },
    }


# ── LR scheduler factory ──────────────────────────────────────────────────────

def _make_scheduler(optimizer: optim.Optimizer, name: str, epochs: int):
    name = (name or "cosine").lower()
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.5)
    if name == "reduce":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    return None


# ── Training loop ─────────────────────────────────────────────────────────────

def _train_and_evaluate(
    lr: float,
    wd: float,
    run_epochs: int,
    base_clean_model: nn.Module,
    model_name: str,
    num_classes: int,
    freeze_strategy: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    loss_weight_tensor,
    device: torch.device,
    evaluate_each_epoch: bool = True,
    early_stopping_patience: int = 0,
    lr_scheduler: str = "none",
    return_model: bool = False,
    seed: int = 42,
    epoch_callback=None,
    checkpoint_dir: Optional[str] = None,
    checkpoint_tag: str = "",
    resume: bool = True,
) -> tuple:
    """
    Returns:
        (best_val_f1, all_labels, best_preds, epoch_train_losses,
         epoch_val_f1s, fitted_model_or_None)

    When `checkpoint_dir` is set, a checkpoint is written after every epoch and
    a matching one is resumed from on entry, so an interrupted run continues
    from the last completed epoch rather than restarting. The checkpoint is
    deleted once training finishes normally.
    """
    set_master_seed(seed)

    model = copy.deepcopy(base_clean_model)
    for param in model.parameters():
        param.requires_grad = False
    model = replace_classification_head(model, model_name, num_classes)
    if freeze_strategy == "last_block":
        model = _unfreeze_last_block(model, model_name)
    model = model.to(device)

    criterion = (nn.CrossEntropyLoss(weight=loss_weight_tensor)
                 if loss_weight_tensor is not None
                 else nn.CrossEntropyLoss())
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=wd)

    scheduler = _make_scheduler(optimizer, lr_scheduler, run_epochs) if lr_scheduler != "none" else None

    epoch_train_losses: list = []
    epoch_val_f1s:      list = []
    best_val_f1         = -1.0
    best_preds:         list = []
    best_labels:        list = []
    best_weights             = None
    patience_counter         = 0
    start_epoch              = 0

    # ── Checkpoint setup ──────────────────────────────────────────────────
    ckpt_path = None
    if checkpoint_dir:
        # `phase` keeps the three long training calls (per-model, final-test,
        # winner) from colliding: they can share model/lr/wd and differ only in
        # which loader they evaluate against.
        ckpt_cfg = {
            "kind": "vision", "phase": checkpoint_tag or "train",
            "model_name": model_name, "lr": float(lr),
            "wd": float(wd), "run_epochs": int(run_epochs),
            "freeze_strategy": freeze_strategy, "num_classes": int(num_classes),
            "lr_scheduler": lr_scheduler, "seed": int(seed),
        }
        ckpt_path = _ckpt.checkpoint_path(
            f"vision_{model_name}_{checkpoint_tag or 'train'}",
            ckpt_cfg, root=checkpoint_dir)
        if resume:
            saved = _ckpt.load_checkpoint(ckpt_path, ckpt_cfg)
            if saved is not None:
                try:
                    model.load_state_dict(saved["model_state"])
                    optimizer.load_state_dict(saved["optimizer_state"])
                    if scheduler is not None and saved.get("scheduler_state"):
                        scheduler.load_state_dict(saved["scheduler_state"])
                    start_epoch        = int(saved.get("epoch", 0))
                    best_val_f1        = float(saved.get("best_val_f1", -1.0))
                    best_preds         = list(saved.get("best_preds", []))
                    best_labels        = list(saved.get("best_labels", []))
                    best_weights       = saved.get("best_weights")
                    epoch_train_losses = list(saved.get("epoch_train_losses", []))
                    epoch_val_f1s      = list(saved.get("epoch_val_f1s", []))
                    patience_counter   = int(saved.get("patience_counter", 0))
                except Exception as e:
                    warnings.warn(
                        f"Checkpoint for '{model_name}' could not be applied "
                        f"({e}); starting from epoch 0.", UserWarning)
                    start_epoch = 0

    if start_epoch >= run_epochs:
        # Already finished in a previous session; rebuild and return.
        fitted = None
        if return_model and best_weights is not None:
            model.load_state_dict(best_weights)
            fitted = model
        _ckpt.clear_checkpoint(ckpt_path)
        return (best_val_f1, best_labels, best_preds,
                epoch_train_losses, epoch_val_f1s, fitted)

    for epoch in range(start_epoch, run_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_train_losses.append(running_loss / max(len(train_loader.dataset), 1))

        if evaluate_each_epoch or (epoch == run_epochs - 1):
            model.eval()
            ep_preds: list = []
            ep_labels: list = []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    _, preds = torch.max(model(inputs), 1)
                    ep_preds.extend(preds.cpu().numpy())
                    ep_labels.extend(labels.cpu().numpy())

            val_f1 = float(f1_score(ep_labels, ep_preds, average="macro", zero_division=0))
            epoch_val_f1s.append(val_f1)

            if val_f1 > best_val_f1:
                best_val_f1     = val_f1
                best_preds      = ep_preds
                best_labels     = ep_labels
                best_weights    = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch_callback is not None:
                try:
                    epoch_callback(epoch + 1, run_epochs,
                                   epoch_train_losses[-1], val_f1)
                except Exception:
                    pass

            if scheduler is not None and lr_scheduler == "reduce":
                scheduler.step(val_f1)

        # Step the epoch-based schedulers exactly once per epoch. This must sit
        # outside the evaluate/skip branch above: stepping inside it as well
        # advanced cosine annealing twice per epoch whenever evaluation was
        # skipped, so BO trials ran a compressed LR schedule that the final
        # full-training run did not reproduce.
        if scheduler is not None and lr_scheduler != "reduce":
            scheduler.step()

        # Persist progress after each completed epoch.
        if ckpt_path:
            _ckpt.save_checkpoint(ckpt_path, {
                "epoch":              epoch + 1,
                "model_state":        model.state_dict(),
                "optimizer_state":    optimizer.state_dict(),
                "scheduler_state":    scheduler.state_dict() if scheduler is not None else None,
                "best_val_f1":        best_val_f1,
                "best_preds":         [int(p) for p in best_preds],
                "best_labels":        [int(l) for l in best_labels],
                "best_weights":       best_weights,
                "epoch_train_losses": epoch_train_losses,
                "epoch_val_f1s":      epoch_val_f1s,
                "patience_counter":   patience_counter,
            }, config=ckpt_cfg)

        if early_stopping_patience > 0 and patience_counter >= early_stopping_patience:
            break

    fitted = None
    if return_model and best_weights is not None:
        model.load_state_dict(best_weights)
        fitted = model

    # Training completed normally — the checkpoint has served its purpose.
    if ckpt_path:
        _ckpt.clear_checkpoint(ckpt_path)

    if not return_model:
        del model
        torch.cuda.empty_cache()
        gc.collect()

    return best_val_f1, best_labels, best_preds, epoch_train_losses, epoch_val_f1s, fitted


# ── McNemar's test ────────────────────────────────────────────────────────────

def calculate_mcnemar_p_value(y_true, y_base_pred, y_opt_pred) -> float:
    b = sum(1 for t, b, o in zip(y_true, y_base_pred, y_opt_pred) if o == t and b != t)
    c = sum(1 for t, b, o in zip(y_true, y_base_pred, y_opt_pred) if b == t and o != t)
    if b + c == 0:
        return 1.0
    return float(chi2.sf(((abs(b - c) - 1) ** 2) / (b + c), 1))


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def _get_gradcam_layer(model: nn.Module, model_name: str):
    """Return the target conv layer for Grad-CAM, or None for transformers."""
    if model_name in ("ResNet18", "ResNet50"):
        return model.layer4[-1]
    if model_name == "VGG16":
        return model.features[-3]
    if model_name in ("EfficientNet_B0", "EfficientNet_V2_S", "MobileNet_v3"):
        return model.features[-1]
    if model_name in ("ConvNeXt_T", "ConvNeXt_S"):
        return model.features[-1]
    if model_name == "DenseNet121":
        return model.features.denseblock4
    return None


def compute_gradcam(
    model: nn.Module,
    model_name: str,
    image_tensor: torch.Tensor,
    device: torch.device,
    class_idx: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Compute a Grad-CAM heatmap for the given image.
    Returns a (H, W) float32 array in [0, 1], or None for transformers / errors.
    """
    if model_name in _TRANSFORMER_MODELS:
        return None

    target_layer = _get_gradcam_layer(model, model_name)
    if target_layer is None:
        return None

    model.eval()
    img = image_tensor.unsqueeze(0).to(device)
    activations: list = []
    gradients:   list = []

    def fwd_hook(module, input, output):
        activations.append(output.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        output = model(img)
        if class_idx is None:
            class_idx = int(output.argmax(dim=1))
        model.zero_grad()
        output[0, class_idx].backward()

        act  = activations[0].squeeze(0)     # C × H × W
        grad = gradients[0].squeeze(0)       # C × H × W
        weights = grad.mean(dim=(1, 2))      # C

        cam = torch.relu((weights[:, None, None] * act).sum(dim=0))
        cam = cam.cpu().numpy().astype(np.float32)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam

    except Exception as e:
        warnings.warn(f"Grad-CAM failed for '{model_name}': {e}", UserWarning)
        return None
    finally:
        h1.remove()
        h2.remove()


# ── Model export ──────────────────────────────────────────────────────────────

def export_vision_model(
    model: nn.Module,
    export_dir: str,
    model_name: str,
    metadata: Optional[dict] = None,
) -> str:
    """Save model state dict as .pt; optionally write metadata JSON."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in model_name)
    path = os.path.join(export_dir, f"MLatelier_{safe}.pt")
    torch.save(model.state_dict(), path)
    if metadata:
        with open(os.path.join(export_dir, f"MLatelier_{safe}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4, default=str)
    return path


# ── Public API: baseline ──────────────────────────────────────────────────────

def run_vision_baseline(
    data_dir: str,
    selected_models: list,
    batch_size: int = 32,
    use_augmentation: bool = True,
    freeze_strategy: str = "head_only",
    handle_imbalance: bool = True,
    baseline_epochs: int = 3,
    progress_bar=None,
    status_text=None,
    random_seed: int = 42,
    epoch_callback=None,
    model_callback=None,
) -> tuple:
    """
    Fixed-hyperparameter baseline (lr=1e-3, wd=0) for `baseline_epochs`.
    Returns (baseline_results, class_names).
    baseline_results: {model_name: {"base_score": float, "y_true": list,
                                     "base_preds": list, "time_s": float}}
    epoch_callback(epoch, total_ep, loss, f1) — called each epoch.
    model_callback(name, score, time_s)       — called after each architecture.
    """
    set_master_seed(random_seed)
    ctx = _build_vision_loaders(data_dir, use_augmentation=use_augmentation,
                                 batch_size=batch_size, handle_imbalance=handle_imbalance,
                                 random_seed=random_seed)

    baseline_results: dict = {}
    total = max(len(selected_models), 1)

    for idx, model_name in enumerate(selected_models, 1):
        _update_status(status_text, f"Baseline {idx}/{total}: {model_name}")
        base_clean = get_vision_model(model_name).cpu()
        t0 = time.perf_counter()
        score, y_true, preds, _, _, _ = _train_and_evaluate(
            lr=1e-3, wd=0.0, run_epochs=baseline_epochs,
            base_clean_model=base_clean, model_name=model_name,
            num_classes=ctx["num_classes"], freeze_strategy=freeze_strategy,
            train_loader=ctx["train_loader"], val_loader=ctx["val_loader"],
            loss_weight_tensor=ctx["loss_weight_tensor"], device=ctx["device"],
            seed=random_seed, epoch_callback=epoch_callback,
        )
        elapsed = round(time.perf_counter() - t0, 2)
        baseline_results[model_name] = {
            "base_score": score, "y_true": y_true,
            "base_preds": preds, "time_s": elapsed,
        }
        if model_callback is not None:
            try:
                model_callback(model_name, score, elapsed)
            except Exception:
                pass
        del base_clean
        torch.cuda.empty_cache()
        gc.collect()
        _update_progress(progress_bar, idx / total)

    return baseline_results, ctx["class_names"]


# ── Public API: optimisation ──────────────────────────────────────────────────

def run_vision_optimization(
    data_dir: str,
    selected_models: list,
    baseline_results: dict,
    epochs: int = 10,
    batch_size: int = 32,
    use_augmentation: bool = True,
    freeze_strategy: str = "head_only",
    handle_imbalance: bool = True,
    n_bo_calls: int = 20,
    early_stopping_patience: int = 5,
    lr_scheduler: str = "cosine",
    export_model: bool = True,
    export_dir: Optional[str] = None,
    progress_bar=None,
    status_text=None,
    random_seed: int = 42,
    epoch_callback=None,
    model_callback=None,
    checkpoint_dir: Optional[str] = None,
) -> tuple:
    """
    Bayesian Optimisation over lr / weight_decay for each model, then full
    retraining of winner. Evaluates ALL models on test set.

    Returns 10-tuple:
      pipeline_results, best_f1, avg_base, improvement, p_val_str,
      winning_params, winning_report, class_names, winning_curves, exported_path
    epoch_callback(epoch, total_ep, loss, f1) — called each epoch of full training.
    model_callback(name, opt_score, time_s)   — called after each architecture finishes.
    """
    set_master_seed(random_seed)
    ctx = _build_vision_loaders(data_dir, use_augmentation=use_augmentation,
                                 batch_size=batch_size, handle_imbalance=handle_imbalance,
                                 random_seed=random_seed)

    search_space = [
        Real(1e-5, 1e-2, prior="log-uniform", name="learning_rate"),
        Real(1e-6, 1e-3, prior="log-uniform", name="weight_decay"),
    ]

    pipeline_results:    list    = []
    best_sel_f1          = -float("inf")
    winning_model_name   = None
    winning_lr = winning_wd = None
    winning_params       = None
    winning_curves       = None

    total_models  = max(len(selected_models), 1)
    total_steps   = total_models * (n_bo_calls + 1) + 1
    completed     = 0

    model_records: dict = {}

    for midx, model_name in enumerate(selected_models, 1):
        _update_status(status_text, f"Optimizing {midx}/{total_models}: {model_name}")
        base_clean = get_vision_model(model_name).cpu()
        model_t0 = time.perf_counter()

        bl         = baseline_results.get(model_name, {})
        base_score = bl.get("base_score", 0.0)

        trial_n    = {"v": 0}

        def objective(params, _m=base_clean, _n=model_name):
            nonlocal completed
            score, _, _, _, _, _ = _train_and_evaluate(
                lr=params[0], wd=params[1], run_epochs=3,
                base_clean_model=_m, model_name=_n,
                num_classes=ctx["num_classes"], freeze_strategy=freeze_strategy,
                train_loader=ctx["train_loader"], val_loader=ctx["val_loader"],
                loss_weight_tensor=ctx["loss_weight_tensor"], device=ctx["device"],
                seed=random_seed,
            )
            trial_n["v"] += 1
            completed += 1
            _update_status(status_text,
                           f"Optimizing {_n}: trial {trial_n['v']}/{n_bo_calls}")
            _update_progress(progress_bar, completed / total_steps)
            return -score

        res = gp_minimize(objective, search_space, n_calls=n_bo_calls,
                          n_initial_points=3, random_state=random_seed)
        opt_lr, opt_wd = res.x

        bo_scores  = [-v for v in res.func_vals]
        bo_history = [
            {"trial": i + 1,
             "learning_rate": float(res.x_iters[i][0]),
             "weight_decay":  float(res.x_iters[i][1]),
             "score":         float(bo_scores[i])}
            for i in range(len(bo_scores))
        ]

        # Val F1 after full training with optimal HPs
        opt_score, _, _, train_losses, val_f1s, _ = _train_and_evaluate(
            lr=opt_lr, wd=opt_wd, run_epochs=epochs,
            base_clean_model=base_clean, model_name=model_name,
            num_classes=ctx["num_classes"], freeze_strategy=freeze_strategy,
            train_loader=ctx["train_loader"], val_loader=ctx["val_loader"],
            loss_weight_tensor=ctx["loss_weight_tensor"], device=ctx["device"],
            early_stopping_patience=early_stopping_patience,
            lr_scheduler=lr_scheduler, seed=random_seed,
            epoch_callback=epoch_callback,
            checkpoint_dir=checkpoint_dir, checkpoint_tag="opt",
        )
        completed += 1
        _update_progress(progress_bar, completed / total_steps)

        model_elapsed = round(time.perf_counter() - model_t0, 2)
        pipeline_results.append({
            "Architecture":       model_name,
            "Baseline Val F1":    round(base_score, 4),
            "Optimized Val F1":   round(opt_score, 4),
            "Final Test F1":      None,
            "Training Time (s)":  model_elapsed,
        })
        if model_callback is not None:
            try:
                model_callback(model_name, opt_score, model_elapsed)
            except Exception:
                pass

        model_records[model_name] = {
            "base_score":  base_score,
            "opt_score":   opt_score,
            "opt_lr":      opt_lr,
            "opt_wd":      opt_wd,
            "train_losses": train_losses,
            "val_f1s":     val_f1s,
            "bo_scores":   bo_scores,
            "bo_history":  bo_history,
        }

        if opt_score > best_sel_f1:
            best_sel_f1        = opt_score
            winning_model_name = model_name

        del base_clean
        torch.cuda.empty_cache()
        gc.collect()

    # ── Early exit ────────────────────────────────────────────────────────────
    if winning_model_name is None:
        return (None,) * 10

    # ── Evaluate ALL models on test set ──────────────────────────────────────
    for item in pipeline_results:
        mn = item["Architecture"]
        rec = model_records[mn]
        test_clean = get_vision_model(mn).cpu()

        test_f1, _, _, _, _, _ = _train_and_evaluate(
            lr=rec["opt_lr"], wd=rec["opt_wd"], run_epochs=epochs,
            base_clean_model=test_clean, model_name=mn,
            num_classes=ctx["num_classes"], freeze_strategy=freeze_strategy,
            train_loader=ctx["train_loader"], val_loader=ctx["test_loader"],
            loss_weight_tensor=ctx["loss_weight_tensor"], device=ctx["device"],
            early_stopping_patience=early_stopping_patience,
            lr_scheduler=lr_scheduler, seed=random_seed,
            evaluate_each_epoch=False,
            checkpoint_dir=checkpoint_dir, checkpoint_tag="final_test",
        )
        item["Final Test F1"] = round(test_f1, 4)
        model_records[mn]["test_f1"] = test_f1
        del test_clean
        torch.cuda.empty_cache()
        gc.collect()

    # ── Winner: full retraining to get preds + model object ──────────────────
    wr   = model_records[winning_model_name]
    base_clean = get_vision_model(winning_model_name).cpu()

    # Baseline test preds for McNemar
    base_test_f1, base_true, base_preds, _, _, _ = _train_and_evaluate(
        lr=1e-3, wd=0.0, run_epochs=3,
        base_clean_model=base_clean, model_name=winning_model_name,
        num_classes=ctx["num_classes"], freeze_strategy=freeze_strategy,
        train_loader=ctx["train_loader"], val_loader=ctx["test_loader"],
        loss_weight_tensor=ctx["loss_weight_tensor"], device=ctx["device"],
        evaluate_each_epoch=False, seed=random_seed,
    )

    # Optimized test preds + keep fitted model for Grad-CAM / export
    opt_test_f1, win_true, win_preds, _, _, fitted_model = _train_and_evaluate(
        lr=wr["opt_lr"], wd=wr["opt_wd"], run_epochs=epochs,
        base_clean_model=base_clean, model_name=winning_model_name,
        num_classes=ctx["num_classes"], freeze_strategy=freeze_strategy,
        train_loader=ctx["train_loader"], val_loader=ctx["test_loader"],
        loss_weight_tensor=ctx["loss_weight_tensor"], device=ctx["device"],
        early_stopping_patience=early_stopping_patience,
        lr_scheduler=lr_scheduler, seed=random_seed,
        evaluate_each_epoch=False, return_model=True,
        checkpoint_dir=checkpoint_dir, checkpoint_tag="winner",
    )
    del base_clean
    torch.cuda.empty_cache()
    gc.collect()

    winning_test_f1      = model_records[winning_model_name]["test_f1"]
    winning_base_test_f1 = base_test_f1

    # ── Metrics ────────────────────────────────────────────────────────────────
    winning_report = classification_report(win_true, win_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(win_true, win_preds)
    winning_report["confusion_matrix"]        = cm.tolist()
    winning_report["confusion_matrix_labels"] = ctx["class_names"]

    p_val_raw = calculate_mcnemar_p_value(base_true, base_preds, win_preds)
    p_val_str = (f"{p_val_raw:.3f} (Significant)" if p_val_raw < 0.05
                 else f"{p_val_raw:.3f} (Not Sig.)")

    avg_base    = float(np.mean([r["Baseline Val F1"] for r in pipeline_results]))
    improvement = (
        ((winning_test_f1 - winning_base_test_f1) / winning_base_test_f1) * 100
        if winning_base_test_f1 and winning_base_test_f1 > 0 else 0.0
    )

    winning_params = {
        "learning_rate":  round(wr["opt_lr"], 6),
        "weight_decay":   round(wr["opt_wd"], 8),
        "lr_scheduler":   lr_scheduler,
        "epochs":         epochs,
        "early_stopping_patience": early_stopping_patience,
    }

    # ── Grad-CAM sample ────────────────────────────────────────────────────────
    gradcam_data = None
    if fitted_model is not None and winning_model_name in _GRADCAM_SUPPORTED:
        try:
            sample_batch = next(iter(ctx["test_loader"]))
            sample_img   = sample_batch[0][0]
            pred_cls     = int(win_preds[0]) if win_preds else 0
            cam = compute_gradcam(fitted_model, winning_model_name,
                                   sample_img, ctx["device"], class_idx=pred_cls)
            if cam is not None:
                gradcam_data = {
                    "heatmap":   cam.tolist(),
                    "image":     sample_img.cpu().numpy().tolist(),
                    "pred_class": pred_cls,
                    "class_name": (ctx["class_names"][pred_cls]
                                   if pred_cls < len(ctx["class_names"]) else str(pred_cls)),
                }
        except Exception as e:
            warnings.warn(f"Grad-CAM sample collection failed: {e}", UserWarning)

    # ── Export ─────────────────────────────────────────────────────────────────
    exported_path = None
    if export_model and export_dir and fitted_model is not None:
        try:
            os.makedirs(export_dir, exist_ok=True)
            meta = {
                "model_name":    winning_model_name,
                "num_classes":   ctx["num_classes"],
                "class_names":   ctx["class_names"],
                "best_params":   winning_params,
                "test_f1":       winning_test_f1,
                "freeze_strategy": freeze_strategy,
            }
            exported_path = export_vision_model(
                fitted_model, export_dir, winning_model_name, meta)
        except Exception as e:
            warnings.warn(f"Vision model export failed: {e}", UserWarning)

    cm_data = {"matrix": cm.tolist(), "labels": ctx["class_names"]}

    winning_curves = {
        "train_losses":         wr["train_losses"],
        "val_f1s":              wr["val_f1s"],
        "bo_scores":            wr["bo_scores"],
        "bo_history":           wr["bo_history"],
        "gradcam":              gradcam_data,
        "confusion_matrix":     cm_data,
        "exported_model_path":  exported_path,
        "reproducibility_metadata": {
            **ctx["metadata"],
            "model_selection": {
                "winner_selected_by": "validation Macro F1 after Bayesian optimization",
                "final_test_used_for": "reporting only (holdout never seen during search)",
                "winning_validation_f1": float(best_sel_f1),
                "winning_test_f1": float(winning_test_f1),
            },
            "bayesian_optimization": {
                "library": "skopt.gp_minimize",
                "search_space": {
                    "learning_rate": "Real(1e-5, 1e-2, log-uniform)",
                    "weight_decay": "Real(1e-6, 1e-3, log-uniform)",
                },
                "n_calls": n_bo_calls,
                "n_initial_points": 3,
            },
            "training": {
                "full_epochs": int(epochs),
                "batch_size": int(batch_size),
                "freeze_strategy": freeze_strategy,
                "lr_scheduler": lr_scheduler,
                "early_stopping_patience": early_stopping_patience,
            },
        },
    }

    return (
        pipeline_results,
        winning_test_f1,
        avg_base,
        improvement,
        p_val_str,
        winning_params,
        winning_report,
        ctx["class_names"],
        winning_curves,
        exported_path,
    )


# ── Vision dataset summary ────────────────────────────────────────────────────

def get_vision_dataset_summary(data_dir: str, n_samples_per_class: int = 4) -> dict:
    """
    Scan a folder-per-class dataset and return:
      class_counts: {class_name: int}
      samples:      {class_name: [path, ...]}  (up to n_samples_per_class)
    """
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}
    class_counts: dict = {}
    samples:      dict = {}

    if not os.path.isdir(data_dir):
        return {"class_counts": {}, "samples": {}}

    for entry in sorted(os.scandir(data_dir), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        cls_name = entry.name
        imgs = sorted([
            os.path.join(entry.path, fn)
            for fn in os.listdir(entry.path)
            if os.path.splitext(fn)[1].lower() in _IMG_EXTS
        ])
        if not imgs:
            continue
        class_counts[cls_name] = len(imgs)
        import random
        random.seed(42)
        samples[cls_name] = random.sample(imgs, min(n_samples_per_class, len(imgs)))

    return {"class_counts": class_counts, "samples": samples}

