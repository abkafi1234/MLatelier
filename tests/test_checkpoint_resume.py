"""End-to-end resume: interrupt a real training loop, restart, finish.

test_checkpoint.py covers the storage layer in isolation. This file drives the
actual vision training loop, kills it mid-run, and asserts the restart picks up
where it stopped instead of redoing completed epochs — the behaviour a user
actually cares about when a 40-minute fine-tune dies at minute 30.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/mlatelier"))

torch = pytest.importorskip("torch", reason="torch not installed")
import torch.nn as nn                                       # noqa: E402
from torch.utils.data import DataLoader, TensorDataset      # noqa: E402

import vision_engine as ve                                  # noqa: E402

N_FEATURES = 16
N_CLASSES = 3
SEED = 0


class TinyNet(nn.Module):
    """Minimal stand-in for a torchvision backbone.

    Named `DenseNet121` at the call site so replace_classification_head swaps
    `.classifier` and leaves one trainable parameter group — otherwise
    freeze_strategy="head_only" would freeze everything and Adam would reject an
    empty parameter list.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(N_FEATURES, N_CLASSES)

    def forward(self, x):
        return self.classifier(self.flatten(x))


class ExplodingLoader:
    """Wraps a DataLoader and raises once it has been iterated `fail_on` times."""

    def __init__(self, loader, fail_on):
        self.loader = loader
        self.fail_on = fail_on
        self.n_iters = 0

    def __iter__(self):
        self.n_iters += 1
        if self.n_iters >= self.fail_on:
            raise RuntimeError("simulated crash mid-training")
        return iter(self.loader)

    def __len__(self):
        return len(self.loader)

    @property
    def dataset(self):
        return self.loader.dataset


@pytest.fixture
def loaders():
    g = torch.Generator().manual_seed(SEED)
    X = torch.randn(48, N_FEATURES, generator=g)
    y = torch.randint(0, N_CLASSES, (48,), generator=g)
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=16), DataLoader(ds, batch_size=16)


def _train(tmp_path, train_loader, val_loader, run_epochs, seen_epochs):
    return ve._train_and_evaluate(
        lr=1e-3, wd=0.0, run_epochs=run_epochs,
        base_clean_model=TinyNet(), model_name="DenseNet121",
        num_classes=N_CLASSES, freeze_strategy="head_only",
        train_loader=train_loader, val_loader=val_loader,
        loss_weight_tensor=None, device=torch.device("cpu"),
        lr_scheduler="cosine", seed=SEED,
        epoch_callback=lambda e, t, l, f: seen_epochs.append(e),
        checkpoint_dir=str(tmp_path), checkpoint_tag="test",
    )


def test_interrupted_run_resumes_from_last_epoch(tmp_path, loaders):
    train_loader, val_loader = loaders

    # ── Run 1: crash during the 3rd epoch's validation pass ───────────────
    first_pass: list = []
    crashing_val = ExplodingLoader(val_loader, fail_on=3)

    with pytest.raises(RuntimeError, match="simulated crash"):
        _train(tmp_path, train_loader, crashing_val, 4, first_pass)

    assert first_pass == [1, 2], f"expected 2 completed epochs, got {first_pass}"

    ckpts = [f for f in os.listdir(tmp_path) if f.endswith(".pt")]
    assert len(ckpts) == 1, "a checkpoint should survive the crash"

    # ── Run 2: same configuration, healthy loaders ────────────────────────
    second_pass: list = []
    best_f1, labels, preds, losses, f1s, _ = _train(
        tmp_path, train_loader, val_loader, 4, second_pass)

    # Only the outstanding epochs run again.
    assert second_pass == [3, 4], f"expected resume at epoch 3, got {second_pass}"

    # History from before the crash is carried forward, not discarded.
    assert len(losses) == 4, f"expected 4 epoch losses, got {len(losses)}"
    assert len(f1s) == 4

    # Finishing cleanly removes the checkpoint.
    assert [f for f in os.listdir(tmp_path) if f.endswith(".pt")] == []


def test_changed_config_does_not_resume(tmp_path, loaders):
    """A different learning rate must start over, not inherit stale state."""
    train_loader, val_loader = loaders

    first: list = []
    with pytest.raises(RuntimeError):
        _train(tmp_path, train_loader, ExplodingLoader(val_loader, 3), 4, first)
    assert first == [1, 2]

    # Same call but a different lr → different fingerprint → fresh start.
    second: list = []
    ve._train_and_evaluate(
        lr=5e-4, wd=0.0, run_epochs=4,
        base_clean_model=TinyNet(), model_name="DenseNet121",
        num_classes=N_CLASSES, freeze_strategy="head_only",
        train_loader=train_loader, val_loader=val_loader,
        loss_weight_tensor=None, device=torch.device("cpu"),
        lr_scheduler="cosine", seed=SEED,
        epoch_callback=lambda e, t, l, f: second.append(e),
        checkpoint_dir=str(tmp_path), checkpoint_tag="test",
    )
    assert second == [1, 2, 3, 4], "changed config must retrain from scratch"


def test_no_checkpoint_dir_writes_nothing(tmp_path, loaders):
    """Checkpointing is opt-in; without a directory nothing touches disk."""
    train_loader, val_loader = loaders
    seen: list = []

    ve._train_and_evaluate(
        lr=1e-3, wd=0.0, run_epochs=2,
        base_clean_model=TinyNet(), model_name="DenseNet121",
        num_classes=N_CLASSES, freeze_strategy="head_only",
        train_loader=train_loader, val_loader=val_loader,
        loss_weight_tensor=None, device=torch.device("cpu"),
        lr_scheduler="cosine", seed=SEED,
        epoch_callback=lambda e, t, l, f: seen.append(e),
        checkpoint_dir=None,
    )
    assert seen == [1, 2]
    assert os.listdir(tmp_path) == []


def test_distinct_tags_do_not_collide(tmp_path, loaders):
    """The three long training phases share lr/wd and must not overwrite each other."""
    train_loader, val_loader = loaders

    for tag in ("opt", "final_test", "winner"):
        seen: list = []
        with pytest.raises(RuntimeError):
            ve._train_and_evaluate(
                lr=1e-3, wd=0.0, run_epochs=4,
                base_clean_model=TinyNet(), model_name="DenseNet121",
                num_classes=N_CLASSES, freeze_strategy="head_only",
                train_loader=train_loader,
                val_loader=ExplodingLoader(val_loader, 3),
                loss_weight_tensor=None, device=torch.device("cpu"),
                lr_scheduler="cosine", seed=SEED,
                epoch_callback=lambda e, t, l, f: seen.append(e),
                checkpoint_dir=str(tmp_path), checkpoint_tag=tag,
            )

    ckpts = [f for f in os.listdir(tmp_path) if f.endswith(".pt")]
    assert len(ckpts) == 3, f"expected one checkpoint per phase, got {ckpts}"
