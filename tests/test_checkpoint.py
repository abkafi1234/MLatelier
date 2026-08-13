"""Checkpoint save/load/resume semantics.

The value of a checkpoint is entirely in the failure case, so these tests focus
on what happens when things go wrong: a mismatched configuration, a truncated
file, a missing directory. A checkpoint that silently resumes into the wrong
experiment is worse than no checkpoint at all.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/mlatelier"))

import checkpoint as ckpt  # noqa: E402

torch = pytest.importorskip("torch", reason="torch not installed")

CFG_A = {"kind": "vision", "model_name": "ResNet18", "lr": 1e-3, "epochs": 10}
CFG_B = {"kind": "vision", "model_name": "ResNet18", "lr": 5e-4, "epochs": 10}


# ── fingerprinting ────────────────────────────────────────────────────────────

def test_fingerprint_is_stable_and_order_independent():
    a = ckpt.fingerprint({"x": 1, "y": 2})
    b = ckpt.fingerprint({"y": 2, "x": 1})
    assert a == b


def test_fingerprint_changes_with_any_value():
    assert ckpt.fingerprint(CFG_A) != ckpt.fingerprint(CFG_B)


def test_checkpoint_path_is_deterministic(tmp_path):
    p1 = ckpt.checkpoint_path("vision_ResNet18", CFG_A, root=str(tmp_path))
    p2 = ckpt.checkpoint_path("vision_ResNet18", CFG_A, root=str(tmp_path))
    assert p1 == p2
    assert p1.endswith(".pt")


def test_checkpoint_path_sanitises_tag(tmp_path):
    p = ckpt.checkpoint_path("vision/Res Net:18", CFG_A, root=str(tmp_path))
    base = os.path.basename(p)
    for bad in "/\\: ":
        assert bad not in base


# ── round trip ────────────────────────────────────────────────────────────────

def test_save_then_load_round_trip(tmp_path):
    path = str(tmp_path / "run.pt")
    state = {"epoch": 4, "tensor": torch.ones(3), "losses": [0.5, 0.4]}

    assert ckpt.save_checkpoint(path, state, config=CFG_A) == path
    loaded = ckpt.load_checkpoint(path, CFG_A)

    assert loaded is not None
    assert loaded["epoch"] == 4
    assert loaded["losses"] == [0.5, 0.4]
    assert torch.equal(loaded["tensor"], torch.ones(3))


def test_save_creates_missing_directories(tmp_path):
    path = str(tmp_path / "deep" / "nested" / "run.pt")
    assert ckpt.save_checkpoint(path, {"epoch": 1}, config=CFG_A) == path
    assert os.path.exists(path)


def test_save_leaves_no_temp_file(tmp_path):
    path = str(tmp_path / "run.pt")
    ckpt.save_checkpoint(path, {"epoch": 1}, config=CFG_A)
    leftovers = [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]
    assert leftovers == []


# ── refusing to resume the wrong thing ────────────────────────────────────────

def test_mismatched_config_refuses_to_resume(tmp_path):
    """The whole point: a changed learning rate must not silently continue."""
    path = str(tmp_path / "run.pt")
    ckpt.save_checkpoint(path, {"epoch": 7}, config=CFG_A)

    with pytest.warns(UserWarning, match="does not match"):
        assert ckpt.load_checkpoint(path, CFG_B) is None


def test_missing_file_returns_none_silently(tmp_path):
    assert ckpt.load_checkpoint(str(tmp_path / "nope.pt"), CFG_A) is None


def test_corrupt_file_returns_none_with_warning(tmp_path):
    path = str(tmp_path / "corrupt.pt")
    with open(path, "wb") as fh:
        fh.write(b"this is not a torch checkpoint")

    with pytest.warns(UserWarning, match="could not be read"):
        assert ckpt.load_checkpoint(path, CFG_A) is None


def test_load_without_config_skips_validation(tmp_path):
    path = str(tmp_path / "run.pt")
    ckpt.save_checkpoint(path, {"epoch": 2}, config=CFG_A)
    assert ckpt.load_checkpoint(path)["epoch"] == 2


# ── lifecycle ─────────────────────────────────────────────────────────────────

def test_clear_removes_file_and_tolerates_absence(tmp_path):
    path = str(tmp_path / "run.pt")
    ckpt.save_checkpoint(path, {"epoch": 1}, config=CFG_A)
    ckpt.clear_checkpoint(path)
    assert not os.path.exists(path)
    ckpt.clear_checkpoint(path)          # must not raise


def test_list_checkpoints_reports_metadata(tmp_path):
    ckpt.save_checkpoint(str(tmp_path / "a.pt"), {"epoch": 3}, config=CFG_A)
    ckpt.save_checkpoint(str(tmp_path / "b.pt"), {"epoch": 9}, config=CFG_B)

    entries = ckpt.list_checkpoints(root=str(tmp_path))
    assert len(entries) == 2
    by_epoch = {e["epoch"]: e for e in entries}
    assert set(by_epoch) == {3, 9}
    assert by_epoch[3]["config"]["lr"] == CFG_A["lr"]
    assert all(e["size_mb"] >= 0 for e in entries)


def test_list_checkpoints_on_missing_dir(tmp_path):
    assert ckpt.list_checkpoints(root=str(tmp_path / "absent")) == []


def test_purge_removes_all(tmp_path):
    ckpt.save_checkpoint(str(tmp_path / "a.pt"), {"epoch": 1}, config=CFG_A)
    ckpt.save_checkpoint(str(tmp_path / "b.pt"), {"epoch": 2}, config=CFG_B)
    assert ckpt.purge_checkpoints(root=str(tmp_path)) == 2
    assert ckpt.list_checkpoints(root=str(tmp_path)) == []


# ── optimiser state actually survives ─────────────────────────────────────────

def test_optimizer_and_scheduler_state_survive(tmp_path):
    """Resuming without optimiser state would reset Adam's moment estimates."""
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)

    for _ in range(3):
        opt.zero_grad()
        model(torch.randn(8, 4)).sum().backward()
        opt.step()
        sched.step()

    path = str(tmp_path / "run.pt")
    ckpt.save_checkpoint(path, {
        "epoch": 3,
        "model_state": model.state_dict(),
        "optimizer_state": opt.state_dict(),
        "scheduler_state": sched.state_dict(),
    }, config=CFG_A)

    model2 = torch.nn.Linear(4, 2)
    opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
    sched2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=10)

    loaded = ckpt.load_checkpoint(path, CFG_A)
    model2.load_state_dict(loaded["model_state"])
    opt2.load_state_dict(loaded["optimizer_state"])
    sched2.load_state_dict(loaded["scheduler_state"])

    assert sched2.last_epoch == sched.last_epoch
    assert opt2.param_groups[0]["lr"] == pytest.approx(opt.param_groups[0]["lr"])
    for p1, p2 in zip(model.parameters(), model2.parameters()):
        assert torch.equal(p1, p2)
