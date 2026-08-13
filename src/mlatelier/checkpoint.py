"""Crash-resumable checkpoints for the deep-learning training loops.

Vision transfer learning and HuggingFace fine-tuning are the only parts of
MLatelier that can run for tens of minutes. Losing that to a closed browser tab,
an OOM, or a power cut is the difference between a tool someone trusts with real
work and one they do not.

Design notes
------------
* **Atomic writes.** A checkpoint is written to a temporary file and then
  ``os.replace``-d into position. A half-written checkpoint is worse than none,
  because resume would load garbage into the optimiser.
* **Fingerprinting.** Every checkpoint records a hash of the run configuration.
  Resume only happens when the fingerprint matches, so changing the learning
  rate or the architecture starts fresh instead of silently continuing a
  different experiment.
* **Streamlit-free.** This module imports no UI code so the engines stay
  testable without a browser.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
import warnings
from typing import Any, Optional

__all__ = [
    "checkpoint_root",
    "fingerprint",
    "checkpoint_path",
    "save_checkpoint",
    "load_checkpoint",
    "clear_checkpoint",
    "list_checkpoints",
]


def checkpoint_root() -> str:
    """Directory holding all checkpoints: ``~/MLatelier/checkpoints``."""
    return os.path.join(os.path.expanduser("~"), "MLatelier", "checkpoints")


def fingerprint(config: dict) -> str:
    """Stable short hash of a run configuration.

    Two runs share a fingerprint only if every configuration value matches, so a
    checkpoint can never be resumed into a differently-configured run.
    """
    blob = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _safe_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in str(name))


def checkpoint_path(tag: str, config: dict, root: Optional[str] = None) -> str:
    """Full path for the checkpoint identified by ``tag`` and ``config``."""
    base = root or checkpoint_root()
    return os.path.join(base, f"{_safe_name(tag)}__{fingerprint(config)}.pt")


def save_checkpoint(path: str, state: dict, config: Optional[dict] = None) -> Optional[str]:
    """Write a checkpoint atomically. Returns the path, or None on failure.

    Never raises: a failed checkpoint must not abort a training run that is
    otherwise progressing fine.
    """
    try:
        import torch
    except ImportError:
        return None

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = dict(state)
        payload["_saved_at"] = time.time()
        if config is not None:
            payload["_fingerprint"] = fingerprint(config)
            payload["_config"] = config

        tmp = f"{path}.{os.getpid()}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)          # atomic on POSIX and Windows
        return path
    except Exception as e:
        warnings.warn(f"Checkpoint save failed for '{path}': {e}", UserWarning)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return None


def load_checkpoint(path: str, config: Optional[dict] = None) -> Optional[dict]:
    """Load a checkpoint, or return None if absent, unreadable, or mismatched.

    ``weights_only=False`` is required because the payload carries plain Python
    state (epoch counters, score history) alongside tensors. These files are
    written by MLatelier into the user's own home directory; do not point this
    at a checkpoint from an untrusted source.
    """
    if not path or not os.path.exists(path):
        return None

    try:
        import torch
    except ImportError:
        return None

    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        warnings.warn(
            f"Checkpoint at '{path}' could not be read ({e}); starting fresh.",
            UserWarning)
        return None

    if config is not None:
        expected = fingerprint(config)
        found = state.get("_fingerprint")
        if found != expected:
            warnings.warn(
                "Checkpoint configuration does not match this run "
                f"(expected {expected}, found {found}); starting fresh.",
                UserWarning)
            return None

    return state


def clear_checkpoint(path: str) -> None:
    """Delete a checkpoint file, ignoring failures."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def list_checkpoints(root: Optional[str] = None) -> list[dict]:
    """Describe every checkpoint on disk, newest first.

    Each entry: ``{"path", "name", "size_mb", "saved_at", "config"}``.
    """
    base = root or checkpoint_root()
    if not os.path.isdir(base):
        return []

    try:
        import torch
    except ImportError:
        return []

    out: list[dict] = []
    for fname in os.listdir(base):
        if not fname.endswith(".pt"):
            continue
        full = os.path.join(base, fname)
        entry: dict[str, Any] = {
            "path": full,
            "name": fname,
            "size_mb": round(os.path.getsize(full) / (1024 * 1024), 1),
            "saved_at": os.path.getmtime(full),
            "config": {},
        }
        try:
            meta = torch.load(full, map_location="cpu", weights_only=False)
            entry["config"] = meta.get("_config", {})
            entry["saved_at"] = meta.get("_saved_at", entry["saved_at"])
            entry["epoch"] = meta.get("epoch")
        except Exception:
            pass
        out.append(entry)

    return sorted(out, key=lambda e: e["saved_at"], reverse=True)


def purge_checkpoints(root: Optional[str] = None) -> int:
    """Delete every checkpoint. Returns how many files were removed."""
    base = root or checkpoint_root()
    if not os.path.isdir(base):
        return 0
    n = 0
    for fname in os.listdir(base):
        if fname.endswith(".pt"):
            try:
                os.remove(os.path.join(base, fname))
                n += 1
            except Exception:
                pass
    return n
