"""The context handed to the AI tutor.

The assistant can only teach from what it is given, and it cannot be tested by
asserting on model output. What *is* testable — and what actually determines
answer quality — is whether the diagnostic signals fire on the situations they
were written for, and stay quiet otherwise. A signal that misfires teaches the
user something false, which is worse than saying nothing.
"""

from __future__ import annotations

import sys
import os

import numpy as np
import pandas as pd
import pytest


# conftest.py installs the shared streamlit stub before any test module is
# imported, so nothing is stubbed here — doing so would rebind `reporting.st`
# to a poorer stub and break test_reporting.py depending on collection order.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/mlatelier"))

import reporting as R  # noqa: E402


def _leaderboard(cv_means, test_scores, std="0.0100"):
    return pd.DataFrame({
        "Algorithm": [f"M{i}" for i in range(len(cv_means))],
        "CV Macro F1 (mean±std)": [f"{m:.4f} ± {std}" for m in cv_means],
        "Optimized Test Macro F1": test_scores,
    })


def _joined(state, engine="tabular"):
    return " ".join(R._teaching_signals(state, engine))


# ── leaderboard ordering vs noise ─────────────────────────────────────────────

def test_flags_leaderboard_ordering_as_noise():
    """Spread below fold noise means the ranking is meaningless — the single
    most valuable thing to tell a user staring at a leaderboard."""
    state = {
        "results_df": _leaderboard([0.71, 0.705, 0.66], [0.712, 0.7095, 0.661]),
        "fold_scores": {"M0": [0.62, 0.78, 0.70, 0.66, 0.79]},
    }
    text = _joined(state)
    assert "not statistically meaningful" in text
    assert "0.0510" in text          # the spread, quoted exactly


def test_accepts_genuinely_separated_models():
    state = {
        "results_df": _leaderboard([0.93, 0.72], [0.93, 0.72]),
        "fold_scores": {"M0": [0.93, 0.94, 0.92, 0.93, 0.93]},
    }
    text = _joined(state)
    assert "genuinely separated" in text
    assert "not statistically meaningful" not in text


def test_spread_alone_when_no_fold_scores():
    state = {"results_df": _leaderboard([0.9, 0.8], [0.9, 0.8])}
    text = _joined(state)
    assert "spread" in text.lower()


# ── generalisation gap ────────────────────────────────────────────────────────

def test_flags_cv_above_test_as_overfitting():
    state = {"results_df": _leaderboard([0.93, 0.91], [0.80, 0.72])}
    text = _joined(state)
    assert "HIGHER than held-out test" in text
    assert "fitting the validation folds" in text


def test_flags_test_above_cv_as_lucky_split():
    state = {"results_df": _leaderboard([0.70, 0.68], [0.88, 0.86])}
    text = _joined(state)
    assert "luck of the split" in text


def test_no_gap_flag_when_cv_and_test_agree():
    state = {"results_df": _leaderboard([0.80, 0.78], [0.79, 0.77])}
    text = _joined(state)
    assert "HIGHER than" not in text


# ── class imbalance and thin support ──────────────────────────────────────────

def test_flags_imbalance_and_thin_support():
    state = {"class_report": {
        "0": {"precision": .96, "recall": .99, "f1-score": .97, "support": 74},
        "1": {"precision": .50, "recall": .33, "f1-score": .40, "support": 6},
    }}
    text = _joined(state)
    assert "12.3:1" in text
    assert "accuracy is misleading" in text
    assert "Thin test support" in text


def test_balanced_classes_produce_no_imbalance_flag():
    state = {"class_report": {
        "0": {"precision": .9, "recall": .9, "f1-score": .9, "support": 100},
        "1": {"precision": .9, "recall": .9, "f1-score": .9, "support": 95},
    }}
    text = _joined(state)
    assert "imbalance" not in text.lower()
    assert "Thin test support" not in text


def test_aggregate_rows_are_not_treated_as_classes():
    """'macro avg' and friends carry a support value but are not classes."""
    state = {"class_report": {
        "0": {"precision": .9, "recall": .9, "f1-score": .9, "support": 100},
        "1": {"precision": .9, "recall": .9, "f1-score": .9, "support": 90},
        "macro avg": {"precision": .9, "recall": .9, "f1-score": .9, "support": 190},
        "accuracy": 0.9,
    }}
    text = _joined(state)
    assert "imbalance" not in text.lower()


# ── optimiser convergence ─────────────────────────────────────────────────────

def test_flags_plateaued_optimiser():
    state = {"winning_curves": {
        "bo_history": [0.68, 0.70, 0.71, 0.712, 0.712, 0.712, 0.712, 0.712]}}
    assert "plateaued" in _joined(state)


def test_flags_still_improving_optimiser():
    state = {"winning_curves": {
        "bo_history": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.86]}}
    assert "still improving" in _joined(state)


# ── configuration critique ────────────────────────────────────────────────────

def test_flags_smote_on_regression():
    state = {"task_type": "regression", "_handle_imbalance": True}
    assert "SMOTE is enabled on a regression task" in _joined(state)


def test_no_smote_flag_on_classification():
    state = {"task_type": "classification", "_handle_imbalance": True}
    assert "SMOTE is enabled on a regression" not in _joined(state)


# ── dataset health ────────────────────────────────────────────────────────────

def _health(df, features):
    return " ".join(R._data_health_lines({"_df": df, "_features_x": features}))


def test_health_reports_missing_constant_and_duplicate_features():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "a": rng.normal(size=n),
        "b": rng.normal(size=n),
        "const": 1,
        "cat": rng.choice(["x", "y"], n),
    })
    df["a_copy"] = df["a"] * 1.0001
    df.loc[rng.choice(n, 40, replace=False), "b"] = np.nan

    text = _health(df, ["a", "b", "const", "cat", "a_copy"])
    assert "Features >5% missing" in text and "b (20%)" in text
    assert "Constant features" in text and "const" in text
    assert "Near-duplicate pairs" in text and "a~a_copy" in text
    assert "4 numeric, 1 non-numeric" in text


def test_health_clean_dataset_reports_no_problems():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
    text = _health(df, ["a", "b"])
    assert "Missing values       : none" in text
    assert "Constant features" not in text
    assert "Near-duplicate pairs" not in text


def test_health_tolerates_missing_dataframe():
    assert R._data_health_lines({"_df": None, "_features_x": ["a"]}) == []
    assert R._data_health_lines({}) == []


# ── pipeline configuration ────────────────────────────────────────────────────

def test_pipeline_config_surfaces_user_choices():
    state = {
        "_scaler": "robust", "_handle_imbalance": True,
        "_X_train": list(range(320)), "_X_test": list(range(80)),
        "winning_curves": {"reproducibility_metadata": {
            "cv_folds": 5, "n_iter": 30, "random_state": 42}},
    }
    text = " ".join(R._pipeline_config_lines(state))
    assert "robust" in text
    assert "on" in text
    assert "320" in text and "80" in text and "20%" in text
    assert "cv_folds" in text and "30" in text


# ── robustness: the assistant must never crash the results page ───────────────

@pytest.mark.parametrize("state", [
    {},
    {"results_df": pd.DataFrame()},
    {"results_df": None, "fold_scores": {}, "class_report": None},
    {"class_report": {"weird": "not a dict"}},
    {"winning_curves": {"bo_history": []}},
    {"fold_scores": {"M": [0.5]}},                 # single fold, no std
    {"results_df": _leaderboard([0.5], [0.5])},    # single model, no spread
])
def test_signals_never_raise_on_degenerate_state(state):
    assert isinstance(R._teaching_signals(state, "tabular"), list)


def test_full_context_builds_from_empty_state():
    out = R._build_experiment_context({}, "tabular")
    assert "MLatelier Experiment Context" in out


def test_full_context_includes_all_sections():
    rng = np.random.default_rng(2)
    n = 120
    df = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    df["target"] = rng.choice([0, 1], n)
    state = {
        "task_type": "classification",
        "results_df": _leaderboard([0.71, 0.66], [0.712, 0.661]),
        "fold_scores": {"M0": [0.62, 0.78, 0.70]},
        "_df": df, "_features_x": ["a", "b"], "_target_y": "target",
        "_scaler": "standard", "_handle_imbalance": False,
    }
    out = R._build_experiment_context(state, "tabular")
    assert "Pipeline Configuration" in out
    assert "Dataset Health" in out
    assert "Diagnostic Signals" in out


def test_teacher_prompt_demands_reasoning_not_just_actions():
    p = R._MLATELIER_TEACHER_PROMPT
    assert "WHY IT MATTERS" in p
    assert "TEACHING, NOT OPTIMISING" in p
    assert "Diagnostic Signals" in p
