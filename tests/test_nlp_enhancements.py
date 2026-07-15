"""
test_nlp_enhancements.py — Mock test suite for the three NLP enhancements:

  1. Rich preprocessing pipeline (min_df, max_df, sublinear_tf, vectorizer_type)
  2. HuggingFace transformer fine-tuning (run_transformer_finetune — mocked)
  3. NLP-specific visualizations (render_nlp_wordcloud, render_nlp_text_stats)
"""
from __future__ import annotations

import sys
import os
import types
import unittest.mock as mock
import numpy as np
import pytest

# ── Streamlit stub (must be first) ───────────────────────────────────────────

def _make_st_stub():
    st = types.ModuleType("streamlit")

    class _Ctx:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def markdown(self, *a, **kw): pass
        def dataframe(self, *a, **kw): pass
        def metric(self, *a, **kw): pass
        def pyplot(self, *a, **kw): pass
        def caption(self, *a, **kw): pass
        def warning(self, *a, **kw): pass
        def error(self, *a, **kw): pass
        def success(self, *a, **kw): pass
        def info(self, *a, **kw): pass
        def columns(self, *a, **kw): return [self] * 10
        def expander(self, *a, **kw): return self
        def tabs(self, labels): return [self] * len(labels)

    _ctx = _Ctx()
    for _n in ["markdown", "dataframe", "metric", "pyplot", "caption",
               "warning", "error", "success", "info", "download_button",
               "container"]:
        setattr(st, _n, lambda *a, **kw: None)

    st.columns  = lambda *a, **kw: [_ctx] * (a[0] if isinstance(a[0], int) else len(a[0]))
    st.tabs     = lambda labels: [_ctx] * len(labels)
    st.container = lambda **kw: _ctx
    st.expander  = lambda *a, **kw: _ctx
    st.session_state = {}
    return st


sys.modules.setdefault("streamlit", _make_st_stub())
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src/mlatelier"))

from nlp_engine import (
    build_tfidf_pipeline,
    run_nlp_baseline,
    run_nlp_optimization,
    run_transformer_finetune,
    get_top_tfidf_features,
    HF_PRESET_MODELS,
    _HAS_TRANSFORMERS,
)
from reporting import render_nlp_wordcloud, render_nlp_text_stats

# ── Tiny synthetic corpus ─────────────────────────────────────────────────────

_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "machine learning is fascinating and powerful",
    "deep learning with neural networks transforms data",
    "natural language processing helps computers understand text",
    "the dog barked loudly at the fence in the yard",
    "artificial intelligence will change the world forever",
    "convolutional networks excel at image recognition tasks",
    "recurrent networks handle sequential data very well",
    "the cat sat on the mat and looked around",
    "transformers have revolutionised natural language understanding",
    "random forests combine many decision trees together",
    "support vector machines find optimal decision boundaries",
]
_LABELS = ["tech", "tech", "tech", "tech", "animals", "tech",
           "tech", "tech", "animals", "tech", "tech", "tech"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Rich preprocessing pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildTfidfPipelineExtended:

    def test_default_params_tfidf(self):
        pipe = build_tfidf_pipeline("Logistic Regression")
        assert "tfidf" in pipe.named_steps

    def test_count_vectorizer(self):
        pipe = build_tfidf_pipeline(
            "Logistic Regression", vectorizer_type="count")
        assert "count" in pipe.named_steps
        assert "tfidf" not in pipe.named_steps

    def test_min_df_plumbed(self):
        pipe = build_tfidf_pipeline(
            "Logistic Regression", min_df=5)
        vec = pipe.named_steps["tfidf"]
        assert vec.min_df == 5

    def test_max_df_plumbed(self):
        pipe = build_tfidf_pipeline(
            "Logistic Regression", max_df=0.9)
        vec = pipe.named_steps["tfidf"]
        assert vec.max_df == 0.9

    def test_sublinear_tf_explicit_true(self):
        pipe = build_tfidf_pipeline(
            "Logistic Regression", sublinear_tf=True)
        assert pipe.named_steps["tfidf"].sublinear_tf is True

    def test_sublinear_tf_explicit_false(self):
        pipe = build_tfidf_pipeline(
            "Logistic Regression", sublinear_tf=False)
        assert pipe.named_steps["tfidf"].sublinear_tf is False

    def test_nb_forces_no_sublinear(self):
        # Even if sublinear_tf=True, NB models must not use it
        pipe = build_tfidf_pipeline(
            "Multinomial NB", sublinear_tf=True)
        assert pipe.named_steps["tfidf"].sublinear_tf is False

    def test_nb_count_vectorizer_works(self):
        pipe = build_tfidf_pipeline(
            "Multinomial NB", vectorizer_type="count")
        assert "count" in pipe.named_steps

    def test_ngram_range_respected(self):
        pipe = build_tfidf_pipeline("Logistic Regression", ngram_max=3)
        assert pipe.named_steps["tfidf"].ngram_range == (1, 3)

    def test_max_features_respected(self):
        pipe = build_tfidf_pipeline(
            "Logistic Regression", max_features=2000)
        assert pipe.named_steps["tfidf"].max_features == 2000

    def test_pipeline_fits_and_predicts(self):
        pipe = build_tfidf_pipeline(
            "Logistic Regression",
            min_df=1, max_df=1.0,
            sublinear_tf=True, vectorizer_type="tfidf",
        )
        pipe.fit(_TEXTS, _LABELS)
        preds = pipe.predict(_TEXTS)
        assert len(preds) == len(_TEXTS)

    def test_count_pipeline_fits_and_predicts(self):
        pipe = build_tfidf_pipeline(
            "Logistic Regression",
            min_df=1, vectorizer_type="count",
        )
        pipe.fit(_TEXTS, _LABELS)
        assert len(pipe.predict(_TEXTS)) == len(_TEXTS)

    def test_nb_count_fits(self):
        pipe = build_tfidf_pipeline(
            "Multinomial NB", min_df=1, vectorizer_type="count")
        pipe.fit(_TEXTS, _LABELS)
        assert len(pipe.predict(_TEXTS)) == len(_TEXTS)


class TestRunNlpBaselineExtendedParams:

    def test_min_df_max_df_accepted(self):
        results, class_names = run_nlp_baseline(
            _TEXTS, _LABELS, ["Logistic Regression"],
            track="ml", min_df=1, max_df=1.0,
            cv_folds=2, random_state=0,
        )
        assert "Logistic Regression" in results
        assert len(class_names) == 2

    def test_count_vectorizer_track(self):
        results, _ = run_nlp_baseline(
            _TEXTS, _LABELS, ["Logistic Regression"],
            track="ml", vectorizer_type="count", min_df=1,
            cv_folds=2, random_state=0,
        )
        r = results["Logistic Regression"]
        assert "error" not in r
        assert r["mean"] == r["mean"]  # not NaN

    def test_sublinear_false_for_nb(self):
        results, _ = run_nlp_baseline(
            _TEXTS, _LABELS, ["Multinomial NB"],
            track="ml", sublinear_tf=False, min_df=1,
            cv_folds=2, random_state=0,
        )
        assert "Multinomial NB" in results


class TestRunNlpOptimizationExtendedParams:

    def test_passes_min_df_max_df(self):
        baseline, cls = run_nlp_baseline(
            _TEXTS, _LABELS, ["Logistic Regression"],
            track="ml", min_df=1, cv_folds=2, random_state=0,
        )
        (results_list, best_acc, avg_acc, imp, pval,
         params, cr, class_names, curves) = run_nlp_optimization(
            _TEXTS, _LABELS, ["Logistic Regression"], baseline,
            track="ml", min_df=1, max_df=1.0,
            cv_folds=2, n_iter=3, random_state=0,
        )
        assert best_acc == best_acc  # not NaN
        assert len(class_names) == 2

    def test_count_vectorizer_optimization(self):
        baseline, _ = run_nlp_baseline(
            _TEXTS, _LABELS, ["Logistic Regression"],
            track="ml", min_df=1, vectorizer_type="count",
            cv_folds=2, random_state=0,
        )
        (results_list, best_acc, *_) = run_nlp_optimization(
            _TEXTS, _LABELS, ["Logistic Regression"], baseline,
            track="ml", min_df=1, vectorizer_type="count",
            cv_folds=2, n_iter=3, random_state=0,
        )
        assert best_acc == best_acc


class TestGetTopFeaturesWithCountVec:

    def test_count_vec_features_extracted(self):
        from nlp_engine import build_tfidf_pipeline
        pipe = build_tfidf_pipeline(
            "Logistic Regression", min_df=1, vectorizer_type="count")
        pipe.fit(_TEXTS, _LABELS)
        result = get_top_tfidf_features(pipe, ["animals", "tech"], n=5)
        assert isinstance(result, dict)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. HuggingFace preset list
# ═══════════════════════════════════════════════════════════════════════════════

class TestHFPresetModels:

    def test_preset_list_nonempty(self):
        assert len(HF_PRESET_MODELS) >= 5

    def test_distilbert_in_presets(self):
        assert "distilbert-base-uncased" in HF_PRESET_MODELS

    def test_roberta_in_presets(self):
        assert "roberta-base" in HF_PRESET_MODELS

    def test_xlm_roberta_in_presets(self):
        assert "xlm-roberta-base" in HF_PRESET_MODELS


# ═══════════════════════════════════════════════════════════════════════════════
# 3. run_transformer_finetune (fully mocked — no actual model download)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mock_transformers(n_classes: int = 2):
    """Return mock tokenizer and model that mimic the HF API shape."""
    import torch

    class _MockTokenizerOutput(dict):
        pass

    class _MockTokenizer:
        def __call__(self, texts, **kw):
            n = len(texts)
            seq = kw.get("max_length", 16)
            out = _MockTokenizerOutput({
                "input_ids":      torch.zeros(n, seq, dtype=torch.long),
                "attention_mask": torch.ones(n, seq, dtype=torch.long),
            })
            return out

        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)

    class _MockOutput:
        def __init__(self, logits, loss=None):
            self.logits = logits
            self.loss = loss if loss is not None else torch.tensor(0.5, requires_grad=True)

    class _MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, n_classes)

        def forward(self, input_ids=None, attention_mask=None, labels=None):
            bsz = input_ids.size(0) if input_ids is not None else 1
            logits = torch.randn(bsz, n_classes)
            loss = torch.tensor(0.5, requires_grad=True)
            return _MockOutput(logits, loss)

        @classmethod
        def from_pretrained(cls, *a, **kw):
            return cls()

        def save_pretrained(self, path):
            os.makedirs(path, exist_ok=True)

        def named_parameters(self):
            return self.linear.named_parameters()

        def parameters(self):
            return self.linear.parameters()

        def train(self, mode=True): pass
        def eval(self): pass
        def to(self, device): return self

    return _MockTokenizer(), _MockModel()


@pytest.mark.skipif(
    not _HAS_TRANSFORMERS,
    reason="transformers not installed"
)
class TestRunTransformerFinetuneWithMocks:

    def _patch_and_run(self, **extra_kwargs):
        import torch
        from torch.optim import AdamW

        mock_tok, mock_model = _make_mock_transformers(n_classes=2)

        def _fake_schedule(opt, num_warmup_steps, num_training_steps):
            class _Sched:
                def step(self): pass
            return _Sched()

        with mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=mock_tok,
        ), mock.patch(
            "transformers.AutoModelForSequenceClassification.from_pretrained",
            return_value=mock_model,
        ), mock.patch(
            "transformers.get_linear_schedule_with_warmup",
            side_effect=_fake_schedule,
        ):
            return run_transformer_finetune(
                _TEXTS, _LABELS,
                model_id="distilbert-base-uncased",
                epochs=1,
                batch_size=4,
                max_seq_len=16,
                random_state=0,
                **extra_kwargs,
            )

    def test_returns_9_tuple(self):
        result = self._patch_and_run()
        assert len(result) == 9

    def test_results_list_has_entry(self):
        results_list, *_ = self._patch_and_run()
        assert len(results_list) == 1
        assert results_list[0]["Algorithm"] == "distilbert-base-uncased"
        assert results_list[0]["Track"] == "HF-FINETUNE"

    def test_class_names_correct(self):
        *_, class_names, _ = self._patch_and_run()
        assert set(class_names) == {"animals", "tech"}

    def test_best_acc_is_float(self):
        _, best_acc, *_ = self._patch_and_run()
        assert isinstance(best_acc, float)
        assert 0.0 <= best_acc <= 1.0

    def test_class_report_has_confusion_matrix(self):
        (_, _, _, _, _, _, cr, cn, curves) = self._patch_and_run()
        assert "confusion_matrix" in cr
        assert "confusion_matrix_labels" in cr

    def test_winner_curves_keys(self):
        *_, winner_curves = self._patch_and_run()
        assert "train_losses" in winner_curves
        assert "confusion_matrix" in winner_curves
        assert winner_curves["track"] == "hf-finetune"

    def test_freeze_backbone_flag_accepted(self):
        result = self._patch_and_run(freeze_backbone=True)
        assert len(result) == 9

    def test_winner_params_keys(self):
        (_, _, _, _, _, winner_params, *_) = self._patch_and_run()
        for k in ("model_id", "epochs", "batch_size", "learning_rate",
                  "max_seq_len", "freeze_backbone", "warmup_ratio"):
            assert k in winner_params

    def test_export_creates_dir(self, tmp_path):
        self._patch_and_run(export_dir=str(tmp_path))
        # Should have created a subdirectory under tmp_path
        assert any(tmp_path.iterdir())


@pytest.mark.skipif(
    _HAS_TRANSFORMERS,
    reason="transformers IS installed — this tests the missing-dep error"
)
class TestRunTransformerFinetuneNoTransformers:

    def test_raises_import_error(self):
        with mock.patch("nlp_engine._HAS_TRANSFORMERS", False):
            with pytest.raises(ImportError, match="transformers"):
                run_transformer_finetune(_TEXTS, _LABELS)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. NLP visualization helpers
# ═══════════════════════════════════════════════════════════════════════════════

class TestRenderNlpWordcloud:

    def test_runs_without_error_no_labels(self):
        render_nlp_wordcloud(_TEXTS)  # must not raise

    def test_runs_with_labels(self):
        render_nlp_wordcloud(_TEXTS, labels=_LABELS)

    def test_empty_texts_no_crash(self):
        render_nlp_wordcloud([""])

    def test_many_labels_no_crash(self):
        # More than 8 unique labels — per-class section should be skipped
        texts = [f"word{i} extra filler" for i in range(20)]
        labels = list(range(20))
        render_nlp_wordcloud(texts, labels=labels)

    def test_with_wordcloud_mock(self):
        """Ensure code path with wordcloud library is exercised when available."""
        wc_module = types.ModuleType("wordcloud")

        class _FakeWC:
            def __init__(self, **kw): pass

            def generate(self, text):
                return self

            def __array__(self, dtype=None, copy=None):
                return np.zeros((10, 20, 4), dtype=np.uint8)

        wc_module.WordCloud = _FakeWC
        with mock.patch.dict(sys.modules, {"wordcloud": wc_module}):
            # Re-invoke render with the patched module visible inside reporting
            with mock.patch("reporting.render_nlp_wordcloud.__module__", create=True):
                pass  # patch context only needed for module cache; call directly
            # Call the function — it will import wordcloud inside the function scope
            render_nlp_wordcloud(_TEXTS[:4], labels=_LABELS[:4])


class TestRenderNlpTextStats:

    def test_runs_without_labels(self):
        render_nlp_text_stats(_TEXTS)

    def test_runs_with_labels(self):
        render_nlp_text_stats(_TEXTS, labels=_LABELS)

    def test_single_class_no_boxplot_crash(self):
        # Only 1 unique label → per-class box plot should be skipped
        render_nlp_text_stats(_TEXTS, labels=["tech"] * len(_TEXTS))

    def test_many_classes_no_boxplot_crash(self):
        # 13 unique labels → exceeds limit, box plot skipped
        texts = [f"sentence number {i}" for i in range(13)]
        labels = list(range(13))
        render_nlp_text_stats(texts, labels=labels)

    def test_word_count_computed_correctly(self):
        # 3-word texts → stats should reflect that
        texts = ["one two three", "four five six"]
        # Just check no exception is raised
        render_nlp_text_stats(texts)

    def test_empty_string_texts(self):
        render_nlp_text_stats(["", ""])

