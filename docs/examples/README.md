# Examples

Runnable scripts that use MLatelier's engines as a library — no Streamlit, no
browser. Each is self-contained and uses a scikit-learn built-in or synthetic
dataset, so nothing needs downloading.

```bash
pip install -e ".[dev]"
python docs/examples/tabular_end_to_end.py
```

| Script | Demonstrates |
|---|---|
| [tabular_end_to_end.py](tabular_end_to_end.py) | Baseline → Bayesian optimisation → SHAP → export → reload → predict |
| [multilabel_classification.py](multilabel_classification.py) | Multi-label targets, per-label metrics, averaged importances |
| [ablation_study.py](ablation_study.py) | Which pipeline components actually earn their keep |
| [nlp_track_comparison.py](nlp_track_comparison.py) | TF-IDF vs Sentence Transformer tracks on the same corpus |

## Reading the results

Engine functions return plain dictionaries and lists — no Streamlit objects — so
you can print them, assert on them in tests, or feed them into your own plots.

`run_tabular_optimization` returns a 12-tuple:

```python
(pipeline_results, winning_test_score, avg_base_score, improvement,
 p_val_str, winning_params, winning_report, class_names, task_type,
 winning_curves, actual_scaler_key, exported_model_path) = run_tabular_optimization(...)
```

## A note on runtimes

These scripts use small budgets (`n_iter=5`, `cv_folds=3`) so they finish in
under a minute. Real experiments want `n_iter=30` and `cv_folds=5`, which costs
roughly an order of magnitude more time.
