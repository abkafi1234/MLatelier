# API Reference

Every engine is importable and usable without Streamlit. Functions that accept
`progress_bar`, `status_text`, or `*_callback` arguments use them only for live
UI updates — omit them entirely when scripting.

```python
from mlatelier import tabular_engine, vision_engine, nlp_engine
from mlatelier import inference_engine, file_utils, tracker
```

---

## `mlatelier.tabular_engine`

### Model catalogues

```python
get_classification_models(rs: int = 42) -> dict
get_regression_models(rs: int = 42) -> dict
get_search_spaces() -> dict
```

Return `{display_name: estimator}` and `{display_name: skopt_search_space}`
respectively. Gradient-boosting backends that are not installed are omitted from
the catalogue rather than raising.

### Training

```python
run_tabular_baseline(
    df, features_x, target_y, selected_models,
    handle_imbalance=True, random_state=42, scaler="auto",
    test_size=0.2, cv_folds=None,
    progress_bar=None, status_text=None, model_callback=None,
) -> tuple[dict, str]
```

Cross-validates every selected model with default hyperparameters.
Returns `({model: {"mean", "std", "fold_scores", "time_s"}}, task_type)`.
`task_type` is one of `"regression"`, `"classification"`, `"multilabel"` — binary
and multiclass are both reported as `"classification"` and scored with macro F1.

A model that fails to fit does not abort the run: it is recorded with
`mean = nan` and a `UserWarning` is emitted.

```python
run_tabular_optimization(
    df, features_x, target_y, selected_models, baseline_results,
    handle_imbalance=True, random_state=42, scaler="auto",
    test_size=0.2, cv_folds=None, n_iter=10,
    export_model=True, export_dir=None,
    progress_bar=None, status_text=None, model_callback=None,
) -> tuple
```

Runs `BayesSearchCV` per model, refits each on the full training set, and scores
on the held-out test set. **The winner is chosen by inner CV score**
(`BayesSearchCV.best_score_`, or the baseline CV mean where no search space is
defined). Test-set performance is computed and reported for every model, both
before and after optimisation, but plays no part in selection. The effective
iteration budget is `n_iter × min(len(search_space), 3)`.

When `export_model=True`, each fitted pipeline is written as `.joblib` with a
JSON metadata sidecar, plus a best-effort `.onnx` file for every convertible
model. The returned `winning_curves` dict carries `all_exported_paths`
(`{model: .joblib path}`) and `all_onnx_paths` (`{model: .onnx path}`, holding
only the models that converted).

### Explainability

```python
compute_shap_values(pipeline, X_test, model_name, feature_names) -> dict
compute_shap_waterfall(...) -> dict
compute_lime_explanation(...) -> dict
compute_pdp(...) -> dict
compute_calibration_data(...) -> dict
compute_learning_curve(...) -> dict
compute_counterfactuals(...) -> dict
```

`compute_shap_values` returns mean absolute SHAP per feature for the top 20
features, subsampled to at most 300 test rows for tractability. All of these
return `{}` (or a dict with an `"error"` key) rather than raising when the
required optional library is missing or the model family is unsupported.

> Sample-level XAI is unavailable for multi-label models —
> `MultiOutputClassifier` breaks the single-output assumption these explainers
> rely on. Multi-label runs report averaged per-label feature importances
> instead.

### Ablation

```python
run_tabular_ablation(
    df, features_x, target_y, model_name, ablation_configs,
    random_state=42, progress_bar=None, status_text=None,
) -> list[dict]
```

Each entry of `ablation_configs` is a dict with `"label"` plus any of
`"handle_imbalance"`, `"scaler"`, `"test_size"`, `"cv_folds"`.
Returns rows keyed `"Config"`, `"CV Score"`, `"Std (±)"`, `"Time (s)"`,
`"Delta (%)"` — note the spaces in those key names. `"Delta (%)"` is measured
against the first configuration and is a preformatted string.

```python
run_feature_ablation(
    df, features_x, target_y, model_name,
    handle_imbalance=True, random_state=42, scaler="auto", cv_folds=None,
) -> list[dict]
```

Drops one feature at a time. Returns `Feature Dropped, Baseline CV, CV Without,
Delta (%), Time (s)`. Raises `ValueError` with fewer than two features.

### Export and utilities

```python
export_tabular_model(pipeline, export_dir, model_name, metadata=None) -> str
export_tabular_model_onnx(pipeline, export_dir, model_name, n_features,
                          X_sample=None) -> str | None
auto_select_scaler(df, features_x) -> str
compute_roc_data(y_test, pipeline, ...) -> dict
calculate_mcnemar_p_value(y_true, y_base_pred, y_opt_pred) -> float
```

`export_tabular_model_onnx` returns `None` (with a warning) when `skl2onnx` is
absent or the model family has no converter. The oversampling step is excluded
from the exported graph — it is training-only and has no role in inference.

Converters for XGBoost and LightGBM are registered automatically on first use
(they come from `onnxmltools`, a required dependency), so pipelines ending in
those estimators convert too. Classifier graphs emit a plain
probability **tensor** rather than skl2onnx's default ZipMap, because most
non-Python runtimes handle tensors and not maps.

Multi-label pipelines wrapped in `MultiOutputClassifier` convert as well.

**Not convertible:** CatBoost — no skl2onnx converter exists, and its native
exporter cannot serialise the surrounding preprocessing. It falls back to
`.joblib` cleanly.

Pass `X_sample` (the training frame). The pipeline's preprocessor is a
`ColumnTransformer` that selects **by column name**, so the graph is built with
one named input per feature; a single anonymous tensor cannot be matched back to
column names and conversion fails. Consumers feed a dict keyed by column name:

```python
feed = {i.name: X[i.name].to_numpy().astype(np.float32).reshape(-1, 1)
        for i in sess.get_inputs()}
```

---

## `mlatelier.nlp_engine`

```python
get_nlp_ml_models() -> dict[str, tuple]
get_nlp_dl_classifiers() -> dict[str, object]
get_nlp_device() -> dict
```

`get_nlp_ml_models` returns six classifiers, plus XGBoost when it is installed.

```python
build_tfidf_pipeline(
    model_name, *, use_stopwords=True, use_stemming=False,
    ngram_max=2, max_features=5000, min_df=2, max_df=1.0,
    sublinear_tf=None, vectorizer_type="tfidf",
) -> sklearn.pipeline.Pipeline
```

Returns a pipeline with named steps `"tfidf"` and `"clf"`. Naive Bayes models
default to non-sublinear term frequency.

```python
run_nlp_baseline(
    texts, labels, model_names, *, track="ml",
    use_stopwords=True, use_stemming=False, ngram_max=2,
    max_features=5000, min_df=2, max_df=1.0, sublinear_tf=None,
    vectorizer_type="tfidf", st_model_name="all-MiniLM-L6-v2",
    device="cpu", cv_folds=5, random_state=42,
) -> tuple[dict, list[str]]

run_nlp_optimization(...) -> tuple
```

`track="ml"` uses TF-IDF; `track="dl"` encodes with a Sentence Transformer
first. Labels are label-encoded internally; `class_names` comes back as the
second element.

```python
run_transformer_finetune(
    texts, labels, model_id="distilbert-base-uncased", *,
    epochs=3, batch_size=16, learning_rate=2e-5, max_seq_len=128,
    warmup_ratio=0.1, freeze_backbone=False, random_state=42,
    export_dir=None,
) -> tuple
```

Fine-tunes any HuggingFace sequence-classification checkpoint. Returns the same
9-tuple as `run_nlp_optimization` for drop-in compatibility:
`(results_list, best_acc, avg_acc, improvement_pct, p_val_str, winner_params,
class_report, class_names, winner_curves)`.

```python
encode_with_transformer(texts, model_name="all-MiniLM-L6-v2", *,
                        device="cpu", batch_size=32) -> np.ndarray
compute_nlp_lime(pipeline, texts, class_names, *, sample_idx=0,
                 num_features=15, num_samples=300) -> dict
get_top_tfidf_features(pipeline, class_names, n=20) -> dict
```

LIME and top-coefficient views apply to the ML track only — the embedding and
fine-tuning tracks have no term-level surrogate.

---

## `mlatelier.vision_engine`

```python
get_device() -> torch.device
get_device_info() -> dict
set_master_seed(seed=42)
get_vision_model(model_name) -> nn.Module
replace_classification_head(model, model_name, num_classes) -> nn.Module
get_vision_dataset_summary(data_dir, n_samples_per_class=4) -> dict
```

Registry keys are exact: `ResNet18`, `ResNet50`, `VGG16`, `EfficientNet_B0`,
`EfficientNet_V2_S`, `MobileNet_v3`, `DenseNet121`, `ConvNeXt_T`, `ConvNeXt_S`,
`ViT_B_16`, `Swin_T`, `Swin_S`. An unknown name raises `ValueError`. Grad-CAM is
supported for the nine convolutional architectures only.

```python
run_vision_baseline(
    data_dir, selected_models, batch_size=32, use_augmentation=True,
    freeze_strategy="head_only", handle_imbalance=True, baseline_epochs=3,
    progress_bar=None, status_text=None, random_seed=42,
    epoch_callback=None, model_callback=None,
) -> tuple

run_vision_optimization(..., checkpoint_dir=None) -> tuple

run_vision_optimization(
    data_dir, selected_models, baseline_results, epochs=10, batch_size=32,
    use_augmentation=True, freeze_strategy="head_only", handle_imbalance=True,
    n_bo_calls=20, early_stopping_patience=5, lr_scheduler="cosine",
    export_model=True, export_dir=None,
    progress_bar=None, status_text=None, random_seed=42,
    epoch_callback=None, model_callback=None,
) -> tuple
```

`data_dir` is a directory of class-labelled subdirectories, split 60/20/20
train/validation/test. The baseline uses fixed hyperparameters (`lr=1e-3`,
`wd=0`); optimisation runs Bayesian search over learning rate and weight decay.

```python
compute_gradcam(...) -> dict
export_vision_model(...) -> str
```

---

## `mlatelier.inference_engine`

```python
load_tabular_model(model_path) -> tuple[pipeline, dict]
predict_tabular(pipeline, df, feature_names, class_names=None) -> pd.DataFrame
load_vision_model(...) -> tuple
predict_vision_batch(...) -> pd.DataFrame
predict_vision_folder(...) -> pd.DataFrame
```

`load_tabular_model` also reads the `_metadata.json` sidecar next to the
`.joblib`, returning `{}` if absent.

`predict_tabular` returns the input frame plus a `Prediction` column, and, when
the estimator supports it, `Confidence` and per-class `P(<class>)` columns —
falling back to `Decision Score` for margin-based estimators. **Feature columns
missing from the new data are added and filled with 0, and a `UserWarning` names
them.** Treat that warning as a schema mismatch worth investigating.

---

## `mlatelier.checkpoint`

Crash-resumable checkpoints for the two long-running training loops. Streamlit-free.

```python
checkpoint_root() -> str                     # ~/MLatelier/checkpoints
fingerprint(config: dict) -> str             # stable 16-char hash
checkpoint_path(tag, config, root=None) -> str
save_checkpoint(path, state, config=None) -> str | None
load_checkpoint(path, config=None) -> dict | None
clear_checkpoint(path) -> None
list_checkpoints(root=None) -> list[dict]
purge_checkpoints(root=None) -> int
```

`save_checkpoint` writes to a temporary file and `os.replace`s it into position,
so an interrupted write cannot leave a half-written checkpoint that would load
garbage into the optimiser. It never raises — a failed checkpoint must not abort
a run that is otherwise progressing.

`load_checkpoint` returns `None` (with a `UserWarning`) when the file is absent,
unreadable, or its `config` fingerprint does not match, so a resume can never
continue a differently-configured experiment.

Pass `checkpoint_dir` to `run_vision_optimization` or `run_transformer_finetune`
to enable it. State is written after every epoch and removed on normal
completion.

> `load_checkpoint` uses `weights_only=False` because the payload carries plain
> Python state alongside tensors. These files are written by MLatelier into the
> user's home directory; do not point it at a checkpoint from an untrusted
> source.

---

## `mlatelier.file_utils`

```python
file_ext(src) -> str
read_tabular(src, sheet_name=0) -> pd.DataFrame
sheet_selector(src, selectbox_fn, key) -> str | int
```

`src` may be a path string, a Streamlit `UploadedFile`, or a `NamedBytesIO`.
`sheet_selector` takes the select-box callable as a parameter rather than
importing Streamlit, which is what keeps this module testable in isolation.

---

## `mlatelier.tracker`

```python
save_minimal_experiment(
    task_type, models_raced, results_df, report_dict, best_score,
    winning_curves=None, scaler_used="N/A", best_params=None,
    label_classes=None, experiment_metadata=None,
)
```

Writes the run to `~/MLatelier/experiments/EXP_<YYYYMMDD_HHMMSS>/`, generating
curve images and a self-contained HTML report in a single pass.

Note that serialised models do **not** go here — `export_tabular_model` writes
them to whatever `export_dir` it is given, which the app sets to a flat
`~/MLatelier/models/` with model-keyed filenames. Successive runs therefore
overwrite each other's model files; copy them out if you need to keep them.

---

## `mlatelier.reporting`

Roughly fifty `render_*` functions that draw results into a Streamlit page
(`render_confusion_matrix`, `render_shap_plot`, `render_pdp`,
`render_multilabel_report`, `render_ai_chat`, …). These are the one part of the
codebase that *does* require an active Streamlit runtime; when using MLatelier
as a library, read the result dictionaries directly and plot them however you
like.
