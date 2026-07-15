from __future__ import annotations

import gc
import io
import os
import shutil
import tempfile
import zipfile
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st

from utils import setup_page
from reporting import (
    render_ablation_chart,
    render_ai_chat,
    render_all_models_download,
    render_calibration_curve,
    render_classification_report,
    render_confidence_histogram,
    render_confusion_matrix,
    render_convergence_plot,
    render_correlation_heatmap,
    render_cv_boxplot,
    render_data_preview,
    render_data_quality_report,
    render_export_buttons,
    render_feature_ablation_chart,
    render_feature_boxplots,
    render_feature_distributions,
    render_feature_importance,
    render_gradcam,
    render_latex_export,
    render_learning_curve,
    render_lime_explanation,
    render_missing_heatmap,
    render_multilabel_confusion_matrix,
    render_multilabel_report,
    render_pdp,
    render_per_class_accuracy,
    render_prediction_results,
    render_regression_report,
    render_regression_residuals,
    render_roc_curves,
    render_shap_plot,
    render_shap_waterfall,
    render_statistical_summary,
    render_target_distribution,
    render_timing_chart,
    render_top_metrics,
    render_training_curve,
    render_vision_class_distribution,
    render_vision_sample_grid,
    render_nlp_lime_explanation,
    render_nlp_top_features,
    render_nlp_wordcloud,
    render_nlp_text_stats,
)
from tabular_engine import (
    _is_regression,
    compute_calibration_data,
    compute_learning_curve,
    compute_lime_explanation,
    compute_pdp,
    compute_shap_waterfall,
    get_classification_models,
    get_regression_models,
    run_feature_ablation,
    run_tabular_ablation,
    run_tabular_baseline,
    run_tabular_optimization,
)
from vision_engine import (
    get_device_info,
    get_vision_dataset_summary,
    run_vision_baseline,
    run_vision_optimization,
)
import inference_engine as infer
from nlp_engine import (
    get_nlp_device,
    get_nlp_ml_models,
    get_nlp_dl_classifiers,
    NLP_ST_MODELS,
    HF_PRESET_MODELS,
    run_nlp_baseline,
    run_nlp_optimization,
    run_transformer_finetune,
    compute_nlp_lime,
    get_top_tfidf_features,
    _HAS_ST,
    _HAS_TRANSFORMERS,
)

# ── Page setup ────────────────────────────────────────────────────────────────
setup_page()

# ── Per-engine session state ──────────────────────────────────────────────────
_ENG_DEFAULTS: dict = {
    "results_df": None,
    "best_acc": "—",
    "avg_acc": "—",
    "imp": "—",
    "pval": "—",
    "best_params": None,
    "class_report": None,
    "label_classes": None,
    "winning_curves": None,
    "save_path": None,
    "html_report_str": "",
    "task_type": None,
    "exported_model_bytes": None,
    "exported_model_filename": None,
    # timing / fold tracking
    "baseline_results": None,
    "calibration_data": None,
    "timing_data": None,
    "fold_scores": None,
    # interpretability
    "pdp_data": None,
    "shap_waterfall": None,
    "lime_data": None,
    # learning curve
    "learning_curve": None,
    # feature ablation
    "feature_ablation_results": None,
    # raw data ref for post-hoc compute
    "_df": None,
    "_features_x": None,
    "_target_y": None,
    "_handle_imbalance": None,
    "_scaler": None,
    "_X_test": None,
    "_X_train": None,
}
_NLP_DEFAULTS: dict = {
    "results_df": None,
    "best_acc": "—",
    "avg_acc": "—",
    "imp": "—",
    "pval": "—",
    "best_params": None,
    "class_report": None,
    "winning_curves": None,
    "task_type": "classification",
    "save_path": None,
    "timing_data": None,
    "fold_scores": None,
    "lime_data": None,
    "top_features": None,
    "_texts": None,
    "_labels": None,
}

for _eng in ("tabular", "vision"):
    if _eng not in st.session_state:
        st.session_state[_eng] = dict(_ENG_DEFAULTS)
    else:
        for _k, _v in _ENG_DEFAULTS.items():
            if _k not in st.session_state[_eng]:
                st.session_state[_eng][_k] = _v

if "nlp" not in st.session_state:
    st.session_state.nlp = dict(_NLP_DEFAULTS)
else:
    for _k, _v in _NLP_DEFAULTS.items():
        if _k not in st.session_state.nlp:
            st.session_state.nlp[_k] = _v

if "last_engine" not in st.session_state:
    st.session_state.last_engine = None
if "ablation_results" not in st.session_state:
    st.session_state.ablation_results = None
if "feat_ablation_results" not in st.session_state:
    st.session_state.feat_ablation_results = None
if "learning_curve_data" not in st.session_state:
    st.session_state.learning_curve_data = None
if "inference_results" not in st.session_state:
    st.session_state.inference_results = None


# ── Helpers ───────────────────────────────────────────────────────────────────
def _fmt_pct(v):
    return f"{v * 100:.2f}%" if isinstance(v, float) and pd.notnull(v) else "—"


def _fmt_f4(v):
    return f"{v:.4f}" if isinstance(v, float) and pd.notnull(v) else "—"


def _model_groups(is_reg: bool) -> dict[str, list[str]]:
    if is_reg:
        return {
            "Boosting & Ensemble": [
                "Random Forest Regressor", "XGBoost Regressor",
                "Gradient Boosting Regressor", "Extra Trees Regressor",
                "Hist Gradient Boosting Regressor",
                "LightGBM Regressor", "CatBoost Regressor", "AdaBoost Regressor",
            ],
            "Linear & SVM": [
                "Linear Regression", "Ridge Regressor", "SGD Regressor",
                "Passive Aggressive Regressor", "Linear SVR",
                "Support Vector Regressor",
            ],
            "Tree & Distance": ["Decision Tree Regressor", "KNN Regressor"],
            "Neural": ["MLP Regressor"],
        }
    return {
        "Boosting & Ensemble": [
            "Random Forest", "XGBoost", "Gradient Boosting", "Extra Trees",
            "Hist Gradient Boosting", "LightGBM", "CatBoost", "AdaBoost",
        ],
        "Linear & SVM": [
            "Logistic Regression", "Ridge Classifier", "SGD Classifier",
            "Passive Aggressive", "Linear SVC", "Support Vector Machine",
        ],
        "Tree & Distance": ["Decision Tree", "KNN"],
        "Neural & Probabilistic": [
            "MLP Classifier", "Gaussian NB", "Multinomial NB",
            "Bernoulli NB", "LDA", "QDA",
        ],
    }


def _available_models(is_reg: bool) -> set[str]:
    registry = get_regression_models() if is_reg else get_classification_models()
    return set(registry.keys())


def _render_results(r: dict, api_key: str):
    """Render the full results panel for one engine's state dict."""
    if r["results_df"] is None:
        st.info("Run the pipeline to see results here.")
        return

    if r.get("save_path"):
        st.success(f"Experiment saved: `{r['save_path']}`")

    st.markdown("### Leaderboard")
    st.dataframe(r["results_df"], use_container_width=True)

    curves = r.get("winning_curves") or {}

    # ── Scientific Insights tabs ───────────────────────────────────────────────
    labels, renderers = [], []

    if r.get("timing_data"):
        labels.append("Training Time")
        renderers.append(lambda d=r: render_timing_chart(d["timing_data"]))

    if r.get("fold_scores"):
        labels.append("CV Distribution")
        renderers.append(lambda d=r: render_cv_boxplot(d["fold_scores"]))

    if r.get("learning_curve"):
        labels.append("Learning Curve")
        renderers.append(lambda d=r: render_learning_curve(d["learning_curve"]))

    if r.get("feature_ablation_results"):
        labels.append("Feature Ablation")
        renderers.append(lambda d=r: render_feature_ablation_chart(d["feature_ablation_results"]))

    if curves.get("bo_scores"):
        labels.append("BO Convergence")
        renderers.append(lambda c=curves: render_convergence_plot(c["bo_scores"]))

    if curves.get("train_losses") and curves.get("val_f1s"):
        labels.append("Training Curves")
        renderers.append(
            lambda c=curves: render_training_curve(c["train_losses"], c["val_f1s"]))

    if curves.get("feature_importances"):
        labels.append("Feature Importance")
        renderers.append(
            lambda c=curves: render_feature_importance(c["feature_importances"]))

    if r.get("pdp_data"):
        labels.append("PDP")
        renderers.append(lambda d=r: render_pdp(d["pdp_data"]))

    if curves.get("shap_values"):
        labels.append("SHAP Global")
        renderers.append(lambda c=curves: render_shap_plot(c["shap_values"]))

    if r.get("shap_waterfall"):
        labels.append("SHAP Waterfall")
        renderers.append(lambda d=r: render_shap_waterfall(d["shap_waterfall"]))

    if r.get("lime_data"):
        labels.append("LIME")
        renderers.append(lambda d=r: render_lime_explanation(d["lime_data"]))

    if curves.get("roc_data"):
        labels.append("ROC Curves")
        renderers.append(lambda c=curves: render_roc_curves(c["roc_data"]))

    if r.get("calibration_data"):
        labels.append("Calibration")
        renderers.append(lambda d=r: render_calibration_curve(d["calibration_data"]))

    if curves.get("confusion_matrix"):
        labels.append("Confusion Matrix")
        renderers.append(
            lambda c=curves: render_confusion_matrix(c["confusion_matrix"]))

    if curves.get("gradcam"):
        labels.append("Grad-CAM")
        renderers.append(lambda c=curves: render_gradcam(c["gradcam"]))

    if labels:
        st.markdown("---")
        st.markdown("### Scientific Insights")
        subtabs = st.tabs(labels)
        for tab, fn in zip(subtabs, renderers):
            with tab:
                fn()

    if r.get("best_params"):
        with st.expander("Best Hyperparameters", expanded=False):
            st.json(r["best_params"])

    if r.get("class_report"):
        st.markdown("---")
        _task = r.get("task_type")
        if _task == "regression":
            st.markdown("### Regression Report")
            render_regression_report(r["class_report"])
            render_regression_residuals(r["class_report"])
        elif _task == "multilabel" or r["class_report"].get("_task") == "multilabel":
            st.markdown("### Multi-label Classification Report")
            render_multilabel_report(r["class_report"])
            cm_ml = curves.get("confusion_matrix") or (
                {"matrix": r["class_report"].get("confusion_matrix"),
                 "labels": r["class_report"].get("confusion_matrix_labels", []),
                 "multilabel": True}
                if r["class_report"].get("confusion_matrix") else None
            )
            if cm_ml:
                render_multilabel_confusion_matrix(cm_ml)
        else:
            st.markdown("### Classification Report")
            render_classification_report(r["class_report"], r.get("label_classes"))
            render_per_class_accuracy(r["class_report"], r.get("label_classes"))
            cm_in_report = r["class_report"].get("confusion_matrix")
            if cm_in_report and not curves.get("confusion_matrix"):
                render_confusion_matrix({
                    "matrix": cm_in_report,
                    "labels": r["class_report"].get("confusion_matrix_labels", []),
                })

    st.markdown("---")
    render_export_buttons(
        results_df=r.get("results_df"),
        html_content=r.get("html_report_str") or "",
        model_bytes=r.get("exported_model_bytes"),
        model_filename=r.get("exported_model_filename") or "MLatelier_model",
    )
    _all_paths = (r.get("winning_curves") or {}).get("all_exported_paths", {})
    if len(_all_paths) > 1:
        with st.expander("Download All Models", expanded=False):
            render_all_models_download(_all_paths)
    with st.expander("Export LaTeX Table", expanded=False):
        render_latex_export(r.get("results_df"))


from file_utils import (
    NamedBytesIO as _NamedBytesIO,
    TABULAR_TYPES as _TABULAR_TYPES,
    file_ext as _file_ext,
    read_tabular as _read_tabular,
    sheet_selector as _sheet_selector_fn,
)

_TABULAR_ACCEPT = _TABULAR_TYPES


def _sheet_selector(src, key: str):
    """Thin wrapper that passes st.selectbox as the picker function."""
    return _sheet_selector_fn(src, st.selectbox, key)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## MLatelier")
    try:
        dev = get_device_info()
        if dev["type"] == "cuda":
            st.success(
                f"GPU: {dev.get('name', 'CUDA')}  "
                f"({dev.get('memory_gb', '?')} GB | CUDA {dev.get('cuda_version', '')})"
            )
        elif dev["type"] == "mps":
            st.success("Accelerator: Apple MPS")
        else:
            st.info("CPU mode (no GPU detected)")
    except Exception:
        st.info("CPU mode")

    st.markdown("---")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        help="Optional — powers the AI Assistant tab.",
    )
    _PRESET_MODELS = [
        "gemini-1.5-flash",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "Custom…",
    ]
    _model_sel = st.selectbox(
        "Gemini Model",
        _PRESET_MODELS,
        key="_gemini_model_sel",
        disabled=not api_key,
        help="Pick the model your API key supports. gemini-1.5-flash works on most free-tier keys.",
    )
    if _model_sel == "Custom…":
        _custom_model = st.text_input(
            "Model ID",
            key="_gemini_custom_model",
            placeholder="e.g. gemini-2.5-pro-preview-05-06",
        )
        gemini_model = _custom_model.strip() if _custom_model.strip() else "gemini-1.5-flash"
    else:
        gemini_model = _model_sel


# ── Header ────────────────────────────────────────────────────────────────────
_hdr_left, _hdr_right = st.columns([6, 1])
with _hdr_left:
    st.markdown(
        '<div class="hero-title">MLatelier <span class="hero-accent">Dashboard</span></div>'
        '<div class="hero-sub">Zero-code ML · Bayesian Optimization · '
        'SHAP · Grad-CAM · Ablation Study · Inference</div>',
        unsafe_allow_html=True,
    )
with _hdr_right:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⛔ Stop App", use_container_width=True,
                 help="Shut down the MLatelier server process"):
        import os
        os._exit(0)
st.markdown("<hr>", unsafe_allow_html=True)

# ── Metric cards (last-run engine) ────────────────────────────────────────────
_last = st.session_state.last_engine
_r    = st.session_state[_last] if _last else {}
render_top_metrics(
    _r.get("best_acc", "—"),
    _r.get("avg_acc", "—"),
    _r.get("imp", "—"),
    _r.get("pval", "—"),
    "R² Score" if _r.get("task_type") == "regression" else "Macro F1",
)
st.markdown("<br>", unsafe_allow_html=True)


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_tabular, tab_vision, tab_ablation, tab_predict, tab_nlp, tab_ai = st.tabs([
    "Tabular ML",
    "Vision DL",
    "Ablation Study",
    "Predict",
    "NLP",
    "AI Assistant",
])


# ══════════════════════════════════════════════════════════════════════════════
# TABULAR ML TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_tabular:
    cfg_col, res_col = st.columns([1, 2], gap="large")
    tab_run_btn = False

    with cfg_col:
        st.markdown("#### Dataset")

        tab_uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=_TABULAR_ACCEPT,
            key="tab_file_upload",
        )
        tab_path = st.text_input(
            "Or local path / URL",
            placeholder="D:/data.xlsx  or  https://.../data.csv",
            key="tab_path_input",
        )

        df = None
        if not tab_uploaded and tab_path:
            if tab_path.startswith("http://") or tab_path.startswith("https://"):
                with st.spinner("Downloading…"):
                    try:
                        resp = requests.get(tab_path, timeout=20)
                        resp.raise_for_status()
                        fn = urlparse(tab_path).path.split("/")[-1] or "data.csv"
                        fn = fn or "data.csv"
                        tab_uploaded = _NamedBytesIO(resp.content, fn)
                    except Exception as e:
                        st.error(f"Download failed: {e}")
            elif os.path.exists(tab_path):
                tab_uploaded = tab_path
            else:
                st.error("Path not found.")

        if tab_uploaded is not None:
            try:
                _tab_sheet = _sheet_selector(tab_uploaded, key="tab_sheet")
                df = _read_tabular(tab_uploaded, sheet_name=_tab_sheet)
                fname = (tab_uploaded if isinstance(tab_uploaded, str)
                         else tab_uploaded.name)
                st.success(
                    f"{os.path.basename(fname)} — {len(df):,} rows × {len(df.columns)} cols"
                )
            except Exception as e:
                st.error(f"Could not read file: {e}")

        # ── Feature selection ─────────────────────────────────────────────────
        target_y = features_x = None
        is_reg = False
        is_multilabel = False
        if df is not None:
            st.markdown("#### Features")
            cols = df.columns.tolist()
            target_y = st.multiselect(
                "Target Y column(s) — select multiple for multi-label",
                cols,
                default=[cols[-1]],
                key="tab_target",
            )
            # Flatten: single item → str for backward-compat with single-label path
            _target_arg = target_y[0] if len(target_y) == 1 else target_y
            features_x = st.multiselect(
                "Features (X)",
                [c for c in cols if c not in set(target_y)],
                default=[c for c in cols if c not in set(target_y)],
                key="tab_features",
            )
            if target_y:
                is_multilabel = len(target_y) > 1
                if is_multilabel:
                    st.info(f"Multi-label mode — {len(target_y)} label columns selected.")
                elif _is_regression(df[target_y[0]]):
                    is_reg = True
                    st.info("Continuous target — Regression mode.")
                else:
                    vc = df[target_y[0]].value_counts()
                    if len(vc) > 1 and vc.min() < 0.25 * vc.max():
                        st.warning("Imbalanced classes detected.")

        # ── Model selection ───────────────────────────────────────────────────
        tab_models: list[str] = []
        if df is not None and target_y and features_x:
            st.markdown("#### Models")
            available = _available_models(is_reg)
            groups = _model_groups(is_reg)
            for gname, glist in groups.items():
                opts = [m for m in glist if m in available]
                if not opts:
                    continue
                with st.expander(gname):
                    sel = st.multiselect(
                        gname, opts, default=[],
                        key=f"tab_models_{gname}",
                        label_visibility="collapsed",
                    )
                    tab_models.extend(sel)
            tab_models = list(dict.fromkeys(tab_models))

        # ── Data handling ─────────────────────────────────────────────────────
        selected_scaler = "auto"
        handle_imbalance = False
        if df is not None and target_y:
            st.markdown("#### Data Handling")
            if is_reg:
                st.caption("SMOTE disabled for regression targets.")
            else:
                handle_imbalance = st.toggle(
                    "Auto-balance imbalanced classes", value=True, key="tab_imb")
            with st.expander("Scaling strategy"):
                scaler_ui = st.selectbox(
                    "Scaler",
                    ["Auto (Recommended)", "RobustScaler",
                     "StandardScaler", "MinMaxScaler"],
                    key="tab_scaler",
                )
                selected_scaler = {
                    "Auto (Recommended)": "auto",
                    "RobustScaler": "robust",
                    "StandardScaler": "standard",
                    "MinMaxScaler": "minmax",
                }[scaler_ui]

        # ── Advanced settings ─────────────────────────────────────────────────
        test_size    = 0.2
        cv_folds     = None
        n_iter       = 10
        random_state = 42
        export_model = True
        run_full_opt = True

        if df is not None and target_y and features_x:
            with st.expander("Advanced settings"):
                run_full_opt = st.toggle(
                    "Run Bayesian Optimization", value=True, key="tab_opt")
                test_size = st.slider(
                    "Test split", 0.1, 0.4, 0.2, 0.05, key="tab_ts")
                cv_folds_ui = st.selectbox(
                    "CV folds", [3, 5, 10, "auto"], key="tab_cv")
                cv_folds = None if cv_folds_ui == "auto" else int(cv_folds_ui)
                n_iter = st.slider(
                    "BO iterations (base)", 5, 30, 10, key="tab_niter")
                random_state = st.number_input(
                    "Random seed", 0, 9999, 42, key="tab_seed")
                export_model = st.toggle(
                    "Export trained model", value=True, key="tab_export")

        # ── Run button ────────────────────────────────────────────────────────
        if df is not None and target_y and features_x and tab_models:
            tab_run_btn = st.button(
                "Run Tabular Pipeline",
                type="primary",
                use_container_width=True,
                key="tab_run",
            )
        elif df is not None:
            st.button(
                "Run Tabular Pipeline",
                type="primary",
                use_container_width=True,
                disabled=True,
                key="tab_run_dis",
            )
            if not tab_models:
                st.caption("Select at least one model above.")

    # ── Results column ────────────────────────────────────────────────────────
    with res_col:
        # Full EDA panel (shown when data is loaded, before training)
        if df is not None and features_x and target_y:
            with st.expander("Exploratory Data Analysis", expanded=False):
                eda_tabs = st.tabs([
                    "Summary Stats", "Distributions",
                    "Box Plots", "Target", "Missing",
                    "Correlations", "Quality",
                ])
                with eda_tabs[0]:
                    render_data_preview(df)
                    render_statistical_summary(df, list(features_x))
                with eda_tabs[1]:
                    render_feature_distributions(df, list(features_x))
                with eda_tabs[2]:
                    render_feature_boxplots(df, list(features_x))
                with eda_tabs[3]:
                    task_guess = "regression" if is_reg else "classification"
                    if is_multilabel:
                        for _lbl in target_y:
                            st.markdown(f"**{_lbl}**")
                            render_target_distribution(df, _lbl, "classification")
                    else:
                        render_target_distribution(df, target_y[0], task_guess)
                with eda_tabs[4]:
                    _all_target_cols = list(target_y)
                    render_missing_heatmap(df, list(features_x) + _all_target_cols)
                with eda_tabs[5]:
                    render_correlation_heatmap(df, list(features_x))
                with eda_tabs[6]:
                    render_data_quality_report(df, list(features_x), target_y[0])

        if tab_run_btn:
            gc.collect()
            try:
                import torch as _t
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
            except ImportError:
                pass

            status_text  = st.empty()
            progress_bar = st.progress(0)
            if is_reg:
                metric_name = "R² Score"
            elif is_multilabel:
                metric_name = "Micro F1"
            else:
                metric_name = "Macro F1"

            # ── Live racing leaderboard ────────────────────────────────────────
            race_header = st.empty()
            race_table  = st.empty()
            live_rows: list = []

            def _on_tab_model(name, mean, std, elapsed, folds):
                live_rows.append({
                    "Model":           name,
                    metric_name:       f"{mean:.4f}" if not np.isnan(mean) else "—",
                    "±Std":            f"{std:.4f}" if not np.isnan(mean) else "—",
                    "Time (s)":        elapsed,
                    "Status":          "✓",
                })
                with race_header.container():
                    st.markdown("#### Live Baseline Race")
                with race_table.container():
                    df_live = pd.DataFrame(live_rows)
                    try:
                        df_live = df_live.sort_values(metric_name, ascending=False)
                    except Exception:
                        pass
                    st.dataframe(df_live, use_container_width=True, hide_index=True)

            # Phase 1 — Baselines
            status_text.info("Running baseline cross-validation…")
            baseline_results, task_type = run_tabular_baseline(
                df, features_x, target_y, tab_models, handle_imbalance,
                random_state=int(random_state), scaler=selected_scaler,
                test_size=float(test_size), cv_folds=cv_folds,
                progress_bar=progress_bar, status_text=status_text,
                model_callback=_on_tab_model,
            )
            race_header.empty()
            race_table.empty()

            timing_data = {
                m: v.get("time_s", 0.0)
                for m, v in baseline_results.items()
                if isinstance(v, dict)
            }
            fold_scores = {
                m: v.get("fold_scores", [])
                for m, v in baseline_results.items()
                if isinstance(v, dict) and v.get("fold_scores")
            }

            if run_full_opt:
                status_text.info("Running Bayesian Optimization…")
                progress_bar.progress(0.0)
                export_dir = os.path.join(os.path.expanduser("~"), "MLatelier", "models")
                (
                    results_list, best_acc, avg_acc, imp, p_val,
                    best_params, class_report, label_classes,
                    task_type, winning_curves, actual_scaler, exported_path,
                ) = run_tabular_optimization(
                    df, features_x, target_y, tab_models, baseline_results,
                    handle_imbalance, random_state=int(random_state),
                    scaler=selected_scaler, test_size=float(test_size),
                    cv_folds=cv_folds, n_iter=int(n_iter),
                    export_model=export_model,
                    export_dir=export_dir if export_model else None,
                    progress_bar=progress_bar, status_text=status_text,
                )

                # Add baseline timing into results_list for the leaderboard
                if results_list:
                    for row in results_list:
                        m = row.get("Algorithm", "")
                        if "Training Time (s)" not in row and m in timing_data:
                            row["Training Time (s)"] = timing_data[m]

                # Post-optimization interpretability (uses the exported winning model)
                calibration_data = pdp_data = shap_wf = lime_expl = None
                _winning_pipe = None
                _X_test_ref = _X_train_ref = None
                if winning_curves:
                    ep = winning_curves.get("exported_model_path")
                    if ep and os.path.exists(ep):
                        try:
                            import joblib
                            from tabular_engine import _prepare_data
                            _winning_pipe = joblib.load(ep)
                            _d = _prepare_data(
                                df, list(features_x), target_y, handle_imbalance,
                                random_state=int(random_state),
                                scaler=actual_scaler or selected_scaler,
                                test_size=float(test_size), cv_folds=cv_folds)
                            _X_test_ref  = _d.X_test
                            _X_train_ref = _d.X_train
                            feat_names   = _d.feature_names

                            if task_type == "classification":
                                try:
                                    calibration_data = compute_calibration_data(
                                        _winning_pipe, _d.X_test, _d.y_test,
                                        _d.class_names)
                                except Exception:
                                    pass

                            # PDP/SHAP/LIME not applicable for multilabel pipelines
                            if task_type != "multilabel":
                                status_text.info("Computing PDP…")
                                try:
                                    pdp_data = compute_pdp(
                                        _winning_pipe, _d.X_test, feat_names, top_n=6)
                                except Exception:
                                    pass

                                status_text.info("Computing SHAP waterfall…")
                                try:
                                    shap_wf = compute_shap_waterfall(
                                        _winning_pipe, _d.X_test, feat_names, sample_idx=0)
                                except Exception:
                                    pass

                                status_text.info("Computing LIME explanation…")
                                try:
                                    lime_expl = compute_lime_explanation(
                                        _winning_pipe, _d.X_train, _d.X_test,
                                        feat_names, task_type, sample_idx=0)
                                except Exception:
                                    pass
                        except Exception:
                            pass
            else:
                status_text.info("Baselines complete.")
                results_list = []
                valid = [
                    v["mean"] for v in baseline_results.values()
                    if isinstance(v, dict) and not pd.isna(v.get("mean", float("nan")))
                ]
                best_acc  = max(valid) if valid else 0.0
                avg_acc   = sum(valid) / len(valid) if valid else 0.0
                imp, p_val = 0.0, "N/A"
                class_report = winning_curves = None
                best_params  = {"Status": "Optimization skipped"}
                label_classes = None
                actual_scaler = selected_scaler
                exported_path = None
                calibration_data = pdp_data = shap_wf = lime_expl = None
                for m, bl in baseline_results.items():
                    mean = bl.get("mean", float("nan")) if isinstance(bl, dict) else float("nan")
                    std  = bl.get("std", 0.0) if isinstance(bl, dict) else 0.0
                    t_s  = bl.get("time_s", 0.0) if isinstance(bl, dict) else 0.0
                    results_list.append({
                        "Algorithm":              m,
                        f"CV {metric_name} (mean)": mean,
                        f"CV {metric_name} (±std)": std,
                        "Baseline Time (s)":      t_s,
                    })

            progress_bar.empty()
            status_text.empty()

            if results_list is not None:
                res_df   = pd.DataFrame(results_list)
                sort_col = next(
                    (c for c in [f"Selection CV {metric_name}",
                                  f"CV {metric_name} (mean)"]
                     if c in res_df.columns),
                    res_df.columns[-1],
                )
                res_df = res_df.sort_values(sort_col, ascending=False).reset_index(drop=True)
                fmt    = _fmt_pct if task_type in ("classification", "multilabel") else _fmt_f4
                numeric_score_cols = [c for c in res_df.columns
                                      if c not in ("Algorithm",)
                                      and "Time" not in c]
                for col in numeric_score_cols:
                    res_df[col] = res_df[col].apply(
                        lambda x: fmt(x) if isinstance(x, float) else (x or "—"))

                best_acc_str = (f"{best_acc * 100:.2f}%"
                                if task_type in ("classification", "multilabel")
                                else f"{best_acc:.4f}")
                avg_acc_str  = (f"{avg_acc * 100:.2f}%"
                                if task_type in ("classification", "multilabel")
                                else f"{avg_acc:.4f}")

                model_bytes = None
                model_fname = None
                if exported_path and os.path.exists(exported_path):
                    with open(exported_path, "rb") as f:
                        model_bytes = f.read()
                    model_fname = os.path.basename(exported_path)

                try:
                    from tracker import save_minimal_experiment
                    saved, html = save_minimal_experiment(
                        task_type=task_type, models_raced=tab_models,
                        results_df=res_df, report_dict=class_report or {},
                        best_score=best_acc_str, winning_curves=winning_curves,
                        scaler_used=actual_scaler, best_params=best_params,
                        label_classes=label_classes,
                        experiment_metadata=(winning_curves or {}).get(
                            "reproducibility_metadata"),
                    )
                except Exception:
                    saved, html = None, ""

                st.session_state.tabular.update({
                    "results_df":    res_df,
                    "best_acc":      best_acc_str,
                    "avg_acc":       avg_acc_str,
                    "imp":           f"+{imp:.2f}%" if run_full_opt else "N/A",
                    "pval":          p_val,
                    "best_params":   best_params,
                    "class_report":  class_report,
                    "label_classes": label_classes,
                    "winning_curves": winning_curves,
                    "task_type":     task_type,
                    "save_path":     saved,
                    "html_report_str": html,
                    "exported_model_bytes":    model_bytes,
                    "exported_model_filename": model_fname,
                    "baseline_results":        baseline_results,
                    "calibration_data":        calibration_data,
                    "timing_data":             timing_data,
                    "fold_scores":             fold_scores,
                    "pdp_data":                pdp_data,
                    "shap_waterfall":          shap_wf,
                    "lime_data":               lime_expl,
                    # store raw data refs for on-demand learning curve / feature ablation
                    "_df":               df,
                    "_features_x":       list(features_x),
                    "_target_y":         target_y,
                    "_handle_imbalance": handle_imbalance,
                    "_scaler":           actual_scaler or selected_scaler,
                })
                st.session_state.last_engine = "tabular"
                st.rerun()

        # ── On-demand: Learning Curve ─────────────────────────────────────────
        _tab_st = st.session_state.tabular
        if _tab_st.get("results_df") is not None and _tab_st.get("_df") is not None:
            with st.expander("On-demand Analysis", expanded=False):
                oda_t1, oda_t2 = st.tabs(["Learning Curve", "Feature Ablation"])
                with oda_t1:
                    lc_model_opts = _tab_st.get("_df") is not None and _tab_st.get("_features_x")
                    if lc_model_opts:
                        _avail = sorted(_available_models(
                            _tab_st.get("task_type") == "regression"))
                        lc_model = st.selectbox(
                            "Model for learning curve", _avail, key="lc_model")
                        lc_btn = st.button("Compute Learning Curve", key="lc_btn")
                        if lc_btn:
                            try:
                                lc_prog = st.progress(0)
                                lc_stat = st.empty()
                                lc_data = compute_learning_curve(
                                    _tab_st["_df"],
                                    _tab_st["_features_x"],
                                    _tab_st["_target_y"],
                                    lc_model,
                                    handle_imbalance=_tab_st.get("_handle_imbalance", False),
                                    random_state=42,
                                    scaler=_tab_st.get("_scaler", "auto"),
                                    progress_bar=lc_prog,
                                    status_text=lc_stat,
                                )
                                lc_prog.empty()
                                lc_stat.empty()
                                st.session_state.tabular["learning_curve"] = lc_data
                                st.rerun()
                            except Exception as _e:
                                st.error(f"Learning curve failed: {_e}")
                        if _tab_st.get("learning_curve"):
                            render_learning_curve(_tab_st["learning_curve"])

                with oda_t2:
                    if _tab_st.get("_df") is not None and _tab_st.get("_features_x"):
                        _fablate_opts = sorted(_available_models(
                            _tab_st.get("task_type") == "regression"))
                        fa_model = st.selectbox(
                            "Model for feature ablation", _fablate_opts, key="fa_model")
                        fa_btn = st.button("Run Feature Ablation", key="fa_btn")
                        if fa_btn:
                            try:
                                fa_prog = st.progress(0)
                                fa_stat = st.empty()
                                fa_results = run_feature_ablation(
                                    _tab_st["_df"],
                                    _tab_st["_features_x"],
                                    _tab_st["_target_y"],
                                    fa_model,
                                    handle_imbalance=_tab_st.get("_handle_imbalance", False),
                                    random_state=42,
                                    scaler=_tab_st.get("_scaler", "auto"),
                                    progress_bar=fa_prog,
                                    status_text=fa_stat,
                                )
                                fa_prog.empty()
                                fa_stat.empty()
                                st.session_state.tabular["feature_ablation_results"] = fa_results
                                st.rerun()
                            except Exception as _e:
                                st.error(f"Feature ablation failed: {_e}")
                        if _tab_st.get("feature_ablation_results"):
                            render_feature_ablation_chart(_tab_st["feature_ablation_results"])

        _render_results(st.session_state.tabular, api_key)


# ══════════════════════════════════════════════════════════════════════════════
# VISION DL TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_vision:
    vis_cfg_col, vis_res_col = st.columns([1, 2], gap="large")
    vis_run_btn = False

    with vis_cfg_col:
        st.markdown("#### Image Dataset")

        vis_folder = st.text_input(
            "Folder path (recommended)",
            placeholder=r"C:\datasets\my_images",
            help="Each sub-folder = one class label. No zipping needed.",
            key="vis_folder",
        )
        st.caption("Structure:  `dataset/class_a/img1.jpg`  `dataset/class_b/img2.jpg`")
        st.markdown("---")
        st.caption("No local folder? Upload a ZIP instead:")
        vis_zip = st.file_uploader(
            "ZIP file", type=["zip"],
            label_visibility="collapsed",
            key="vis_zip",
        )

        vis_data_ok    = False
        vis_input_type = None
        if vis_folder:
            if os.path.isdir(vis_folder):
                vis_data_ok    = True
                vis_input_type = "folder"
                st.success(f"Folder found: `{os.path.basename(vis_folder)}`")
            else:
                st.error("Folder not found — check the path.")
        if vis_zip and not vis_folder:
            vis_data_ok    = True
            vis_input_type = "zip"
            st.success(f"ZIP loaded: `{vis_zip.name}`")

        vis_models: list[str] = []
        if vis_data_ok:
            st.markdown("#### Architectures")
            _vis_groups = {
                "Classic CNNs":        ["ResNet18", "ResNet50", "VGG16"],
                "Efficient Networks":  ["EfficientNet_B0", "EfficientNet_V2_S",
                                        "MobileNet_v3", "DenseNet121"],
                "ConvNeXt":            ["ConvNeXt_T", "ConvNeXt_S"],
                "Vision Transformers": ["ViT_B_16", "Swin_T", "Swin_S"],
            }
            for gname, gopts in _vis_groups.items():
                with st.expander(gname):
                    sel = st.multiselect(
                        gname, gopts, default=[],
                        key=f"vis_arch_{gname}",
                        label_visibility="collapsed",
                    )
                    vis_models.extend(sel)
            vis_models = list(dict.fromkeys(vis_models))

        vis_epochs     = 10
        vis_baseline_e = 3
        vis_batch      = 32
        vis_n_bo       = 20
        vis_es_pat     = 5
        vis_lr_sched   = "cosine"
        vis_aug        = True
        vis_freeze     = "head_only"
        vis_imb        = True
        vis_export     = True
        vis_seed       = 42
        vis_full_opt   = True

        if vis_data_ok and vis_models:
            st.markdown("#### Training")
            vis_full_opt = st.toggle(
                "Run Bayesian Optimization", value=True, key="vis_opt")
            vis_epochs = st.slider("Training epochs", 1, 50, 10, key="vis_ep")
            vis_batch  = st.select_slider(
                "Batch size", [8, 16, 32, 64, 128], value=32, key="vis_bs")
            vis_aug    = st.toggle("Data augmentation", value=True, key="vis_aug")
            vis_freeze_ui = st.radio(
                "Fine-tune strategy",
                ["Head only", "Last block + head"],
                horizontal=True, key="vis_freeze",
            )
            vis_freeze = "head_only" if vis_freeze_ui == "Head only" else "last_block"
            vis_imb    = st.toggle(
                "Handle class imbalance", value=True, key="vis_imb")

            with st.expander("Advanced"):
                vis_baseline_e = st.slider(
                    "Baseline epochs", 1, 5, 3, key="vis_be")
                vis_n_bo   = st.slider(
                    "BO trials per architecture", 5, 50, 20, key="vis_nbo")
                vis_es_pat = st.slider(
                    "Early stopping patience (0=off)", 0, 15, 5, key="vis_esp")
                vis_lr_sched = st.selectbox(
                    "LR scheduler", ["cosine", "step", "reduce", "none"],
                    key="vis_lrs")
                vis_seed   = st.number_input(
                    "Random seed", 0, 9999, 42, key="vis_seed")
                vis_export = st.toggle(
                    "Export trained model (.pt)", value=True, key="vis_exp")

        if vis_data_ok and vis_models:
            vis_run_btn = st.button(
                "Run Vision Pipeline",
                type="primary",
                use_container_width=True,
                key="vis_run",
            )
        elif vis_data_ok:
            st.button(
                "Run Vision Pipeline",
                type="primary",
                use_container_width=True,
                disabled=True,
                key="vis_run_dis",
            )
            st.caption("Select at least one architecture above.")

    with vis_res_col:
        # Vision dataset EDA (shown whenever a valid folder is available)
        _vis_eda_dir = vis_folder if vis_input_type == "folder" and vis_data_ok else None
        if _vis_eda_dir:
            with st.expander("Dataset Analysis", expanded=False):
                try:
                    _summary = get_vision_dataset_summary(_vis_eda_dir)
                    vis_eda_t1, vis_eda_t2 = st.tabs(["Class Distribution", "Sample Images"])
                    with vis_eda_t1:
                        render_vision_class_distribution(_summary.get("class_counts", {}))
                    with vis_eda_t2:
                        render_vision_sample_grid(_summary.get("samples", {}))
                except Exception as _e:
                    st.info(f"Dataset preview unavailable: {_e}")

        if vis_run_btn:
            gc.collect()
            try:
                import torch as _t
                if _t.cuda.is_available():
                    _t.cuda.empty_cache()
            except ImportError:
                pass

            temp_dir = None
            data_dir = None
            try:
                if vis_input_type == "folder":
                    data_dir = vis_folder
                else:
                    temp_dir = tempfile.mkdtemp()
                    dest = os.path.realpath(temp_dir)
                    with zipfile.ZipFile(vis_zip, "r") as zf:
                        for member in zf.namelist():
                            mp = os.path.realpath(os.path.join(dest, member))
                            if not mp.startswith(dest + os.sep):
                                st.error(f"Security: unsafe path in ZIP: {member}")
                                st.stop()
                        zf.extractall(temp_dir)
                    data_dir = temp_dir

                status_text  = st.empty()
                progress_bar = st.progress(0)

                # ── Live epoch progress placeholder ────────────────────────────
                epoch_ph      = st.empty()
                vis_live_rows: list = []
                vis_race_ph   = st.empty()

                def _on_vis_epoch(ep, total_ep, loss, f1):
                    with epoch_ph.container():
                        st.progress(ep / max(total_ep, 1))
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Epoch", f"{ep}/{total_ep}")
                        c2.metric("Train Loss", f"{loss:.4f}")
                        c3.metric("Val F1", f"{f1:.4f}")
                        c4.metric("Best F1", f"{max((r.get('Opt F1', 0.0) for r in vis_live_rows), default=0.0):.4f}")

                def _on_vis_model(name, opt_score, elapsed):
                    vis_live_rows.append({
                        "Architecture": name,
                        "Opt F1":       round(opt_score, 4),
                        "Time (s)":     elapsed,
                    })
                    with vis_race_ph.container():
                        st.dataframe(
                            pd.DataFrame(vis_live_rows).sort_values(
                                "Opt F1", ascending=False),
                            use_container_width=True, hide_index=True)

                # Phase 1 — Baselines
                status_text.info("Computing vision baselines…")
                baseline_results, class_names = run_vision_baseline(
                    data_dir=data_dir,
                    selected_models=vis_models,
                    batch_size=vis_batch,
                    use_augmentation=vis_aug,
                    freeze_strategy=vis_freeze,
                    handle_imbalance=vis_imb,
                    baseline_epochs=vis_baseline_e,
                    progress_bar=progress_bar,
                    status_text=status_text,
                    random_seed=int(vis_seed),
                    epoch_callback=_on_vis_epoch,
                )
                epoch_ph.empty()

                vis_timing = {
                    m: v.get("time_s", 0.0)
                    for m, v in baseline_results.items()
                    if isinstance(v, dict)
                }

                if vis_full_opt:
                    status_text.info("Running Deep Learning Bayesian Optimization…")
                    progress_bar.progress(0.0)
                    export_dir = os.path.join(
                        os.path.expanduser("~"), "MLatelier", "models")
                    (
                        results_list, best_f1, avg_base, imp, p_val,
                        params, report, classes, winning_curves, exported_path,
                    ) = run_vision_optimization(
                        data_dir=data_dir,
                        selected_models=vis_models,
                        baseline_results=baseline_results,
                        epochs=vis_epochs,
                        batch_size=vis_batch,
                        use_augmentation=vis_aug,
                        freeze_strategy=vis_freeze,
                        handle_imbalance=vis_imb,
                        n_bo_calls=vis_n_bo,
                        early_stopping_patience=vis_es_pat,
                        lr_scheduler=vis_lr_sched,
                        export_model=vis_export,
                        export_dir=export_dir if vis_export else None,
                        progress_bar=progress_bar,
                        status_text=status_text,
                        random_seed=int(vis_seed),
                        epoch_callback=_on_vis_epoch,
                        model_callback=_on_vis_model,
                    )
                else:
                    status_text.info("Baselines complete.")
                    results_list = []
                    valid = [
                        v["base_score"] for v in baseline_results.values()
                        if not pd.isna(v.get("base_score", float("nan")))
                    ]
                    best_f1  = max(valid) if valid else 0.0
                    avg_base = sum(valid) / len(valid) if valid else 0.0
                    imp, p_val, report, winning_curves = 0.0, "N/A", None, None
                    params = {"Status": "Optimization skipped"}
                    classes = class_names
                    exported_path = None
                    for mn, bd in baseline_results.items():
                        results_list.append({
                            "Architecture":    mn,
                            "Baseline Val F1": bd.get("base_score", pd.NA),
                            "Time (s)":        bd.get("time_s", 0.0),
                        })

                epoch_ph.empty()
                vis_race_ph.empty()
                progress_bar.empty()
                status_text.empty()

                if results_list is not None:
                    res_df   = pd.DataFrame(results_list)
                    sort_col = ("Final Test F1"
                                if vis_full_opt and "Final Test F1" in res_df.columns
                                else "Baseline Val F1")
                    if sort_col in res_df.columns:
                        res_df = res_df.sort_values(
                            sort_col, ascending=False, na_position="last")
                    res_df = res_df.reset_index(drop=True)
                    for col in ["Baseline Val F1", "Optimized Val F1", "Final Test F1"]:
                        if col in res_df.columns:
                            res_df[col] = res_df[col].apply(_fmt_pct)

                    best_acc_str = f"{best_f1 * 100:.2f}%"
                    avg_acc_str  = f"{avg_base * 100:.2f}%"

                    model_bytes = None
                    model_fname = None
                    if exported_path and os.path.exists(exported_path):
                        with open(exported_path, "rb") as f:
                            model_bytes = f.read()
                        model_fname = os.path.basename(exported_path)

                    # Merge opt timing into timing_data
                    if results_list and vis_full_opt:
                        for row in results_list:
                            mn = row.get("Architecture", "")
                            if mn and row.get("Training Time (s)"):
                                vis_timing[mn] = row["Training Time (s)"]

                    try:
                        from tracker import save_minimal_experiment
                        saved, html = save_minimal_experiment(
                            task_type="classification",
                            models_raced=vis_models,
                            results_df=res_df,
                            report_dict=report or {},
                            best_score=best_acc_str,
                            winning_curves=winning_curves,
                            best_params=params,
                            label_classes=classes,
                            experiment_metadata=(winning_curves or {}).get(
                                "reproducibility_metadata"),
                        )
                    except Exception:
                        saved, html = None, ""

                    st.session_state.vision.update({
                        "results_df":    res_df,
                        "best_acc":      best_acc_str,
                        "avg_acc":       avg_acc_str,
                        "imp":           f"+{imp:.2f}%" if vis_full_opt else "N/A",
                        "pval":          p_val if vis_full_opt else "N/A",
                        "best_params":   params,
                        "class_report":  report,
                        "label_classes": classes,
                        "winning_curves": winning_curves,
                        "task_type":     "classification",
                        "save_path":     saved,
                        "html_report_str": html,
                        "exported_model_bytes":    model_bytes,
                        "exported_model_filename": model_fname,
                        "baseline_results": baseline_results,
                        "timing_data":   vis_timing,
                        "fold_scores":   None,
                        "calibration_data": None,
                    })
                    st.session_state.last_engine = "vision"
                    st.rerun()

            except ValueError as ve:
                st.error(f"Dataset error: {ve}")
            except Exception as e:
                st.error(f"Vision pipeline failed: {e}")
            finally:
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)

        _render_results(st.session_state.vision, api_key)


# ══════════════════════════════════════════════════════════════════════════════
# ABLATION STUDY TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_ablation:
    st.markdown("### Ablation Study")
    st.caption(
        "Systematically remove or vary individual pipeline components to understand "
        "their contribution to model performance. Supports tabular pipelines."
    )
    abl_c1, abl_c2 = st.columns([1, 2], gap="large")

    with abl_c1:
        st.markdown("#### Dataset")
        abl_file = st.file_uploader(
            "Upload CSV or Excel", type=_TABULAR_ACCEPT, key="abl_file")
        abl_path = st.text_input(
            "Or local path", placeholder="D:/data.xlsx", key="abl_path")

        abl_df = None
        if not abl_file and abl_path and os.path.exists(abl_path):
            abl_file = abl_path
        if abl_file is not None:
            try:
                _abl_sheet = _sheet_selector(abl_file, key="abl_sheet")
                abl_df = _read_tabular(abl_file, sheet_name=_abl_sheet)
                fname  = abl_file if isinstance(abl_file, str) else abl_file.name
                st.success(f"{os.path.basename(fname)} — {len(abl_df):,} rows")
            except Exception as e:
                st.error(f"Cannot read file: {e}")

        abl_target   = None
        abl_features = None
        abl_model    = None
        if abl_df is not None:
            st.markdown("#### Target & Features")
            abl_cols   = abl_df.columns.tolist()
            abl_target = st.selectbox(
                "Target (Y)", abl_cols, index=len(abl_cols) - 1, key="abl_target")
            abl_features = st.multiselect(
                "Features (X)",
                [c for c in abl_cols if c != abl_target],
                default=[c for c in abl_cols if c != abl_target],
                key="abl_features",
            )

            st.markdown("#### Model to Ablate")
            _is_reg_abl = bool(abl_target and _is_regression(abl_df[abl_target]))
            _all_abl = sorted(_available_models(_is_reg_abl))
            abl_model = st.selectbox("Model", _all_abl, key="abl_model")

            st.markdown("#### What to Vary")
            # SMOTE is only meaningful for classification (never applied for regression/multilabel)
            _smote_applicable = not _is_reg_abl
            if _smote_applicable:
                abl_do_imb = st.checkbox("Imbalance handling (SMOTE on vs off)",
                                          key="abl_imb")
            else:
                abl_do_imb = False
                st.checkbox("Imbalance handling (SMOTE on vs off)",
                             key="abl_imb", value=False, disabled=True,
                             help="SMOTE is not applicable to regression tasks.")
            abl_do_scaler = st.checkbox("Feature scaling (auto / robust / standard / minmax)",
                                         key="abl_scaler")
            abl_do_split  = st.checkbox("Test split (10% / 20% / 30%)",
                                         key="abl_split")
            abl_do_cv     = st.checkbox("Cross-validation folds (3 / 5 / 10)",
                                         key="abl_cv")

            st.markdown("#### Baseline Configuration")
            if _smote_applicable:
                abl_base_imb = st.toggle("SMOTE", value=True, key="abl_base_imb")
            else:
                abl_base_imb = False
                st.caption("SMOTE: disabled (regression task)")
            abl_base_sc    = st.selectbox("Scaler",
                ["auto", "robust", "standard", "minmax"], key="abl_base_sc")
            abl_base_ts    = st.select_slider(
                "Test split", [0.10, 0.15, 0.20, 0.25, 0.30], value=0.20,
                key="abl_base_ts")
            abl_base_cv    = st.selectbox(
                "CV folds", [3, 5, 10], index=1, key="abl_base_cv")
            abl_seed       = st.number_input(
                "Random seed", 0, 9999, 42, key="abl_seed")

        abl_run_btn = False
        if (abl_df is not None and abl_target and abl_features and abl_model
                and (abl_do_imb or abl_do_scaler or abl_do_split or abl_do_cv)):
            abl_run_btn = st.button(
                "Run Ablation Study",
                type="primary",
                use_container_width=True,
                key="abl_run",
            )
        elif abl_df is not None and abl_model:
            st.button(
                "Run Ablation Study",
                type="primary",
                use_container_width=True,
                disabled=True,
                key="abl_run_dis",
            )
            st.caption("Tick at least one dimension to vary above.")

    with abl_c2:
        if abl_run_btn:
            # Build configs
            configs = [{
                "label":            "★ Baseline",
                "handle_imbalance": abl_base_imb,
                "scaler":           abl_base_sc,
                "test_size":        abl_base_ts,
                "cv_folds":         int(abl_base_cv),
            }]

            if abl_do_imb and not _is_reg_abl:
                configs.append({
                    "label":            "Without SMOTE" if abl_base_imb else "With SMOTE",
                    "handle_imbalance": not abl_base_imb,
                    "scaler":           abl_base_sc,
                    "test_size":        abl_base_ts,
                    "cv_folds":         int(abl_base_cv),
                })

            if abl_do_scaler:
                sc_labels = {
                    "auto":     "Auto Scaler",
                    "robust":   "RobustScaler",
                    "standard": "StandardScaler",
                    "minmax":   "MinMaxScaler",
                }
                for sc, slabel in sc_labels.items():
                    if sc != abl_base_sc:
                        configs.append({
                            "label":            slabel,
                            "handle_imbalance": abl_base_imb,
                            "scaler":           sc,
                            "test_size":        abl_base_ts,
                            "cv_folds":         int(abl_base_cv),
                        })

            if abl_do_split:
                for ts in [0.10, 0.20, 0.30]:
                    if abs(ts - abl_base_ts) > 0.01:
                        configs.append({
                            "label":            f"Test={int(ts * 100)}%",
                            "handle_imbalance": abl_base_imb,
                            "scaler":           abl_base_sc,
                            "test_size":        ts,
                            "cv_folds":         int(abl_base_cv),
                        })

            if abl_do_cv:
                for cf in [3, 5, 10]:
                    if cf != int(abl_base_cv):
                        configs.append({
                            "label":            f"CV {cf}-fold",
                            "handle_imbalance": abl_base_imb,
                            "scaler":           abl_base_sc,
                            "test_size":        abl_base_ts,
                            "cv_folds":         cf,
                        })

            abl_status = st.empty()
            abl_prog   = st.progress(0)
            abl_status.info(f"Running {len(configs)} ablation configurations…")

            try:
                abl_results = run_tabular_ablation(
                    df=abl_df,
                    features_x=abl_features,
                    target_y=abl_target,
                    model_name=abl_model,
                    ablation_configs=configs,
                    random_state=int(abl_seed),
                    progress_bar=abl_prog,
                    status_text=abl_status,
                )
                st.session_state.ablation_results = abl_results
            except Exception as e:
                st.error(f"Ablation failed: {e}")
                abl_results = None

            abl_status.empty()
            abl_prog.empty()

        if st.session_state.ablation_results:
            render_ablation_chart(st.session_state.ablation_results)

            csv_bytes = pd.DataFrame(
                st.session_state.ablation_results
            ).to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download Ablation Results (.csv)",
                data=csv_bytes,
                file_name="MLatelier_Ablation.csv",
                mime="text/csv",
            )
        else:
            if not abl_run_btn:
                st.info(
                    "Configure and run the ablation study on the left to see results here. "
                    "Each bar shows how one pipeline change affects CV performance."
                )


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT TAB  (Weka-style inference on new data)
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("### Predict on New Data")
    st.caption(
        "Load a trained MLatelier model and get predictions on unseen data. "
        "Use the model from the last training run, or upload a saved file."
    )

    pred_tab_t, pred_vis_t = st.tabs(["Tabular Predict", "Vision Predict"])

    # ── Tabular Predict ───────────────────────────────────────────────────────
    with pred_tab_t:
        tp_c1, tp_c2 = st.columns([1, 2], gap="large")

        with tp_c1:
            st.markdown("#### Load Model")

            # Option A — from session
            session_model_path = None
            _tw   = st.session_state.tabular.get("winning_curves") or {}
            _s_mp = _tw.get("exported_model_path")
            if _s_mp and os.path.exists(_s_mp):
                session_model_path = _s_mp
                st.success(f"Session model ready: `{os.path.basename(_s_mp)}`")

            use_session_model = False
            if session_model_path:
                use_session_model = st.toggle(
                    "Use model from last training run", value=True, key="tp_use_sess")

            # Option B — upload .joblib
            tp_model_file = None
            if not use_session_model:
                tp_model_file = st.file_uploader(
                    "Upload .joblib model file",
                    type=["joblib"],
                    key="tp_model_upload",
                )

            st.markdown("#### New Data")
            tp_new_file = st.file_uploader(
                "Upload CSV or Excel (new data, no target column needed)",
                type=_TABULAR_ACCEPT,
                key="tp_new_data",
            )

            tp_predict_btn = False
            _can_predict_t = (tp_new_file is not None and
                               (use_session_model or tp_model_file is not None))
            if _can_predict_t:
                tp_predict_btn = st.button(
                    "Predict",
                    type="primary",
                    use_container_width=True,
                    key="tp_predict",
                )
            else:
                st.button(
                    "Predict",
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="tp_predict_dis",
                )
                if tp_new_file is None:
                    st.caption("Upload new CSV data above.")
                elif not use_session_model and tp_model_file is None:
                    st.caption("Load or upload a trained model.")

        with tp_c2:
            if tp_predict_btn:
                try:
                    # Resolve model path
                    if use_session_model and session_model_path:
                        model_path = session_model_path
                    else:
                        # Write uploaded bytes to a temp file
                        _tmp = tempfile.NamedTemporaryFile(
                            suffix=".joblib", delete=False)
                        _tmp.write(tp_model_file.read())
                        _tmp.flush()
                        _tmp.close()
                        model_path = _tmp.name

                    pipeline, metadata = infer.load_tabular_model(model_path)

                    feature_names = metadata.get("feature_names") or []
                    class_names   = metadata.get("class_names")

                    # Fallback: try to get feature names from last training session
                    if not feature_names and st.session_state.tabular.get("winning_curves"):
                        _rm = (st.session_state.tabular["winning_curves"]
                               .get("reproducibility_metadata", {}))
                        feature_names = _rm.get("feature_columns", [])

                    _tp_sheet = _sheet_selector(tp_new_file, key="tp_sheet")
                    df_new = _read_tabular(tp_new_file, sheet_name=_tp_sheet)
                    st.info(
                        f"Loaded: {len(df_new):,} rows × {len(df_new.columns)} cols "
                        f"| Expected features: {len(feature_names)}"
                    )

                    if not feature_names:
                        st.warning(
                            "Feature names not found in model metadata. "
                            "Using all CSV columns as features."
                        )
                        feature_names = df_new.columns.tolist()

                    with st.spinner("Running predictions…"):
                        df_preds = infer.predict_tabular(
                            pipeline, df_new, feature_names, class_names)

                    st.session_state.inference_results = df_preds

                except Exception as e:
                    st.error(f"Tabular inference failed: {e}")

            if st.session_state.inference_results is not None:
                render_prediction_results(st.session_state.inference_results)
                st.markdown("---")
                dl_cols = st.columns(2)
                dl_cols[0].download_button(
                    "Download Predictions (.csv)",
                    data=st.session_state.inference_results.to_csv(
                        index=False).encode("utf-8"),
                    file_name="MLatelier_Predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                if not tp_predict_btn:
                    st.info(
                        "Configure the model and data on the left, then click "
                        "**Predict** to see predictions here."
                    )

    # ── Vision Predict ────────────────────────────────────────────────────────
    with pred_vis_t:
        vp_c1, vp_c2 = st.columns([1, 2], gap="large")

        with vp_c1:
            st.markdown("#### Load Vision Model")

            vis_session_path = None
            _vw   = st.session_state.vision.get("winning_curves") or {}
            _v_mp = _vw.get("exported_model_path")
            if _v_mp and os.path.exists(_v_mp):
                vis_session_path = _v_mp
                st.success(f"Session model: `{os.path.basename(_v_mp)}`")

            use_vis_session = False
            if vis_session_path:
                use_vis_session = st.toggle(
                    "Use model from last training run",
                    value=True, key="vp_use_sess")

            vp_model_file = vp_meta_file = None
            if not use_vis_session:
                vp_model_file = st.file_uploader(
                    "Upload .pt model file", type=["pt"], key="vp_model_up")
                vp_meta_file  = st.file_uploader(
                    "Upload metadata JSON (optional)",
                    type=["json"], key="vp_meta_up")

            st.markdown("#### New Images")
            vp_folder = st.text_input(
                "Image folder path",
                placeholder=r"C:\new_images",
                key="vp_folder",
            )
            st.caption("Or upload individual files:")
            vp_images = st.file_uploader(
                "Upload images",
                type=["jpg", "jpeg", "png", "bmp", "webp"],
                accept_multiple_files=True,
                key="vp_images",
            )

            vp_has_model  = use_vis_session or vp_model_file is not None
            vp_has_images = (vp_folder and os.path.isdir(vp_folder)) or bool(vp_images)

            vp_predict_btn = False
            if vp_has_model and vp_has_images:
                vp_predict_btn = st.button(
                    "Predict",
                    type="primary",
                    use_container_width=True,
                    key="vp_predict",
                )
            else:
                st.button(
                    "Predict",
                    type="primary",
                    use_container_width=True,
                    disabled=True,
                    key="vp_predict_dis",
                )
                if not vp_has_model:
                    st.caption("Load or upload a vision model.")
                elif not vp_has_images:
                    st.caption("Provide a folder path or upload images.")

        with vp_c2:
            if vp_predict_btn:
                _tmp_m = _tmp_meta = None
                try:
                    # Resolve model path + metadata
                    if use_vis_session and vis_session_path:
                        model_path = vis_session_path
                        meta_path  = vis_session_path.replace(".pt", "_metadata.json")
                    else:
                        _tmp_m = tempfile.NamedTemporaryFile(
                            suffix=".pt", delete=False)
                        _tmp_m.write(vp_model_file.read())
                        _tmp_m.flush(); _tmp_m.close()
                        model_path = _tmp_m.name
                        meta_path  = None
                        if vp_meta_file:
                            _tmp_meta = tempfile.NamedTemporaryFile(
                                suffix=".json", delete=False)
                            _tmp_meta.write(vp_meta_file.read())
                            _tmp_meta.flush(); _tmp_meta.close()
                            meta_path = _tmp_meta.name

                    with st.spinner("Loading vision model…"):
                        model, metadata, device = infer.load_vision_model(
                            model_path, meta_path)

                    class_names = metadata.get("class_names", [])
                    if not class_names:
                        class_names = (
                            st.session_state.vision.get("label_classes") or [])
                    if not class_names:
                        st.error("Class names not found. "
                                  "Include a metadata JSON with class_names.")
                        st.stop()

                    st.info(
                        f"Model: {metadata.get('model_name', '?')} | "
                        f"Classes: {class_names} | Device: {device}"
                    )

                    with st.spinner("Running predictions…"):
                        if vp_folder and os.path.isdir(vp_folder):
                            df_vis_preds = infer.predict_vision_folder(
                                model, vp_folder, class_names, device)
                        else:
                            items = [(f.name, f) for f in vp_images]
                            df_vis_preds = infer.predict_vision_batch(
                                model, items, class_names, device)

                    st.session_state.inference_results = df_vis_preds

                except Exception as e:
                    st.error(f"Vision inference failed: {e}")
                finally:
                    for _tf in [_tmp_m, _tmp_meta]:
                        if _tf:
                            try:
                                os.unlink(_tf.name)
                            except Exception:
                                pass

            if st.session_state.inference_results is not None:
                render_prediction_results(st.session_state.inference_results)
                st.download_button(
                    "Download Predictions (.csv)",
                    data=st.session_state.inference_results.to_csv(
                        index=False).encode("utf-8"),
                    file_name="MLatelier_Vision_Predictions.csv",
                    mime="text/csv",
                )
            else:
                if not vp_predict_btn:
                    st.info(
                        "Load a vision model and provide images on the left, "
                        "then click **Predict** to see class predictions here."
                    )


# ══════════════════════════════════════════════════════════════════════════════
# NLP TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_nlp:
    nlp_cfg_col, nlp_res_col = st.columns([1, 2], gap="large")
    nlp_run_btn = False

    with nlp_cfg_col:
        st.markdown("#### NLP Dataset")
        nlp_uploaded = st.file_uploader(
            "Upload CSV or Excel",
            type=_TABULAR_ACCEPT,
            key="nlp_file_upload",
        )
        nlp_path = st.text_input(
            "Or local path",
            placeholder="D:/text_data.csv",
            key="nlp_path_input",
        )

        nlp_df = None
        if not nlp_uploaded and nlp_path and os.path.exists(nlp_path):
            nlp_uploaded = nlp_path
        if nlp_uploaded is not None:
            try:
                _nlp_sheet = _sheet_selector(nlp_uploaded, key="nlp_sheet")
                nlp_df = _read_tabular(nlp_uploaded, sheet_name=_nlp_sheet)
                fname = (nlp_uploaded if isinstance(nlp_uploaded, str)
                         else nlp_uploaded.name)
                st.success(
                    f"{os.path.basename(fname)} — "
                    f"{len(nlp_df):,} rows × {len(nlp_df.columns)} cols"
                )
            except Exception as e:
                st.error(f"Could not read file: {e}")

        # ── Column selection ───────────────────────────────────────────────────
        nlp_text_col = nlp_label_col = None
        if nlp_df is not None:
            st.markdown("#### Columns")
            _nlp_cols = nlp_df.columns.tolist()
            nlp_text_col = st.selectbox(
                "Text column", _nlp_cols, key="nlp_text_col")
            nlp_label_col = st.selectbox(
                "Label column",
                [c for c in _nlp_cols if c != nlp_text_col],
                key="nlp_label_col",
            )
            if nlp_text_col and nlp_label_col:
                _vc = nlp_df[nlp_label_col].value_counts()
                st.info(
                    f"{_vc.nunique()} classes · "
                    f"{nlp_df[nlp_text_col].notna().sum():,} non-null texts"
                )
                if _vc.min() < 0.25 * _vc.max():
                    st.warning("Imbalanced class distribution detected.")

        # ── Track & model selection ────────────────────────────────────────────
        nlp_track = "ml"
        nlp_ml_models: list[str] = []
        nlp_dl_models: list[str] = []
        nlp_st_model = NLP_ST_MODELS[0]
        # HF finetune defaults
        nlp_hf_model_id = HF_PRESET_MODELS[0]
        nlp_hf_epochs = 3
        nlp_hf_batch = 16
        nlp_hf_lr = 2e-5
        nlp_hf_max_len = 128
        nlp_hf_warmup = 0.1
        nlp_hf_freeze = False
        # Preprocessing defaults (ML track)
        nlp_use_sw = True
        nlp_stemming = False
        nlp_ngram = 2
        nlp_max_feat = 5000
        nlp_min_df = 2
        nlp_max_df = 1.0
        nlp_sublinear = None   # auto
        nlp_vec_type = "tfidf"

        if nlp_df is not None and nlp_text_col and nlp_label_col:
            st.markdown("#### Track")
            _track_opts = [
                "ML (TF-IDF + sklearn)",
                "DL (Sentence Transformers)",
                "DL (HuggingFace Fine-tune)",
            ]
            _track_ui = st.radio(
                "Select track",
                _track_opts,
                horizontal=False,
                key="nlp_track",
                label_visibility="collapsed",
            )
            if "ML" in _track_ui:
                nlp_track = "ml"
            elif "Sentence" in _track_ui:
                nlp_track = "dl"
            else:
                nlp_track = "hf"

            # ── ML track ──────────────────────────────────────────────────────
            if nlp_track == "ml":
                st.markdown("#### Models (ML track)")
                _ml_reg = list(get_nlp_ml_models().keys())
                _ml_groups = {
                    "Linear & NB": [
                        "Logistic Regression", "Linear SVC",
                        "SGD Classifier", "Multinomial NB", "Complement NB",
                    ],
                    "Ensemble": [m for m in ["Random Forest", "XGBoost"] if m in _ml_reg],
                }
                for _gname, _gopts in _ml_groups.items():
                    _available_g = [m for m in _gopts if m in _ml_reg]
                    if not _available_g:
                        continue
                    with st.expander(_gname):
                        _sel = st.multiselect(
                            _gname, _available_g, default=[],
                            key=f"nlp_ml_{_gname}",
                            label_visibility="collapsed",
                        )
                        nlp_ml_models.extend(_sel)
                nlp_ml_models = list(dict.fromkeys(nlp_ml_models))

                st.markdown("#### Preprocessing")
                _vec_ui = st.radio(
                    "Vectorizer", ["TF-IDF", "Count"],
                    horizontal=True, key="nlp_vec_type")
                nlp_vec_type = "tfidf" if _vec_ui == "TF-IDF" else "count"
                nlp_use_sw = st.toggle(
                    "Remove stopwords", value=True, key="nlp_sw")
                nlp_stemming = st.toggle(
                    "Porter stemming", value=False, key="nlp_stem",
                    help="Requires NLTK. Slows down preprocessing.",
                )
                nlp_ngram = st.select_slider(
                    "N-gram max", [1, 2, 3], value=2, key="nlp_ngram")
                nlp_max_feat = st.select_slider(
                    "Max features",
                    [1000, 2000, 5000, 10000, 20000], value=5000,
                    key="nlp_maxfeat",
                )
                _mdf_col1, _mdf_col2 = st.columns(2)
                with _mdf_col1:
                    nlp_min_df = st.selectbox(
                        "min_df", [1, 2, 5, 10], index=1, key="nlp_mindf",
                        help="Ignore terms appearing in fewer docs than this.",
                    )
                with _mdf_col2:
                    nlp_max_df = st.select_slider(
                        "max_df", [0.80, 0.90, 0.95, 1.0], value=1.0,
                        key="nlp_maxdf",
                        help="Ignore terms appearing in more than this fraction of docs.",
                    )
                if nlp_vec_type == "tfidf":
                    nlp_sublinear = st.toggle(
                        "Sublinear TF scaling", value=True, key="nlp_sublinear",
                        help="Apply 1+log(tf) instead of raw tf. Disabled automatically for NB models.",
                    )

            # ── DL / Sentence Transformers track ──────────────────────────────
            elif nlp_track == "dl":
                if not _HAS_ST:
                    st.warning(
                        "sentence-transformers not installed. "
                        "Run `pip install sentence-transformers` to enable the DL track."
                    )
                st.markdown("#### Sentence Transformer Model")
                nlp_st_model = st.selectbox(
                    "Embedding model", NLP_ST_MODELS, key="nlp_st_model")
                st.caption(
                    "The text is encoded once with the chosen model; "
                    "a sklearn classifier is then trained on the fixed embeddings."
                )

                st.markdown("#### Classifiers (DL track)")
                _dl_reg = list(get_nlp_dl_classifiers().keys())
                _dl_groups = {
                    "Linear": ["Logistic Regression", "Linear SVC", "SGD Classifier"],
                    "Ensemble": [m for m in ["Random Forest", "XGBoost"] if m in _dl_reg],
                }
                for _gname, _gopts in _dl_groups.items():
                    _available_g = [m for m in _gopts if m in _dl_reg]
                    if not _available_g:
                        continue
                    with st.expander(_gname):
                        _sel = st.multiselect(
                            _gname, _available_g, default=[],
                            key=f"nlp_dl_{_gname}",
                            label_visibility="collapsed",
                        )
                        nlp_dl_models.extend(_sel)
                nlp_dl_models = list(dict.fromkeys(nlp_dl_models))

            # ── HuggingFace Fine-tune track ────────────────────────────────────
            else:
                if not _HAS_TRANSFORMERS:
                    st.warning(
                        "`transformers` not installed. "
                        "Run `pip install transformers` to enable this track."
                    )
                st.markdown("#### HuggingFace Model")
                _hf_preset_opts = HF_PRESET_MODELS + ["Custom…"]
                _hf_preset_sel = st.selectbox(
                    "Preset", _hf_preset_opts, key="nlp_hf_preset",
                    help="Select a preset or choose Custom… to type any HuggingFace model ID.",
                )
                if _hf_preset_sel == "Custom…":
                    nlp_hf_model_id = st.text_input(
                        "HuggingFace model ID",
                        value="distilbert-base-uncased",
                        key="nlp_hf_custom_id",
                        placeholder="e.g. microsoft/deberta-v3-small",
                    )
                else:
                    nlp_hf_model_id = _hf_preset_sel

                st.markdown("#### Training Config")
                _hf_c1, _hf_c2 = st.columns(2)
                with _hf_c1:
                    nlp_hf_epochs = st.select_slider(
                        "Epochs", [1, 2, 3, 5, 10], value=3, key="nlp_hf_epochs")
                    nlp_hf_lr = st.select_slider(
                        "Learning rate",
                        [1e-5, 2e-5, 3e-5, 5e-5],
                        value=2e-5, key="nlp_hf_lr",
                        format_func=lambda v: f"{v:.0e}",
                    )
                    nlp_hf_max_len = st.select_slider(
                        "Max seq length",
                        [64, 128, 256, 512], value=128, key="nlp_hf_maxlen")
                with _hf_c2:
                    nlp_hf_batch = st.select_slider(
                        "Batch size", [4, 8, 16, 32], value=16, key="nlp_hf_batch")
                    nlp_hf_warmup = st.slider(
                        "Warmup ratio", 0.0, 0.3, 0.1, 0.05, key="nlp_hf_warmup")
                    nlp_hf_freeze = st.toggle(
                        "Freeze backbone (head-only)", value=False,
                        key="nlp_hf_freeze",
                        help="Train only the classifier head; much faster but lower quality.",
                    )
                st.caption(
                    f"Model will be downloaded from HuggingFace Hub on first run. "
                    f"GPU recommended for models larger than DistilBERT."
                )
                # HF track uses no sklearn models
                nlp_ml_models = ["__hf_finetune__"]  # sentinel so _active_models is truthy

        # ── Advanced / run settings ────────────────────────────────────────────
        nlp_cv_folds = 5
        nlp_n_iter = 20
        nlp_seed = 42
        nlp_run_opt = True
        nlp_export = True

        if nlp_track == "ml":
            _active_models = nlp_ml_models
        elif nlp_track == "dl":
            _active_models = nlp_dl_models
        else:
            _active_models = nlp_ml_models  # contains sentinel when HF track selected

        if nlp_df is not None and nlp_text_col and nlp_label_col and _active_models:
            with st.expander("Advanced settings"):
                if nlp_track != "hf":
                    nlp_run_opt = st.toggle(
                        "Run Bayesian Optimization", value=True, key="nlp_opt")
                    _cv_ui = st.selectbox(
                        "CV folds", [3, 5, 10], index=1, key="nlp_cv")
                    nlp_cv_folds = int(_cv_ui)
                    nlp_n_iter = st.slider(
                        "BO iterations", 5, 30, 20, key="nlp_niter")
                else:
                    nlp_run_opt = False
                    st.caption("Bayesian Optimization not applicable for HF fine-tuning.")
                nlp_seed = st.number_input(
                    "Random seed", 0, 9999, 42, key="nlp_seed")
                nlp_export = st.toggle(
                    "Export trained model", value=True, key="nlp_export")

        # ── Run button ─────────────────────────────────────────────────────────
        if nlp_df is not None and nlp_text_col and nlp_label_col and _active_models:
            nlp_run_btn = st.button(
                "Run NLP Pipeline",
                type="primary",
                use_container_width=True,
                key="nlp_run",
            )
        elif nlp_df is not None:
            st.button(
                "Run NLP Pipeline",
                type="primary",
                use_container_width=True,
                disabled=True,
                key="nlp_run_dis",
            )
            if not _active_models:
                st.caption("Select at least one model above.")

    # ── Results column ─────────────────────────────────────────────────────────
    with nlp_res_col:
        if nlp_run_btn:
            gc.collect()

            # Prepare texts and labels
            _texts_raw = (nlp_df[nlp_text_col]
                          .fillna("").astype(str).tolist())
            _labels_raw = nlp_df[nlp_label_col].tolist()

            _nlp_status  = st.empty()
            _nlp_prog    = st.progress(0)
            _nlp_live_h  = st.empty()
            _nlp_live_t  = st.empty()
            _nlp_live_rows: list = []

            def _on_nlp_model(name, mean, std, elapsed, folds):
                _nlp_live_rows.append({
                    "Model":       name,
                    "Macro F1":    f"{mean:.4f}" if not np.isnan(mean) else "—",
                    "±Std":        f"{std:.4f}" if not np.isnan(mean) else "—",
                    "Time (s)":    elapsed,
                    "Status":      "✓",
                })
                with _nlp_live_h.container():
                    st.markdown("#### Live Baseline Race")
                with _nlp_live_t.container():
                    df_live = pd.DataFrame(_nlp_live_rows)
                    try:
                        df_live = df_live.sort_values("Macro F1", ascending=False)
                    except Exception:
                        pass
                    st.dataframe(df_live, use_container_width=True, hide_index=True)

            # Phase 1 — Baselines (or HF fine-tune)
            _nlp_status.info("Initialising NLP pipeline…")
            try:
                _nlp_dev = get_nlp_device()
                _dev_str = _nlp_dev.get("type", "cpu")
                _nlp_export_dir = os.path.join(
                    os.path.expanduser("~"), "MLatelier", "models")

                nlp_fold_scores: dict = {}
                nlp_timing: dict = {}
                _nlp_lime: dict = {}
                _nlp_top_feats: dict = {}

                # ── HuggingFace fine-tune path ─────────────────────────────────
                if nlp_track == "hf":
                    _nlp_status.info(
                        f"Fine-tuning {nlp_hf_model_id} "
                        f"({nlp_hf_epochs} epoch(s))…"
                    )
                    (
                        nlp_results_list, nlp_best_acc, nlp_avg_acc,
                        nlp_imp, nlp_pval,
                        nlp_best_params, nlp_class_report, nlp_class_names,
                        nlp_winner_curves,
                    ) = run_transformer_finetune(
                        _texts_raw, _labels_raw,
                        model_id=nlp_hf_model_id,
                        epochs=nlp_hf_epochs,
                        batch_size=nlp_hf_batch,
                        learning_rate=nlp_hf_lr,
                        max_seq_len=nlp_hf_max_len,
                        warmup_ratio=nlp_hf_warmup,
                        freeze_backbone=nlp_hf_freeze,
                        random_state=nlp_seed,
                        export_dir=_nlp_export_dir if nlp_export else None,
                        progress_bar=_nlp_prog,
                        status_text=_nlp_status,
                    )
                    nlp_timing = nlp_winner_curves.get("timing_data", {})

                else:
                    # ── ML / DL (Sentence Transformer) path ───────────────────
                    _nlp_status.info("Running baseline cross-validation…")
                    baseline_results_nlp, nlp_class_names = run_nlp_baseline(
                        _texts_raw, _labels_raw, _active_models,
                        track=nlp_track,
                        use_stopwords=nlp_use_sw,
                        use_stemming=nlp_stemming,
                        ngram_max=nlp_ngram,
                        max_features=nlp_max_feat,
                        min_df=nlp_min_df,
                        max_df=nlp_max_df,
                        sublinear_tf=nlp_sublinear,
                        vectorizer_type=nlp_vec_type,
                        st_model_name=nlp_st_model,
                        device=_dev_str,
                        cv_folds=nlp_cv_folds,
                        random_state=nlp_seed,
                        progress_bar=_nlp_prog,
                        status_text=_nlp_status,
                        model_callback=_on_nlp_model,
                    )
                    _nlp_live_h.empty()
                    _nlp_live_t.empty()

                    nlp_fold_scores = {
                        m: v.get("fold_scores", [])
                        for m, v in baseline_results_nlp.items()
                        if isinstance(v, dict) and v.get("fold_scores")
                    }
                    nlp_timing = {
                        m: v.get("time_s", 0.0)
                        for m, v in baseline_results_nlp.items()
                        if isinstance(v, dict)
                    }

                    # Phase 2 — Optimisation
                    if nlp_run_opt:
                        _nlp_status.info("Running Bayesian Optimisation…")
                        _nlp_prog.progress(0.0)
                        (
                            nlp_results_list, nlp_best_acc, nlp_avg_acc,
                            nlp_imp, nlp_pval,
                            nlp_best_params, nlp_class_report, nlp_class_names,
                            nlp_winner_curves,
                        ) = run_nlp_optimization(
                            _texts_raw, _labels_raw, _active_models,
                            baseline_results_nlp,
                            track=nlp_track,
                            use_stopwords=nlp_use_sw,
                            use_stemming=nlp_stemming,
                            ngram_max=nlp_ngram,
                            max_features=nlp_max_feat,
                            min_df=nlp_min_df,
                            max_df=nlp_max_df,
                            sublinear_tf=nlp_sublinear,
                            vectorizer_type=nlp_vec_type,
                            st_model_name=nlp_st_model,
                            device=_dev_str,
                            cv_folds=nlp_cv_folds,
                            n_iter=nlp_n_iter,
                            random_state=nlp_seed,
                            export_model=nlp_export,
                            export_dir=_nlp_export_dir if nlp_export else None,
                            progress_bar=_nlp_prog,
                            status_text=_nlp_status,
                        )
                    else:
                        _nlp_status.info("Baselines complete.")
                        _valid = [
                            v["mean"] for v in baseline_results_nlp.values()
                            if isinstance(v, dict) and not np.isnan(v.get("mean", float("nan")))
                        ]
                        nlp_best_acc  = max(_valid) if _valid else 0.0
                        nlp_avg_acc   = sum(_valid) / len(_valid) if _valid else 0.0
                        nlp_imp, nlp_pval = 0.0, "N/A"
                        nlp_class_report  = None
                        nlp_winner_curves = {}
                        nlp_best_params   = {"Status": "Optimization skipped"}
                        nlp_results_list  = []
                        for m, bl in baseline_results_nlp.items():
                            _mn = bl.get("mean", float("nan")) if isinstance(bl, dict) else float("nan")
                            _sd = bl.get("std", 0.0) if isinstance(bl, dict) else 0.0
                            _ts = bl.get("time_s", 0.0) if isinstance(bl, dict) else 0.0
                            nlp_results_list.append({
                                "Algorithm": m,
                                "Track": nlp_track.upper(),
                                "CV Macro F1 (mean)": _mn,
                                "CV Macro F1 (±std)": _sd,
                                "Baseline Time (s)": _ts,
                            })

                    # Post-run: LIME + top features (ML track only)
                    _winner_ep = (nlp_winner_curves or {}).get("exported_model_path")
                    if _winner_ep and os.path.exists(_winner_ep) and nlp_track == "ml":
                        try:
                            import joblib as _jl
                            _loaded_pipe = _jl.load(_winner_ep)
                            _nlp_status.info("Computing LIME explanation…")
                            _nlp_lime = compute_nlp_lime(
                                _loaded_pipe, _texts_raw, nlp_class_names,
                                sample_idx=0, num_features=15, num_samples=300)
                            _nlp_top_feats = get_top_tfidf_features(
                                _loaded_pipe, nlp_class_names, n=20)
                        except Exception:
                            pass

                _nlp_prog.empty()
                _nlp_status.empty()

                # Build leaderboard
                nlp_res_df = pd.DataFrame(nlp_results_list)
                _sort_col = next(
                    (c for c in ["Selection CV Macro F1", "CV Macro F1 (mean)"]
                     if c in nlp_res_df.columns),
                    nlp_res_df.columns[-1],
                )
                nlp_res_df = nlp_res_df.sort_values(
                    _sort_col, ascending=False).reset_index(drop=True)

                nlp_best_str = f"{nlp_best_acc * 100:.2f}%"
                nlp_avg_str  = f"{nlp_avg_acc * 100:.2f}%"

                st.session_state.nlp.update({
                    "results_df":    nlp_res_df,
                    "best_acc":      nlp_best_str,
                    "avg_acc":       nlp_avg_str,
                    "imp":           f"+{nlp_imp:.2f}%" if nlp_run_opt else "N/A",
                    "pval":          nlp_pval,
                    "best_params":   nlp_best_params,
                    "class_report":  nlp_class_report,
                    "winning_curves": nlp_winner_curves,
                    "task_type":     "classification",
                    "timing_data":   nlp_timing,
                    "fold_scores":   nlp_fold_scores,
                    "lime_data":     _nlp_lime,
                    "top_features":  _nlp_top_feats,
                    "_texts":        _texts_raw,
                    "_labels":       _labels_raw,
                    "track":         nlp_track,
                })
                st.session_state.last_engine = "nlp"
                st.rerun()

            except Exception as _e:
                _nlp_prog.empty()
                _nlp_status.empty()
                st.error(f"NLP pipeline failed: {_e}")

        # ── Show stored NLP results ────────────────────────────────────────────
        _nlp_st = st.session_state.nlp
        if _nlp_st.get("results_df") is not None:
            if _nlp_st.get("save_path"):
                st.success(f"Experiment saved: `{_nlp_st['save_path']}`")

            st.markdown("### Leaderboard")
            st.dataframe(_nlp_st["results_df"], use_container_width=True)

            # Metric cards
            render_top_metrics(
                _nlp_st.get("best_acc", "—"),
                _nlp_st.get("avg_acc", "—"),
                _nlp_st.get("imp", "—"),
                _nlp_st.get("pval", "—"),
                "Macro F1",
            )

            # Scientific insights sub-tabs
            _nlp_labels, _nlp_renderers = [], []
            _nlp_curves = _nlp_st.get("winning_curves") or {}

            if _nlp_st.get("timing_data"):
                _nlp_labels.append("Training Time")
                _nlp_renderers.append(
                    lambda d=_nlp_st: render_timing_chart(d["timing_data"]))

            if _nlp_st.get("fold_scores"):
                _nlp_labels.append("CV Distribution")
                _nlp_renderers.append(
                    lambda d=_nlp_st: render_cv_boxplot(d["fold_scores"]))

            if _nlp_curves.get("bo_scores"):
                _nlp_labels.append("BO Convergence")
                _nlp_renderers.append(
                    lambda c=_nlp_curves: render_convergence_plot(c["bo_scores"]))

            if _nlp_curves.get("confusion_matrix"):
                _nlp_labels.append("Confusion Matrix")
                _nlp_renderers.append(
                    lambda c=_nlp_curves: render_confusion_matrix(c["confusion_matrix"]))

            if _nlp_st.get("_texts"):
                _nlp_labels.append("Word Cloud")
                _nlp_renderers.append(
                    lambda d=_nlp_st: render_nlp_wordcloud(
                        d["_texts"], d.get("_labels")))

            if _nlp_st.get("_texts"):
                _nlp_labels.append("Text Stats")
                _nlp_renderers.append(
                    lambda d=_nlp_st: render_nlp_text_stats(
                        d["_texts"], d.get("_labels")))

            if _nlp_st.get("lime_data"):
                _nlp_labels.append("LIME")
                _nlp_renderers.append(
                    lambda d=_nlp_st: render_nlp_lime_explanation(d["lime_data"]))

            if _nlp_st.get("top_features"):
                _nlp_labels.append("Top Features")
                _nlp_renderers.append(
                    lambda d=_nlp_st: render_nlp_top_features(d["top_features"]))

            if _nlp_labels:
                st.markdown("---")
                st.markdown("### Scientific Insights")
                _nlp_subtabs = st.tabs(_nlp_labels)
                for _t, _fn in zip(_nlp_subtabs, _nlp_renderers):
                    with _t:
                        _fn()

            if _nlp_st.get("class_report"):
                st.markdown("---")
                st.markdown("### Classification Report")
                render_classification_report(
                    _nlp_st["class_report"],
                    _nlp_st["class_report"].get("confusion_matrix_labels"),
                )

            if _nlp_st.get("best_params"):
                with st.expander("Best Hyperparameters", expanded=False):
                    st.json(_nlp_st["best_params"])

        else:
            if not nlp_run_btn:
                st.info(
                    "Upload a CSV with a text column and a label column, "
                    "select your models, then click **Run NLP Pipeline**."
                )


# ══════════════════════════════════════════════════════════════════════════════
# AI ASSISTANT TAB
# ══════════════════════════════════════════════════════════════════════════════
with tab_ai:
    if not api_key:
        st.info(
            "Enter your Gemini API key in the sidebar to enable the AI assistant. "
            "Run at least one pipeline first so the assistant has results to discuss."
        )
    else:
        _last_eng = st.session_state.last_engine
        _eng_state = st.session_state[_last_eng] if _last_eng else None
        if _last_eng and _eng_state and _eng_state.get("results_df") is not None:
            _eng_label = {"tabular": "Tabular ML", "vision": "Vision DL",
                          "nlp": "NLP"}.get(_last_eng, _last_eng.title())
            st.caption(
                f"Context: **{_eng_label}** pipeline — "
                f"best score **{_eng_state['best_acc']}**"
            )
        render_ai_chat(api_key, eng_state=_eng_state, engine=_last_eng, model=gemini_model)

