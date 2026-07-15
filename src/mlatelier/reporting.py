from __future__ import annotations

import io
import os
from typing import Optional

import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

# ── Shared theme constants ────────────────────────────────────────────────────
_BG_DARK  = "#121214"
_BG_CARD  = "#0A0A0B"
_BORDER   = "#2A2A2D"
_TEXT     = "#E8E8E8"
_MUTED    = "#8C8C91"
_BLUE     = "#5B9AFF"
_GREEN    = "#3FB950"
_PURPLE   = "#8A63FF"
_AMBER    = "#D29630"
_RED      = "#E05252"
_TEAL     = "#17becf"

def _dark_fig(w: float = 7, h: float = 3.6):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_CARD)
    ax.tick_params(colors=_MUTED)
    ax.spines[:].set_color(_BORDER)
    return fig, ax


# ── Top-level metric cards ────────────────────────────────────────────────────

def render_top_metrics(
    best_acc="—", avg_acc="—", improvement="—", pvalue="—", metric_name="Macro F1"
):
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        f'<div class="metric-card cyan"><div class="metric-label">Best {metric_name}</div>'
        f'<div class="metric-value cyan">{best_acc}</div>'
        f'<div class="metric-delta">Highest across all models</div></div>',
        unsafe_allow_html=True,
    )
    m2.markdown(
        f'<div class="metric-card teal"><div class="metric-label">Average {metric_name}</div>'
        f'<div class="metric-value teal">{avg_acc}</div>'
        f'<div class="metric-delta">Mean over all iterations</div></div>',
        unsafe_allow_html=True,
    )
    m3.markdown(
        f'<div class="metric-card indigo"><div class="metric-label">Improvement %</div>'
        f'<div class="metric-value indigo">{improvement}</div>'
        f'<div class="metric-delta">Optimization vs Baseline</div></div>',
        unsafe_allow_html=True,
    )
    m4.markdown(
        f'<div class="metric-card amber"><div class="metric-label">Status Check</div>'
        f'<div class="metric-value amber">{pvalue}</div>'
        f'<div class="metric-delta">Pipeline indicator</div></div>',
        unsafe_allow_html=True,
    )


# ── Data preview ──────────────────────────────────────────────────────────────

def render_data_preview(df: pd.DataFrame):
    """Show dataset head + summary statistics."""
    if df is None:
        return
    st.markdown("#### Sample Rows")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.markdown("#### Numeric Summary")
        desc = numeric_df.describe().round(4)
        st.dataframe(desc, use_container_width=True)

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        st.markdown("#### Categorical Columns")
        for col in cat_cols[:5]:
            vc = df[col].value_counts().head(8)
            fig, ax = plt.subplots(figsize=(5, 2.2))
            fig.patch.set_facecolor(_BG_DARK)
            ax.set_facecolor(_BG_CARD)
            ax.barh(vc.index.astype(str), vc.values, color=_BLUE)
            ax.set_title(f"{col}", color=_TEXT, fontsize=10)
            ax.tick_params(colors=_MUTED, labelsize=8)
            ax.spines[:].set_color(_BORDER)
            ax.set_xlabel("Count", color=_MUTED, fontsize=8)
            ax.invert_yaxis()
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)


# ── Classification report ─────────────────────────────────────────────────────

def render_classification_report(raw_report: dict, label_classes: Optional[list]):
    # ── Accuracy metric card ─────────────────────────────────────────────────
    acc = raw_report.get("accuracy_score") or raw_report.get("accuracy")
    if acc is not None:
        acc_pct = f"{float(acc) * 100:.2f}%"
        macro_f1 = raw_report.get("macro avg", {}).get("f1-score")
        weighted_f1 = raw_report.get("weighted avg", {}).get("f1-score")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy",    acc_pct)
        c2.metric("Macro F1",    f"{macro_f1 * 100:.2f}%" if macro_f1 is not None else "—")
        c3.metric("Weighted F1", f"{weighted_f1 * 100:.2f}%" if weighted_f1 is not None else "—")
        st.markdown("---")

    per_class = {}
    for k, v in raw_report.items():
        if not isinstance(v, dict):
            continue
        label = (label_classes[int(k)]
                 if k.lstrip("-").isdigit() and label_classes is not None
                 else k)
        per_class[label] = v

    plot_classes = [c for c in per_class if c not in ("macro avg", "weighted avg")]
    if plot_classes:
        prec = [per_class[c]["precision"] for c in plot_classes]
        rec  = [per_class[c]["recall"]    for c in plot_classes]
        f1   = [per_class[c]["f1-score"]  for c in plot_classes]
        sup  = [int(per_class[c]["support"]) for c in plot_classes]

        x, w = np.arange(len(plot_classes)), 0.26
        fig, ax = _dark_fig(max(6, len(plot_classes) * 1.4), 3.6)
        ax.bar(x - w, prec, w, label="Precision", color=_BLUE,   zorder=3)
        ax.bar(x,     rec,  w, label="Recall",    color=_GREEN,  zorder=3)
        ax.bar(x + w, f1,   w, label="F1-Score",  color=_PURPLE, zorder=3)
        for i, n in enumerate(sup):
            ax.text(x[i], max(prec[i], rec[i], f1[i]) + 0.06,
                    f"n={n}", ha="center", va="bottom",
                    color=_MUTED, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_classes, color=_TEXT, fontsize=10)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("Score", color=_MUTED, fontsize=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.yaxis.grid(True, color=_BORDER, linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.legend(
            handles=[mpatches.Patch(color=c, label=l) for c, l in [
                (_BLUE, "Precision"), (_GREEN, "Recall"), (_PURPLE, "F1-Score")]],
            loc="upper right", framealpha=0, labelcolor=_TEXT, fontsize=9)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    report_df = pd.DataFrame(per_class).transpose().reset_index()
    report_df.columns = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    report_df["Support"] = report_df["Support"].astype(int)

    def color_score(val):
        if isinstance(val, float):
            if val >= 0.85:
                return f"color: {_GREEN}; font-weight: 600"
            if val >= 0.70:
                return f"color: {_AMBER}; font-weight: 600"
            return f"color: {_RED}; font-weight: 600"
        return ""

    st.dataframe(
        report_df.style
            .map(color_score, subset=["Precision", "Recall", "F1-Score"])
            .format({"Precision": "{:.3f}", "Recall": "{:.3f}",
                     "F1-Score": "{:.3f}", "Support": "{:d}"})
            .set_properties(**{"background-color": _BG_CARD,
                                "color": _TEXT, "border-color": _BORDER}),
        use_container_width=True, hide_index=True,
    )


# ── Confusion matrix ──────────────────────────────────────────────────────────

def render_confusion_matrix(cm_data: dict):
    """Render a confusion matrix heatmap (single-label) or per-label bar chart (multi-label)."""
    if not cm_data or "matrix" not in cm_data:
        return
    matrix = np.array(cm_data["matrix"])
    labels = cm_data.get("labels")

    # ── Multi-label path: shape (n_labels, 2, 2) ─────────────────────────────
    if matrix.ndim == 3:
        n_labels = matrix.shape[0]
        if labels is None:
            labels = [f"Label {i}" for i in range(n_labels)]
        # Each slice: [[TN, FP], [FN, TP]]
        tp = matrix[:, 1, 1].astype(float)
        fp = matrix[:, 0, 1].astype(float)
        fn = matrix[:, 1, 0].astype(float)

        fig, ax = _dark_fig(7.0, max(3.5, n_labels * 0.5))
        y      = np.arange(n_labels)
        bar_h  = 0.26
        ax.barh(y + bar_h, tp, bar_h, color=_GREEN, label="TP", alpha=0.85)
        ax.barh(y,          fp, bar_h, color=_AMBER, label="FP", alpha=0.85)
        ax.barh(y - bar_h,  fn, bar_h, color=_RED,   label="FN", alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, color=_TEXT, fontsize=8)
        ax.set_xlabel("Count", color=_MUTED, fontsize=9)
        ax.set_title("Per-label Confusion  (TP / FP / FN)", color=_TEXT, fontsize=11)
        ax.legend(loc="lower right", framealpha=0.25,
                  labelcolor=_TEXT, facecolor=_BG_CARD, edgecolor=_BORDER, fontsize=8)
        ax.xaxis.grid(True, color=_BORDER, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=_MUTED)
        ax.spines[:].set_color(_BORDER)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
        return

    # ── Standard single-label path: shape (n_classes, n_classes) ─────────────
    n      = matrix.shape[0]
    labels = labels or [str(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(4, n * 1.2), max(3.5, n * 1.0)))
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_BG_CARD)

    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues", aspect="auto")
    plt.colorbar(im, ax=ax).ax.tick_params(colors=_MUTED)

    thresh = matrix.max() / 2.0
    for r in range(n):
        for c in range(n):
            ax.text(c, r, str(matrix[r, c]),
                    ha="center", va="center", fontsize=9,
                    color="white" if matrix[r, c] > thresh else _BG_DARK)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", color=_TEXT, fontsize=8)
    ax.set_yticklabels(labels, color=_TEXT, fontsize=8)
    ax.set_xlabel("Predicted", color=_MUTED, fontsize=9)
    ax.set_ylabel("Actual",    color=_MUTED, fontsize=9)
    ax.set_title("Confusion Matrix", color=_TEXT, fontsize=11)
    ax.spines[:].set_color(_BORDER)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ── ROC curves ────────────────────────────────────────────────────────────────

def render_roc_curves(roc_data: dict):
    """Render binary or multiclass ROC curves."""
    if not roc_data:
        return
    fig, ax = _dark_fig(5.5, 4.0)

    if roc_data.get("type") == "binary":
        fpr = roc_data["fpr"]
        tpr = roc_data["tpr"]
        auc = roc_data["auc"]
        ax.plot(fpr, tpr, color=_BLUE, lw=2,
                label=f"ROC (AUC = {auc:.3f})")
    else:
        colors = [_BLUE, _GREEN, _PURPLE, _AMBER, _RED, _TEAL,
                  "#ff7f0e", "#d62728", "#9467bd", "#8c564b"]
        for i, (cls, vals) in enumerate(roc_data.get("classes", {}).items()):
            c = colors[i % len(colors)]
            ax.plot(vals["fpr"], vals["tpr"], color=c, lw=1.5,
                    label=f"{cls} (AUC={vals['auc']:.3f})")

    ax.plot([0, 1], [0, 1], color=_BORDER, linestyle="--", lw=1)
    ax.set_xlabel("False Positive Rate", color=_MUTED, fontsize=9)
    ax.set_ylabel("True Positive Rate",  color=_MUTED, fontsize=9)
    ax.set_title("ROC Curve", color=_TEXT, fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER,
              labelcolor=_TEXT, fontsize=8,
              loc="lower right")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ── SHAP bar plot ─────────────────────────────────────────────────────────────

def render_shap_plot(shap_values: dict):
    """Bar chart of mean |SHAP| per feature."""
    if not shap_values:
        st.info("SHAP values not available for this model type.")
        return
    df_s = pd.DataFrame({"Feature": list(shap_values.keys()),
                          "Mean |SHAP|": list(shap_values.values())}) \
             .sort_values("Mean |SHAP|", ascending=True)

    fig, ax = _dark_fig(5.5, max(3.0, len(df_s) * 0.32))
    ax.barh(df_s["Feature"], df_s["Mean |SHAP|"], color=_PURPLE, height=0.7)
    ax.set_xlabel("Mean |SHAP value|", color=_MUTED, fontsize=9)
    ax.set_title("SHAP Feature Importance", color=_TEXT, fontsize=11)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.yaxis.grid(False)
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ── Grad-CAM overlay ──────────────────────────────────────────────────────────

def render_gradcam(gradcam_data: dict):
    """Show original image + Grad-CAM overlay side-by-side."""
    if not gradcam_data or "heatmap" not in gradcam_data:
        return
    try:
        import matplotlib.cm as cm

        heatmap = np.array(gradcam_data["heatmap"], dtype=np.float32)
        raw_img = np.array(gradcam_data["image"], dtype=np.float32)

        # raw_img is C×H×W normalized; denorm for display
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std  = np.array([0.229, 0.224, 0.225])[:, None, None]
        img_display = np.clip(raw_img * std + mean, 0, 1).transpose(1, 2, 0)

        # Resize heatmap to image size
        from PIL import Image as PILImage
        h_pil  = PILImage.fromarray(np.uint8(heatmap * 255)).resize(
            (img_display.shape[1], img_display.shape[0]), PILImage.BILINEAR)
        heat_r = np.array(h_pil) / 255.0
        colormap = cm.get_cmap("jet")
        heat_rgb = colormap(heat_r)[..., :3]
        overlay  = np.clip(0.5 * img_display + 0.5 * heat_rgb, 0, 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
        fig.patch.set_facecolor(_BG_DARK)
        for a in (ax1, ax2):
            a.set_facecolor(_BG_CARD)
            a.axis("off")
        ax1.imshow(img_display)
        ax1.set_title("Original", color=_TEXT, fontsize=9)
        ax2.imshow(overlay)
        cn = gradcam_data.get("class_name", str(gradcam_data.get("pred_class", "")))
        ax2.set_title(f"Grad-CAM  |  {cn}", color=_TEXT, fontsize=9)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)
    except Exception as e:
        st.warning(f"Grad-CAM display failed: {e}")


# ── Regression report ─────────────────────────────────────────────────────────

def render_regression_report(raw_report: dict):
    r2  = raw_report.get("R-Squared (R2)", "—")
    mse = raw_report.get("Mean Squared Error (MSE)", "—")
    mae = raw_report.get("Mean Absolute Error (MAE)", "—")

    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<div class="metric-card cyan"><div class="metric-label">R²</div>'
                f'<div class="metric-value cyan">{r2}</div>'
                f'<div class="metric-delta">Higher is better (Max 1.0)</div></div>',
                unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card amber"><div class="metric-label">MAE</div>'
                f'<div class="metric-value amber">{mae}</div>'
                f'<div class="metric-delta">Lower is better</div></div>',
                unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card indigo"><div class="metric-label">MSE</div>'
                f'<div class="metric-value indigo">{mse}</div>'
                f'<div class="metric-delta">Lower is better</div></div>',
                unsafe_allow_html=True)

    if "y_true" in raw_report and "y_pred" in raw_report:
        y_true = np.array(raw_report["y_true"])
        y_pred = np.array(raw_report["y_pred"])
        st.markdown("<br><h4 style='color:var(--text-color);font-weight:600;font-size:1.1rem;'>"
                    "Actual vs. Predicted</h4>", unsafe_allow_html=True)
        fig, ax = _dark_fig(8, 3.8)
        ax.scatter(y_true, y_pred, color=_BLUE, alpha=0.5, edgecolors=_BG_DARK,
                   linewidth=0.5, s=40, label="Predictions")
        mn = min(y_true.min(), y_pred.min())
        mx = max(y_true.max(), y_pred.max())
        ax.plot([mn, mx], [mn, mx], color=_GREEN, linestyle="--", lw=2,
                label="Perfect Fit")
        ax.set_xlabel("Actual",    color=_MUTED, fontsize=10, fontweight="bold")
        ax.set_ylabel("Predicted", color=_MUTED, fontsize=10, fontweight="bold")
        ax.grid(True, color=_BORDER, linewidth=0.6, linestyle=":", zorder=0)
        ax.set_axisbelow(True)
        ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT,
                  loc="upper left", fontsize=9)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ── BO convergence ────────────────────────────────────────────────────────────

def render_convergence_plot(bo_scores: list):
    if not bo_scores:
        return
    st.markdown("#### Bayesian Optimization Convergence")
    st.caption("How the optimizer discovered improvements across successive trials.")
    best_so_far = np.maximum.accumulate(bo_scores)
    trials = range(1, len(bo_scores) + 1)
    fig, ax = _dark_fig(5, 3)
    ax.plot(trials, bo_scores, marker="x", linestyle="", color="#aec7e8",
            alpha=0.7, label="Trial Score")
    ax.plot(trials, best_so_far, marker="o", linestyle="-", color=_BLUE,
            linewidth=2, label="Best So Far")
    ax.set_xlabel("Optimization Trial", color=_MUTED, fontsize=9)
    ax.set_ylabel("Metric Score",       color=_MUTED, fontsize=9)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ── Training curve (vision) ───────────────────────────────────────────────────

def render_training_curve(train_losses: list, val_f1s: list):
    if not train_losses or not val_f1s:
        return
    st.markdown("#### Deep Learning Training Dynamics")
    st.caption("Epoch-by-epoch learning loss and validation generalisation.")
    epochs = range(1, len(train_losses) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    fig.patch.set_facecolor(_BG_DARK)
    for ax in (ax1, ax2):
        ax.set_facecolor(_BG_CARD)
        ax.tick_params(colors=_MUTED)
        ax.spines[:].set_color(_BORDER)
        ax.yaxis.grid(True, color=_BORDER, linewidth=0.6)
        ax.set_axisbelow(True)
    ax1.plot(epochs, train_losses, marker="o", color=_RED, lw=2)
    ax1.set_xlabel("Epoch", color=_MUTED, fontsize=9)
    ax1.set_ylabel("Training Loss", color=_MUTED, fontsize=9)
    ax1.set_title("Model Learning", color=_TEXT, fontsize=10)
    ax2.plot(epochs, val_f1s, marker="o", color=_GREEN, lw=2)
    ax2.set_xlabel("Epoch", color=_MUTED, fontsize=9)
    ax2.set_ylabel("Validation Macro F1", color=_MUTED, fontsize=9)
    ax2.set_title("Generalisation", color=_TEXT, fontsize=10)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ── Feature importance ────────────────────────────────────────────────────────

def render_feature_importance(importances_dict: dict):
    if not importances_dict:
        st.info("Feature importance not available for this algorithm.")
        return
    st.markdown("#### Top Feature Importances")
    st.caption("Data columns that most influenced the model's decision.")
    df_imp = pd.DataFrame({"Feature":    list(importances_dict.keys()),
                            "Importance": list(importances_dict.values())}) \
               .sort_values("Importance", ascending=True)
    fig, ax = _dark_fig(5.5, max(3.0, len(df_imp) * 0.32))
    ax.barh(df_imp["Feature"], df_imp["Importance"], color=_TEAL, height=0.7)
    ax.set_xlabel("Relative Importance / Weight", color=_MUTED, fontsize=9)
    ax.set_title("Drivers of Model Prediction",   color=_TEXT,  fontsize=11)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


# ── Export buttons ────────────────────────────────────────────────────────────

def render_export_buttons(results_df: Optional[pd.DataFrame],
                           html_content: str,
                           model_bytes: Optional[bytes] = None,
                           model_filename: str = "MLatelier_model.joblib"):
    st.markdown("### Export Results")
    st.caption("Download your optimization artifacts for publication and reproducibility.")
    st.markdown("""<style>
    div.stDownloadButton > button { color: #000 !important; }
    div.stDownloadButton > button * { color: #000 !important; }
    div.stDownloadButton > button:hover { color: #FFF !important; }
    div.stDownloadButton > button:hover * { color: #FFF !important; }
    </style>""", unsafe_allow_html=True)

    cols = st.columns(3 if model_bytes else 2)

    if results_df is not None:
        cols[0].download_button(
            label="Download Leaderboard (.csv)",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="MLatelier_Leaderboard.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if html_content:
        cols[1].download_button(
            label="Download Full Report (.html)",
            data=html_content.encode("utf-8"),
            file_name="MLatelier_Report.html",
            mime="text/html",
            use_container_width=True,
        )

    if model_bytes and len(cols) > 2:
        cols[2].download_button(
            label=f"Download Best Model ({model_filename.split('.')[-1].upper()})",
            data=model_bytes,
            file_name=model_filename,
            mime="application/octet-stream",
            use_container_width=True,
        )


def render_all_models_download(all_exported_paths: dict):
    """Show a download button for every trained model, 3 per row."""
    if not all_exported_paths:
        return
    st.markdown("#### Download Individual Models")
    st.caption(
        "Every model trained during this run is available for download "
        "with its hyperparameter metadata (.json sidecar)."
    )
    names  = list(all_exported_paths.keys())
    n_cols = 3
    for i in range(0, len(names), n_cols):
        row_names = names[i : i + n_cols]
        row_cols  = st.columns(n_cols)
        for col, name in zip(row_cols, row_names):
            path = all_exported_paths[name]
            if path and os.path.exists(path):
                with open(path, "rb") as fh:
                    data = fh.read()
                fname = os.path.basename(path)
                col.download_button(
                    label=f"{name}",
                    data=data,
                    file_name=fname,
                    mime="application/octet-stream",
                    use_container_width=True,
                    key=f"dl_model_{name}",
                )
            else:
                col.button(name, disabled=True, use_container_width=True,
                           key=f"dl_model_{name}_na")


# ── AI assistant ──────────────────────────────────────────────────────────────

_MLatelier_APP_GUIDE = """
=== MLatelier Application Structure & Workflow Guide ===

MLatelier is a 6-tab Streamlit dashboard for no-code ML prototyping.
Every suggestion you give MUST reference a specific control, tab, expander, or button
inside this app — never give abstract ML advice that the user cannot act on immediately.

─── TAB 1 · TABULAR ML ────────────────────────────────────────────────────────
Purpose : Train and compare multiple ML models on a tabular CSV/Excel dataset.
Workflow:
  1. Upload a CSV/Excel file (or paste a URL / local path).
  2. Set the Target Y column and Feature X columns.
  3. Open model group expanders and tick the models to run:
       • Boosting & Ensemble : Random Forest, XGBoost, Gradient Boosting,
                               Extra Trees, Hist Gradient Boosting,
                               LightGBM, CatBoost, AdaBoost
       • Linear & SVM        : Logistic Regression, Ridge Classifier,
                               SGD Classifier, Passive Aggressive,
                               Linear SVC, Support Vector Machine
       • Tree & Distance     : Decision Tree, KNN
       • Neural & Probabilistic : MLP Classifier, Gaussian NB,
                               Multinomial NB, Bernoulli NB, LDA, QDA
     (Regression equivalents exist for each group.)
  4. Data Handling section:
       • "Auto-balance imbalanced classes" toggle → enables SMOTE oversampling.
       • "Scaling strategy" expander → choose Auto (default), RobustScaler,
         StandardScaler, or MinMaxScaler.
  5. Advanced settings expander:
       • "Run Bayesian Optimization" toggle → ON = BO + baseline; OFF = baseline only.
       • Test split slider : 10 % → 40 % (default 20 %).
       • CV folds dropdown : 3, 5, 10, or auto.
       • BO iterations slider : 5 → 30 per model (default 10).
         Increasing to 20–30 meaningfully improves hyperparameter quality.
       • Random seed (default 42).
       • "Export trained model" toggle → saves .joblib to disk.
  6. Click "Run Tabular Pipeline".
Results shown: leaderboard table, CV distribution boxplot, BO convergence plot,
  feature importance chart, SHAP waterfall (top features), LIME local explanation,
  partial dependence plots, learning curve, training time bar chart,
  calibration curve (classification), per-class report, downloadable model.

Key actionable next steps when scores are low:
  • Add more models from the model expanders (especially XGBoost, LightGBM, CatBoost).
  • Enable SMOTE if classes are imbalanced.
  • Try a different scaler via the Scaling strategy expander.
  • Increase BO iterations to 20 or 30 in Advanced settings.
  • Run Ablation Study (Tab 3) to isolate which preprocessing step hurts/helps most.
  • Reduce test split to 15 % for small datasets (< 500 rows).

─── TAB 2 · VISION DL ─────────────────────────────────────────────────────────
Purpose : Fine-tune pretrained CNN / Transformer architectures on an image dataset.
Dataset : Folder of subfolders (one subfolder = one class label), or a ZIP file.
Architecture groups to pick from:
  • Classic CNNs       : ResNet18, ResNet50, VGG16
  • Efficient Networks : EfficientNet_B0, EfficientNet_V2_S, MobileNet_v3, DenseNet121
  • ConvNeXt           : ConvNeXt_T, ConvNeXt_S
  • Vision Transformers: ViT_B_16, Swin_T, Swin_S
Training controls (shown once architectures are selected):
  • "Run Bayesian Optimization" toggle.
  • Training epochs slider (1 → 50, default 10).
  • Batch size select slider (8 / 16 / 32 / 64 / 128, default 32).
  • Data augmentation toggle (random flip, crop, colour jitter).
  • Fine-tune strategy radio: "Head only" (faster) vs "Last block + head" (more accurate).
  • LR scheduler : cosine, step, plateau, none.
  • Early stopping patience slider.
  • SMOTE-equivalent class balancing toggle.
Results: accuracy leaderboard, training/validation curves, Grad-CAM heatmap,
  confusion matrix, per-class report, downloadable model.

Key next steps:
  • Switch from "Head only" to "Last block + head" for higher accuracy.
  • Enable data augmentation if the dataset is small (< 1000 images per class).
  • Try EfficientNet_B0 or DenseNet121 for a good accuracy/speed trade-off.
  • Increase epochs to 20–30 if validation loss is still declining.
  • Reduce batch size to 16 if you get out-of-memory errors.

─── TAB 3 · ABLATION STUDY ────────────────────────────────────────────────────
Purpose : Systematically vary one pipeline component at a time on tabular data
          to measure each component's true contribution.
Controls:
  • Upload the same CSV used in Tabular ML.
  • Choose Target, Features, and the single model to ablate.
  • "What to Vary" checkboxes (tick any combination):
       ☐ Imbalance handling (SMOTE on vs off)
       ☐ Feature scaling (auto / robust / standard / minmax — all 4 compared)
       ☐ Test split (10 % / 20 % / 30 %)
       ☐ Cross-validation folds (3 / 5 / 10)
  • Baseline configuration: set the reference values to compare against.
  • Click "Run Ablation Study".
Results: bar chart comparing all variants, summary table, winner highlighted.

When to use: after getting a baseline score in Tabular ML, run ablation to see whether
  SMOTE, a different scaler, or a different CV fold count changes results significantly.
  Use the winning configuration back in Tabular ML for the final run.

─── TAB 4 · PREDICT ───────────────────────────────────────────────────────────
Purpose : Run inference on new, unseen data using a trained model.
Sub-tabs:
  • Tabular Predict:
      - Toggle "Use model from last training run" to use the session model automatically.
      - Or upload a saved .joblib model file.
      - Upload a new CSV (no target column needed).
      - Click "Run Tabular Prediction" → downloads a CSV with Prediction + Confidence columns.
  • Vision Predict:
      - Toggle "Use model from last Vision DL run" for the session model.
      - Or upload a .pt / .pth PyTorch checkpoint.
      - Upload individual images or a folder.
      - Click "Run Vision Prediction".

─── TAB 5 · NLP ────────────────────────────────────────────────────────────────
Purpose : Text classification using TF-IDF + sklearn models or Sentence Transformers.
Dataset : CSV with a text column and a label column.
Track selection (radio button):
  • ML track  : TF-IDF vectorisation → sklearn classifiers
      Models: Logistic Regression, Linear SVC, SGD Classifier,
              Multinomial NB, Complement NB  (Linear & NB group),
              Random Forest, XGBoost, LightGBM, Extra Trees (Boosting & Tree group)
  • DL track  : Sentence Transformers (BERT-based embeddings) → sklearn head
      Pretrained models: all-MiniLM-L6-v2, paraphrase-MiniLM-L6-v2,
                         all-mpnet-base-v2, paraphrase-multilingual-MiniLM-L12-v2
NLP settings (shown once track + models selected):
  • SMOTE toggle (classification imbalance handling).
  • TF-IDF max features slider (ML track only).
  • Test split slider.
  • "Run Bayesian Optimization" toggle.
  • BO iterations slider.
Results: leaderboard, per-class report, confusion matrix, top TF-IDF features,
  LIME text explanation, BO convergence.

Key next steps:
  • If ML track scores are low, switch to DL track — sentence transformers give much
    richer text representations, especially for short texts.
  • Try all-mpnet-base-v2 for the best DL accuracy (slower but more powerful).
  • Enable SMOTE if one class dominates the label column.
  • Increase TF-IDF max features (ML track) if vocabulary is large.

─── TAB 6 · AI ASSISTANT ──────────────────────────────────────────────────────
Purpose : Context-aware Gemini AI chat grounded in the latest experiment results.
  Requires a Gemini API key in the sidebar.
  Auto-generates an insight card when new results arrive.
  Use quick-question chips for instant answers.

─── RECOMMENDED WORKFLOW ───────────────────────────────────────────────────────
For tabular data:
  1. Tabular ML → baseline run with a few quick models (Random Forest, Logistic Regression).
  2. Look at the feature importance chart — remove irrelevant features if needed.
  3. Ablation Study → find best scaler and SMOTE setting.
  4. Tabular ML → full run with the best models + settings + 25–30 BO iterations.
  5. Predict tab → run inference on new data.

For image data:
  1. Vision DL → baseline run with ResNet18 (fast).
  2. Vision DL → full run adding EfficientNet_B0 / Swin_T + data augmentation.
  3. Predict tab → Vision Predict for new images.

For text data:
  1. NLP → ML track first (fast, good baseline).
  2. NLP → DL track with all-MiniLM-L6-v2 for better accuracy.
  3. Compare both leaderboards; pick the best model.
  4. Predict tab (if tabular predict supports it) or export the model.

IMPORTANT: Always frame your suggestions as specific actions the user can take
in MLatelier RIGHT NOW — name the exact tab, expander, toggle, slider, or button.
Never say "consider using XGBoost" — say "open the Boosting & Ensemble expander
in the Tabular ML tab and tick XGBoost, then click Run Tabular Pipeline."
"""


def _build_experiment_context(eng_state: dict, engine: str) -> str:
    """Build a structured experiment context string from session state for the system prompt."""
    import io
    lines: list[str] = [f"=== MLatelier Experiment Context  [{engine.upper()} engine] ==="]

    task = eng_state.get("task_type") or "classification"
    lines.append(f"Task type        : {task}")
    lines.append(f"Best score       : {eng_state.get('best_acc', '—')}")
    avg = eng_state.get("avg_acc")
    if avg and avg != "—":
        lines.append(f"Average score    : {avg}")
    imp = eng_state.get("imp")
    if imp and imp != "—":
        lines.append(f"BO improvement   : {imp}")
    pval = eng_state.get("pval")
    if pval and pval != "—":
        lines.append(f"p-value          : {pval}")

    features = eng_state.get("_features_x")
    target   = eng_state.get("_target_y")
    if features:
        n_feat = len(features) if hasattr(features, "__len__") else "?"
        feat_list = list(features)[:15]
        suffix = " …" if len(list(features)) > 15 else ""
        lines.append(f"Features ({n_feat})    : {', '.join(feat_list)}{suffix}")
    if target:
        lines.append(f"Target column    : {target}")

    label_classes = eng_state.get("label_classes") or eng_state.get("class_names")
    if label_classes:
        lines.append(f"Class labels     : {list(label_classes)}")

    df = eng_state.get("_df")
    if df is not None:
        try:
            lines.append(f"Dataset shape    : {df.shape[0]:,} rows × {df.shape[1]} cols")
        except Exception:
            pass

    # Model leaderboard
    res_df = eng_state.get("results_df")
    if res_df is not None and not res_df.empty:
        try:
            top = res_df.head(8)
            lines.append(f"\n--- Model Leaderboard (top {len(top)}) ---")
            buf = io.StringIO()
            top.to_string(buf, index=False)
            lines.append(buf.getvalue())
        except Exception:
            pass

    # Best hyperparameters
    best_params = eng_state.get("best_params")
    if best_params and isinstance(best_params, dict):
        valid = {k: v for k, v in best_params.items() if k not in ("Status", "Error")}
        if valid:
            lines.append("\n--- Best Hyperparameters (winning model) ---")
            for k, v in valid.items():
                lines.append(f"  {k}: {v}")

    # Per-class performance
    class_report = eng_state.get("class_report")
    if class_report and isinstance(class_report, dict):
        try:
            lines.append("\n--- Per-class Performance ---")
            for cls, metrics in class_report.items():
                if isinstance(metrics, dict) and "f1-score" in metrics:
                    f1   = metrics.get("f1-score", 0)
                    prec = metrics.get("precision", 0)
                    rec  = metrics.get("recall", 0)
                    sup  = metrics.get("support", "")
                    sup_s = f", n={int(sup)}" if sup != "" else ""
                    lines.append(
                        f"  {cls}: precision={prec:.3f}, recall={rec:.3f},"
                        f" f1={f1:.3f}{sup_s}"
                    )
        except Exception:
            pass

    # Winning model name
    wc = eng_state.get("winning_curves")
    if wc and isinstance(wc, dict):
        wname = wc.get("winner_name") or wc.get("model_name")
        if wname:
            lines.append(f"\nWinning model    : {wname}")

    # Timing
    timing = eng_state.get("timing_data")
    if timing and isinstance(timing, dict):
        lines.append("\n--- Training Times (seconds) ---")
        for m, t in timing.items():
            try:
                lines.append(f"  {m}: {t:.1f}s")
            except Exception:
                pass

    # Interpretability availability
    interp_bits = []
    if eng_state.get("shap_waterfall"):
        interp_bits.append("SHAP waterfall")
    if eng_state.get("pdp_data"):
        interp_bits.append("partial dependence plots")
    if eng_state.get("lime_data"):
        interp_bits.append("LIME explanations")
    if interp_bits:
        lines.append(f"Interpretability : {', '.join(interp_bits)} available")

    # Top features (NLP TF-IDF or tabular importance)
    top_feats = eng_state.get("top_features")
    if top_feats:
        try:
            rows = list(top_feats)[:10]
            if rows and isinstance(rows[0], (list, tuple)) and len(rows[0]) == 2:
                lines.append("\n--- Top Predictive Features ---")
                for feat, score in rows:
                    lines.append(f"  {feat}: {score:.4f}")
        except Exception:
            pass

    # NLP-specific extras
    if engine == "nlp":
        track = eng_state.get("track")
        if track:
            lines.append(f"NLP track        : {track}")

    return "\n".join(lines)


def _gemini_call(client, contents, system_prompt: str, model: str) -> str:
    """Call Gemini and return response text; raises on any error."""
    from google.genai import types as _gt
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=_gt.GenerateContentConfig(system_instruction=system_prompt),
    )
    return resp.text


def render_ai_chat(
    api_key: str,
    eng_state: "dict | None" = None,
    engine: "str | None" = None,
    model: str = "gemini-1.5-flash",
):
    """
    Agentic AI assistant powered by Gemini 2.5 Flash.

    * Injects full experiment context (leaderboard, best params, class report,
      features, interpretability data) as the Gemini system instruction.
    * Auto-generates a proactive insight card whenever new results are detected.
    * Shows context-aware quick-question chips above the chat.
    * Maintains proper multi-turn history passed to the Gemini API.
    """
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        try:
            from utils import show_ai_dependency_error
            show_ai_dependency_error()
        except Exception:
            st.error("Install the google-genai package: `pip install google-genai`")
        return

    # ── Session-state keys ────────────────────────────────────────────────────
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []
    if "_ai_ctx_hash" not in st.session_state:
        st.session_state._ai_ctx_hash = ""
    if "_ai_chip" not in st.session_state:
        st.session_state._ai_chip = None
    # Model preference: start with 2.5-flash; fall back to 1.5-flash on 403

    # ── Build system context ──────────────────────────────────────────────────
    has_results = (
        eng_state is not None and eng_state.get("results_df") is not None
    )
    exp_context = (
        _build_experiment_context(eng_state, engine or "tabular")
        if has_results
        else "No experiment has been run yet."
    )
    system_prompt = (
        "You are an expert ML research assistant embedded in MLatelier, "
        "a no-code machine learning prototyping platform for researchers.\n"
        "Your job: help users understand their results and tell them the single "
        "best next action to take — always as a specific step inside MLatelier.\n\n"
        "Rules:\n"
        "- Every suggestion MUST name the exact tab, expander, toggle, slider, or "
        "button the user should click. No abstract advice.\n"
        "- Reference the actual numbers from the experiment context.\n"
        "- Prioritise the highest-impact action first. Don't list 10 things; pick 3.\n"
        "- Use markdown: **bold** key numbers, bullet lists, `code` for model names.\n"
        "- When you don't have experiment results yet, still give app-specific "
        "guidance based on the user's question.\n\n"
        f"{_MLatelier_APP_GUIDE}\n\n"
        f"{exp_context}"
    )

    # ── Detect result change and auto-generate proactive insight ──────────────
    import hashlib
    ctx_hash = hashlib.md5(exp_context.encode()).hexdigest()[:12]

    if has_results and st.session_state._ai_ctx_hash != ctx_hash:
        st.session_state._ai_ctx_hash = ctx_hash
        st.session_state.ai_messages = []          # fresh chat for new experiment
        with st.spinner("Analysing experiment results…"):
            try:
                client = genai.Client(api_key=api_key)
                init_q = (
                    "I just finished a machine learning experiment in MLatelier. "
                    "Please give me:\n"
                    "1. A 2-sentence summary of the key results.\n"
                    "2. The single most important observation from the numbers.\n"
                    "3. Exactly three next steps I should take — each must name "
                    "the specific MLatelier tab, expander, toggle, or slider to use. "
                    "Order them by expected impact, highest first."
                )
                _reply = _gemini_call(client, init_q, system_prompt, model)
                st.session_state.ai_messages.append(
                    {"role": "assistant", "content": _reply}
                )
            except Exception as _e:
                st.session_state.ai_messages.append(
                    {"role": "assistant",
                     "content": (
                         "Results loaded! Ask me anything — what the scores mean, "
                         "how to improve, or what to try next.\n\n"
                         f"*(Auto-insight unavailable: {_e})*"
                     )}
                )

    elif not st.session_state.ai_messages:
        st.session_state.ai_messages.append(
            {"role": "assistant",
             "content": (
                 "Hello! I am your MLatelier AI assistant. "
                 "Run a pipeline first and I will automatically analyse your results. "
                 "You can also ask me general ML questions right now."
             )}
        )

    # ── Context-aware suggestion chips ────────────────────────────────────────
    if has_results:
        wc = eng_state.get("winning_curves") or {}
        _res_df = eng_state.get("results_df")
        best_model = (
            wc.get("winner_name") or wc.get("model_name")
            or (str(_res_df.iloc[0].get("Algorithm", "the best model"))
                if _res_df is not None and not _res_df.empty else "the best model")
        )
        task = eng_state.get("task_type") or "classification"

        chips: list[tuple[str, str]] = [
            (
                "Why did it win?",
                f"Explain why **{best_model}** outperformed the other models, "
                "referencing the specific scores in the leaderboard.",
            ),
            (
                "How to improve?",
                "What are the three most impactful things I can do to improve these results? "
                "Be specific — name models, preprocessing steps, and tuning settings to try.",
            ),
        ]
        if eng_state.get("shap_waterfall") or eng_state.get("pdp_data"):
            chips.append((
                "Key features?",
                "Which features are most influential and what do they suggest about the dataset? "
                "Use the interpretability data in the context.",
            ))
        if task == "classification":
            chips.append((
                "Class balance?",
                "How might class imbalance be affecting my results, "
                "and what should I do about it in MLatelier?",
            ))
        elif task == "regression":
            chips.append((
                "Model fit?",
                "What do the regression metrics (R², RMSE, MAE) suggest about "
                "model fit and potential data issues?",
            ))
        if engine == "nlp":
            chips.append((
                "Better NLP?",
                "What NLP preprocessing, feature engineering, or model changes "
                "would most improve text classification accuracy?",
            ))
        chips.append((
            "Report summary",
            "Write one concise paragraph I can include in a research report "
            "summarising these experiment results objectively.",
        ))

        st.markdown(
            f'<p style="color:{_MUTED};font-size:.76rem;font-weight:700;'
            f'text-transform:uppercase;letter-spacing:.06em;margin:0 0 6px 0;">'
            f'Quick questions</p>',
            unsafe_allow_html=True,
        )
        max_chips = min(len(chips), 4)
        chip_cols = st.columns(max_chips)
        for _i, (_label, _prompt) in enumerate(chips[:max_chips]):
            with chip_cols[_i]:
                if st.button(_label, key=f"_ai_chip_btn_{_i}", use_container_width=True):
                    st.session_state._ai_chip = _prompt
                    st.rerun()
        st.divider()

    # ── Chat history display ──────────────────────────────────────────────────
    with st.container(height=400):
        for _msg in st.session_state.ai_messages:
            if _msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(_msg["content"])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown(_msg["content"])

    # ── Handle chip click or typed input ─────────────────────────────────────
    _chip = st.session_state._ai_chip
    if _chip:
        st.session_state._ai_chip = None
        user_input: "str | None" = _chip
    else:
        user_input = st.chat_input(
            "Ask about your results, what to try next, or any ML question…"
        )

    if user_input:
        st.session_state.ai_messages.append({"role": "user", "content": user_input})

        # Build Gemini-compatible history (role must be "user" or "model")
        history = []
        for _m in st.session_state.ai_messages[:-1]:
            _g_role = "model" if _m["role"] == "assistant" else "user"
            history.append({"role": _g_role, "parts": [{"text": _m["content"]}]})
        history.append({"role": "user", "parts": [{"text": user_input}]})

        with st.spinner("Thinking…"):
            try:
                client = genai.Client(api_key=api_key)
                reply = _gemini_call(client, history, system_prompt, model)
            except Exception as _e:
                _el = str(_e).lower()
                if any(k in _el for k in
                       ("api_key_invalid", "api key not valid", "invalid_argument",
                        "401", "unauthenticated")):
                    reply = ("**Invalid API key** — check your Gemini API key "
                             "in the sidebar.")
                elif any(k in _el for k in ("quota", "429", "resource_exhausted")):
                    reply = ("**Quota limit reached.** Please wait a moment "
                             "and try again.")
                elif any(k in _el for k in
                         ("403", "permission_denied", "permission denied",
                          "denied access", "not found", "404")):
                    reply = (
                        f"**Model `{model}` is not available with your API key (403).**\n\n"
                        "Try a different model from the **Gemini Model** dropdown in the sidebar "
                        "— `gemini-1.5-flash` works on most free-tier keys.\n\n"
                        "If all models fail, make sure you're using a key from "
                        "[Google AI Studio](https://aistudio.google.com/apikey)."
                    )
                elif any(k in _el for k in ("connection", "timeout", "network")):
                    reply = ("**Cannot reach Gemini servers.** Check your "
                             "internet connection.")
                else:
                    reply = f"**AI error:** {_e}"

        st.session_state.ai_messages.append({"role": "assistant", "content": reply})
        st.rerun()


# ── Training time bar chart ───────────────────────────────────────────────────

def render_timing_chart(timing_data: dict):
    """Horizontal bar chart of computation time per model."""
    if not timing_data:
        return
    models = list(timing_data.keys())
    times  = [timing_data[m] for m in models]
    pairs  = sorted(zip(times, models), reverse=True)
    times, models = zip(*pairs) if pairs else ([], [])

    fig, ax = _dark_fig(5.5, max(3.0, len(models) * 0.42))
    max_t   = max(times) if times else 1
    colors  = [_BLUE if t == max_t else "#4A6FA5" for t in times]
    bars    = ax.barh(models, times, color=colors, height=0.6)

    for bar, t in zip(bars, times):
        ax.text(bar.get_width() + max_t * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{t:.1f}s", va="center", ha="left", color=_TEXT, fontsize=8)

    ax.set_xlabel("Training Time (seconds)", color=_MUTED, fontsize=9)
    ax.set_title("Computation Time per Model", color=_TEXT, fontsize=11)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.set_xlim(0, max_t * 1.25)
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.5)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.caption("Faster is not always better — complex models spend more time but may generalise better.")


# ── CV fold box / violin plot ─────────────────────────────────────────────────

def render_cv_boxplot(fold_scores_per_model: dict):
    """Box plot of per-fold CV scores — shows variance across models."""
    if not fold_scores_per_model:
        return
    data   = {m: s for m, s in fold_scores_per_model.items() if s}
    if not data:
        st.info("No fold-level scores available.")
        return

    labels = list(data.keys())
    values = [data[m] for m in labels]

    fig, ax = _dark_fig(max(5.5, len(labels) * 1.6), 3.8)
    bp = ax.boxplot(
        values, patch_artist=True,
        medianprops=dict(color=_GREEN, linewidth=2),
        whiskerprops=dict(color=_MUTED, linewidth=1.2),
        capprops=dict(color=_MUTED, linewidth=1.2),
        flierprops=dict(marker="o", color=_AMBER, markersize=4, alpha=0.7),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(_BG_CARD)
        patch.set_edgecolor(_BLUE)
        patch.set_linewidth(1.5)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=35, ha="right", color=_TEXT, fontsize=8)
    ax.set_ylabel("CV Score per Fold", color=_MUTED, fontsize=9)
    ax.set_title("Cross-Validation Score Distribution", color=_TEXT, fontsize=11)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.caption("Green line = median. Box = 25th–75th percentile. Wider box = less stable model.")


# ── Calibration / reliability diagram ────────────────────────────────────────

def render_calibration_curve(calib_data: dict):
    """Reliability diagram — a perfectly calibrated model lies on the diagonal."""
    if not calib_data:
        st.info("Calibration data not available.")
        return

    fig, ax = _dark_fig(4.5, 4.2)
    ax.plot([0, 1], [0, 1], color=_BORDER, linestyle="--", lw=1.5,
            label="Perfect Calibration")

    if calib_data.get("type") == "binary":
        fop   = calib_data["fop"]
        mpv   = calib_data["mpv"]
        brier = calib_data.get("brier_score", "?")
        names = calib_data.get("class_names", ["neg", "pos"])
        ax.plot(mpv, fop, marker="o", color=_BLUE, lw=2, ms=6,
                label=f"{names[-1]} (Brier={brier:.3f})")
    else:
        colors = [_BLUE, _GREEN, _PURPLE, _AMBER, _RED, _TEAL,
                  "#ff7f0e", "#d62728", "#9467bd"]
        for i, (cn, vals) in enumerate(calib_data.get("classes", {}).items()):
            c     = colors[i % len(colors)]
            brier = vals.get("brier_score", "?")
            ax.plot(vals["mpv"], vals["fop"], marker="o", color=c, lw=1.5, ms=4,
                    label=f"{cn} (Brier={brier:.3f})")

    ax.set_xlabel("Mean Predicted Probability", color=_MUTED, fontsize=9)
    ax.set_ylabel("Fraction of Positives",      color=_MUTED, fontsize=9)
    ax.set_title("Reliability Diagram (Calibration)", color=_TEXT, fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)
    st.caption("A well-calibrated model follows the dashed diagonal. "
               "Brier score ↓ is better (0 = perfect). "
               "Curves above diagonal = under-confident; below = over-confident.")


# ── Correlation heatmap ───────────────────────────────────────────────────────

def render_correlation_heatmap(df: pd.DataFrame, feature_cols: list):
    """Seaborn correlation heatmap of numeric features."""
    try:
        import seaborn as sns
    except ImportError:
        st.info("Install seaborn (`pip install seaborn`) for the correlation heatmap.")
        return

    try:
        numeric = df[feature_cols].select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            st.info("Need ≥2 numeric features for a correlation heatmap.")
            return

        if numeric.shape[1] > 20:
            numeric = numeric.iloc[:, :20]
            st.caption("Showing first 20 numeric features.")

        corr = numeric.corr()
        n    = corr.shape[0]

        fig, ax = plt.subplots(figsize=(max(5, n * 0.85), max(4, n * 0.75)))
        fig.patch.set_facecolor(_BG_DARK)
        ax.set_facecolor(_BG_CARD)

        sns.heatmap(
            corr, ax=ax,
            annot=(n <= 14), fmt=".2f", cmap="coolwarm",
            linewidths=0.3, linecolor=_BG_DARK,
            vmin=-1, vmax=1, center=0,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 7, "color": _TEXT} if n <= 14 else {},
        )
        ax.set_title("Feature Correlation Matrix", color=_TEXT, fontsize=11, pad=10)
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Values near ±1 indicate strong (multi)collinearity. "
                   "Highly correlated pairs may be redundant.")
    except Exception as e:
        st.warning(f"Correlation heatmap failed: {e}")


# ── Data quality report ───────────────────────────────────────────────────────

def render_data_quality_report(df: pd.DataFrame, feature_cols: list,
                                target_col: Optional[str] = None):
    """Show missing values, outliers, and feature skewness."""
    if df is None:
        return

    all_cols  = list(feature_cols) + ([target_col] if target_col else [])
    sub       = df[[c for c in all_cols if c in df.columns]]

    missing     = sub.isnull().sum()
    missing_pct = (missing / len(sub) * 100).round(1)
    miss_df     = pd.DataFrame({"Missing Count": missing,
                                  "Missing %": missing_pct})
    miss_df     = miss_df[miss_df["Missing Count"] > 0]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Missing Values")
        if miss_df.empty:
            st.success("No missing values detected.")
        else:
            st.warning(f"{len(miss_df)} column(s) have missing values.")
            st.dataframe(miss_df, use_container_width=True)

    with c2:
        st.markdown("##### Outliers (IQR method)")
        numeric_df = sub.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            outlier_rows = []
            for col in numeric_df.columns:
                s = numeric_df[col].dropna()
                if len(s) < 4:
                    continue
                q1, q3 = s.quantile(0.25), s.quantile(0.75)
                iqr    = q3 - q1
                if iqr > 0:
                    n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
                    if n_out > 0:
                        outlier_rows.append(
                            {"Feature": col, "Outliers": n_out,
                             "% of rows": round(n_out / len(s) * 100, 1)})
            if outlier_rows:
                st.dataframe(pd.DataFrame(outlier_rows),
                              use_container_width=True, hide_index=True)
            else:
                st.success("No significant outliers detected.")
        else:
            st.info("No numeric features to check.")

    # Skewness bar
    numeric_df = sub.select_dtypes(include=[np.number])
    if not numeric_df.empty and numeric_df.shape[1] >= 2:
        from scipy.stats import skew as _skew
        sk_vals = {}
        for col in numeric_df.columns:
            s = numeric_df[col].dropna()
            if len(s) >= 3:
                try:
                    sk_vals[col] = round(float(_skew(s)), 3)
                except Exception:
                    pass
        if sk_vals:
            st.markdown("##### Feature Skewness")
            sk_df = (pd.DataFrame({"Feature": list(sk_vals.keys()),
                                    "Skewness": list(sk_vals.values())})
                       .sort_values("Skewness", key=abs, ascending=False)
                       .head(15))
            fig, ax = _dark_fig(5.5, max(2.5, len(sk_df) * 0.34))
            colors  = [_RED if abs(v) > 2 else (_AMBER if abs(v) > 1 else _GREEN)
                       for v in sk_df["Skewness"]]
            ax.barh(sk_df["Feature"], sk_df["Skewness"], color=colors, height=0.65)
            ax.axvline(0,  color=_BORDER, linewidth=1.0)
            ax.axvline(1,  color=_AMBER, linewidth=0.8, linestyle="--", alpha=0.6)
            ax.axvline(-1, color=_AMBER, linewidth=0.8, linestyle="--", alpha=0.6)
            ax.set_xlabel("Skewness", color=_MUTED, fontsize=9)
            ax.set_title("Feature Skewness  (|>1| skewed · |>2| highly skewed)",
                          color=_TEXT, fontsize=10)
            ax.tick_params(colors=_MUTED, labelsize=7)
            ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
            ax.set_axisbelow(True)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)


# ── Ablation study chart ──────────────────────────────────────────────────────

def render_ablation_chart(results: list):
    """Bar chart + table comparing ablation pipeline configurations."""
    if not results:
        st.info("No ablation results.")
        return

    df_a = pd.DataFrame(results)
    if "Config" not in df_a or "CV Score" not in df_a:
        st.dataframe(df_a, use_container_width=True)
        return

    colors = [_GREEN if ("Baseline" in str(c) or str(c).startswith("★"))
              else _BLUE for c in df_a["Config"]]

    fig, ax = _dark_fig(max(5.5, len(df_a) * 1.3), 3.5)
    bars = ax.bar(df_a["Config"], df_a["CV Score"].fillna(0),
                   color=colors, width=0.6, zorder=3)
    for bar, val in zip(bars, df_a["CV Score"].fillna(0)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.003,
                f"{val:.4f}", ha="center", va="bottom", color=_TEXT, fontsize=7.5)

    ax.set_xlabel("Pipeline Configuration", color=_MUTED, fontsize=9)
    ax.set_ylabel("CV Score", color=_MUTED, fontsize=9)
    ax.set_title("Ablation Study — Component Impact", color=_TEXT, fontsize=11)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.set_xticklabels(df_a["Config"], rotation=30, ha="right", fontsize=8)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.dataframe(df_a, use_container_width=True, hide_index=True)


# ── Inference prediction results ──────────────────────────────────────────────

def render_prediction_results(df_preds: pd.DataFrame):
    """Show inference results: predictions + confidence + distribution."""
    if df_preds is None or df_preds.empty:
        st.info("No predictions to display.")
        return

    st.markdown("#### Predictions")
    pred_cols = ["Prediction", "Confidence"]
    prob_cols = [c for c in df_preds.columns if c.startswith("P(")]
    disp_cols = [c for c in pred_cols + prob_cols if c in df_preds.columns]
    disp_df   = df_preds[disp_cols] if disp_cols else df_preds

    def _conf_color(val):
        if isinstance(val, (int, float)) and pd.notnull(val):
            if val >= 0.90:
                return f"color: {_GREEN}; font-weight: 600"
            if val >= 0.70:
                return f"color: {_AMBER}; font-weight: 600"
            return f"color: {_RED}; font-weight: 600"
        return ""

    if "Confidence" in disp_df.columns:
        styled = disp_df.style.map(_conf_color, subset=["Confidence"])
    else:
        styled = disp_df.style
    st.dataframe(styled, use_container_width=True, hide_index=True)

    if "Prediction" in df_preds.columns:
        vc = df_preds["Prediction"].value_counts()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Class Distribution")
            fig, ax = _dark_fig(4, max(2.2, len(vc) * 0.65))
            ax.barh(vc.index.astype(str), vc.values, color=_BLUE, height=0.6)
            for i, v in enumerate(vc.values):
                ax.text(v + 0.1, i, str(v), va="center", color=_TEXT, fontsize=8)
            ax.set_xlabel("Count", color=_MUTED, fontsize=9)
            ax.tick_params(colors=_MUTED, labelsize=8)
            ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
            ax.set_axisbelow(True)
            plt.tight_layout(pad=0.4)
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
        with c2:
            st.markdown("#### Confidence Summary")
            if "Confidence" in df_preds.columns:
                conf = df_preds["Confidence"].dropna()
                st.metric("Mean Confidence",    f"{conf.mean():.3f}")
                st.metric("Min Confidence",     f"{conf.min():.3f}")
                low_n = int((conf < 0.7).sum())
                if low_n:
                    st.warning(f"{low_n} prediction(s) below 70% confidence — review carefully.")
                else:
                    st.success("All predictions ≥70% confidence.")

    # ── Confidence histogram ───────────────────────────────────────────────────
    if "Confidence" in df_preds.columns and len(df_preds) >= 5:
        render_confidence_histogram(df_preds)

    # ── Uncertain rows table ───────────────────────────────────────────────────
    if "Confidence" in df_preds.columns:
        low_conf = df_preds[df_preds["Confidence"] < 0.70].copy()
        if not low_conf.empty:
            with st.expander(f"Uncertain Predictions ({len(low_conf)} rows < 70% confidence)",
                             expanded=False):
                st.dataframe(low_conf.style.map(
                    lambda v: f"color: {_RED}; font-weight:600"
                    if isinstance(v, float) and v < 0.70 else "",
                    subset=["Confidence"],
                ), use_container_width=True, hide_index=True)


# ── Regression residual diagnostics ──────────────────────────────────────────

def render_regression_residuals(report_dict: dict):
    """4-panel residual diagnostic: Residuals vs Fitted, Q-Q, histogram, Scale-Location."""
    y_true = report_dict.get("y_true")
    y_pred = report_dict.get("y_pred")
    if y_true is None or y_pred is None:
        return
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    residuals = y_true - y_pred

    st.markdown("#### Residual Diagnostics")
    st.caption(
        "Healthy residuals should be randomly scattered around zero (no pattern), "
        "normally distributed, and variance-stable across fitted values."
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    fig.patch.set_facecolor(_BG_DARK)
    for ax in axes.flat:
        ax.set_facecolor(_BG_CARD)
        ax.tick_params(colors=_MUTED, labelsize=8)
        ax.spines[:].set_color(_BORDER)

    # Panel 1 — Residuals vs Fitted
    ax = axes[0, 0]
    ax.scatter(y_pred, residuals, color=_BLUE, alpha=0.45, s=18,
               edgecolors=_BG_DARK, linewidth=0.4)
    ax.axhline(0, color=_GREEN, linewidth=1.4, linestyle="--")
    try:
        from numpy.polynomial.polynomial import polyfit as _pf
        coef = _pf(y_pred, residuals, 2)
        xs   = np.linspace(y_pred.min(), y_pred.max(), 200)
        ax.plot(xs, coef[0] + coef[1]*xs + coef[2]*xs**2,
                color=_AMBER, linewidth=1.2, linestyle=":", label="trend")
    except Exception:
        pass
    ax.set_xlabel("Fitted Values", color=_MUTED, fontsize=9)
    ax.set_ylabel("Residuals",     color=_MUTED, fontsize=9)
    ax.set_title("Residuals vs Fitted", color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)

    # Panel 2 — Q-Q plot
    ax = axes[0, 1]
    try:
        from scipy import stats as _stats
        (osm, osr), (slope, intercept, _) = _stats.probplot(residuals, fit=True)
        ax.scatter(osm, osr, color=_PURPLE, alpha=0.5, s=16, edgecolors=_BG_DARK, linewidth=0.3)
        ref_x = np.array([osm[0], osm[-1]])
        ax.plot(ref_x, slope * ref_x + intercept, color=_GREEN, linewidth=1.4, linestyle="--")
    except Exception:
        ax.text(0.5, 0.5, "scipy not available", transform=ax.transAxes,
                color=_MUTED, ha="center", fontsize=9)
    ax.set_xlabel("Theoretical Quantiles", color=_MUTED, fontsize=9)
    ax.set_ylabel("Sample Quantiles",      color=_MUTED, fontsize=9)
    ax.set_title("Normal Q-Q Plot", color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)

    # Panel 3 — Residual histogram
    ax = axes[1, 0]
    ax.hist(residuals, bins=min(40, max(10, len(residuals) // 20)),
            color=_BLUE, alpha=0.75, edgecolor=_BG_DARK, linewidth=0.4)
    try:
        from scipy.stats import norm as _norm
        mu, sigma = residuals.mean(), residuals.std()
        xs  = np.linspace(residuals.min(), residuals.max(), 200)
        pdf = _norm.pdf(xs, mu, sigma)
        ax2 = ax.twinx()
        ax2.plot(xs, pdf, color=_GREEN, linewidth=1.6, label="Normal PDF")
        ax2.set_facecolor(_BG_CARD)
        ax2.tick_params(colors=_MUTED, labelsize=7)
        ax2.spines[:].set_color(_BORDER)
        ax2.set_ylabel("Density", color=_MUTED, fontsize=8)
    except Exception:
        pass
    ax.axvline(0, color=_AMBER, linewidth=1.2, linestyle="--")
    ax.set_xlabel("Residual Value", color=_MUTED, fontsize=9)
    ax.set_ylabel("Frequency",      color=_MUTED, fontsize=9)
    ax.set_title("Residual Distribution", color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)

    # Panel 4 — Scale-Location (√|residuals| vs fitted)
    ax = axes[1, 1]
    sqrt_abs_res = np.sqrt(np.abs(residuals))
    ax.scatter(y_pred, sqrt_abs_res, color=_TEAL, alpha=0.45, s=18,
               edgecolors=_BG_DARK, linewidth=0.4)
    try:
        coef2 = np.polyfit(y_pred, sqrt_abs_res, 1)
        xs2   = np.linspace(y_pred.min(), y_pred.max(), 200)
        ax.plot(xs2, np.polyval(coef2, xs2), color=_AMBER, linewidth=1.2, linestyle=":")
    except Exception:
        pass
    ax.set_xlabel("Fitted Values",          color=_MUTED, fontsize=9)
    ax.set_ylabel("√|Residuals|",           color=_MUTED, fontsize=9)
    ax.set_title("Scale-Location",          color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)

    fig.suptitle("Residual Diagnostic Plots", color=_TEXT, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(pad=0.8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # Normality test summary
    try:
        from scipy.stats import shapiro, kstest, normaltest
        n   = len(residuals)
        col1, col2, col3 = st.columns(3)
        if n <= 5000:
            stat_sw, p_sw = shapiro(residuals[:5000])
            col1.metric("Shapiro-Wilk p",
                        f"{p_sw:.4f}",
                        delta="Normal ✓" if p_sw > 0.05 else "Non-normal ⚠")
        stat_ks, p_ks = kstest(
            (residuals - residuals.mean()) / (residuals.std() or 1), "norm")
        col2.metric("KS Test p",
                    f"{p_ks:.4f}",
                    delta="Normal ✓" if p_ks > 0.05 else "Non-normal ⚠")
        stat_dp, p_dp = normaltest(residuals)
        col3.metric("D'Agostino p",
                    f"{p_dp:.4f}",
                    delta="Normal ✓" if p_dp > 0.05 else "Non-normal ⚠")
    except Exception:
        pass


# ── EDA: feature distributions (histograms + KDE) ────────────────────────────

def render_feature_distributions(
    df: pd.DataFrame,
    feature_cols: list,
    max_cols: int = 20,
):
    """Grid of per-column histogram + KDE overlays."""
    if df is None or not feature_cols:
        return
    numeric = [c for c in feature_cols if c in df.columns
               and pd.api.types.is_numeric_dtype(df[c])][:max_cols]
    if not numeric:
        st.info("No numeric features to plot.")
        return

    n_feat   = len(numeric)
    n_cols_g = min(3, n_feat)
    n_rows_g = (n_feat + n_cols_g - 1) // n_cols_g
    fig, axes = plt.subplots(n_rows_g, n_cols_g,
                              figsize=(n_cols_g * 3.8, n_rows_g * 2.8))
    fig.patch.set_facecolor(_BG_DARK)
    axes = np.array(axes).flatten() if n_feat > 1 else [axes]

    colors_cycle = [_BLUE, _PURPLE, _TEAL, _GREEN, _AMBER, _RED]
    for i, (col, ax) in enumerate(zip(numeric, axes)):
        ax.set_facecolor(_BG_CARD)
        ax.tick_params(colors=_MUTED, labelsize=7)
        ax.spines[:].set_color(_BORDER)
        s = df[col].dropna()
        if len(s) == 0:
            ax.text(0.5, 0.5, "all NaN", transform=ax.transAxes,
                    color=_MUTED, ha="center", fontsize=8)
            ax.set_title(col, color=_TEXT, fontsize=8)
            continue
        c = colors_cycle[i % len(colors_cycle)]
        n_bins = min(40, max(10, len(s) // 15))
        ax.hist(s, bins=n_bins, color=c, alpha=0.65,
                edgecolor=_BG_DARK, linewidth=0.3, density=True)
        try:
            from scipy.stats import gaussian_kde as _kde
            kde  = _kde(s)
            xs   = np.linspace(s.min(), s.max(), 300)
            ax.plot(xs, kde(xs), color=_TEXT, linewidth=1.2)
        except Exception:
            pass
        ax.axvline(s.mean(),   color=_GREEN, linewidth=1.0, linestyle="--",
                   label=f"μ={s.mean():.2f}")
        ax.axvline(s.median(), color=_AMBER, linewidth=1.0, linestyle=":",
                   label=f"med={s.median():.2f}")
        ax.set_title(col, color=_TEXT, fontsize=8, fontweight="bold")
        ax.yaxis.grid(True, color=_BORDER, linewidth=0.3)
        ax.set_axisbelow(True)

    for ax in axes[n_feat:]:
        ax.set_visible(False)

    plt.tight_layout(pad=0.6)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── EDA: feature box plots ────────────────────────────────────────────────────

def render_feature_boxplots(
    df: pd.DataFrame,
    feature_cols: list,
    max_cols: int = 20,
):
    """Side-by-side box plots showing spread and outliers per feature."""
    if df is None or not feature_cols:
        return
    numeric = [c for c in feature_cols if c in df.columns
               and pd.api.types.is_numeric_dtype(df[c])][:max_cols]
    if not numeric:
        st.info("No numeric features to plot.")
        return

    data_list = [df[c].dropna().values for c in numeric]
    n = len(numeric)

    fig_w = max(6, n * 0.9)
    fig, ax = _dark_fig(fig_w, 4.5)
    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        medianprops=dict(color=_GREEN, linewidth=1.6),
        whiskerprops=dict(color=_BORDER, linewidth=0.9),
        capprops=dict(color=_MUTED, linewidth=0.9),
        flierprops=dict(marker="o", color=_AMBER, markersize=3,
                         alpha=0.5, linestyle="none"),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(_BLUE)
        patch.set_alpha(0.45)
        patch.set_edgecolor(_BLUE)

    ax.set_xticks(range(1, n + 1))
    ax.set_xticklabels(numeric, rotation=45, ha="right", fontsize=7, color=_MUTED)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.set_title("Feature Box Plots (IQR spreads + outliers)",
                  color=_TEXT, fontsize=10, fontweight="bold")
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── EDA: target distribution ──────────────────────────────────────────────────

def render_target_distribution(
    df: pd.DataFrame,
    target_col: str,
    task_type: str = "classification",
):
    """Bar chart (classification) or histogram + KDE (regression) for target column."""
    if df is None or target_col not in df.columns:
        return
    s = df[target_col].dropna()

    st.markdown(f"##### Target Distribution: `{target_col}`")
    if task_type == "regression":
        fig, ax = _dark_fig(7, 3.2)
        n_bins = min(50, max(10, len(s) // 20))
        ax.hist(s, bins=n_bins, color=_BLUE, alpha=0.65,
                edgecolor=_BG_DARK, linewidth=0.3, density=True)
        try:
            from scipy.stats import gaussian_kde as _kde
            kde = _kde(s)
            xs  = np.linspace(s.min(), s.max(), 300)
            ax.plot(xs, kde(xs), color=_GREEN, linewidth=1.6, label="KDE")
        except Exception:
            pass
        ax.axvline(s.mean(),   color=_AMBER, linewidth=1.2, linestyle="--",
                   label=f"mean={s.mean():.2f}")
        ax.axvline(s.median(), color=_RED,   linewidth=1.2, linestyle=":",
                   label=f"median={s.median():.2f}")
        ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
        ax.set_xlabel(target_col, color=_MUTED, fontsize=9)
        ax.set_ylabel("Density",  color=_MUTED, fontsize=9)
    else:
        vc = s.astype(str).value_counts()
        colors = [_BLUE, _PURPLE, _GREEN, _AMBER, _RED, _TEAL]
        fig, ax = _dark_fig(max(5, len(vc) * 0.8), 3.2)
        bars = ax.bar(range(len(vc)), vc.values,
                       color=[colors[i % len(colors)] for i in range(len(vc))],
                       alpha=0.75, edgecolor=_BG_DARK, linewidth=0.4, width=0.6)
        ax.set_xticks(range(len(vc)))
        ax.set_xticklabels(vc.index, rotation=30, ha="right", fontsize=8, color=_MUTED)
        for bar, val in zip(bars, vc.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    str(val), ha="center", va="bottom", color=_TEXT, fontsize=8,
                    fontweight="bold")
        ax.set_ylabel("Count",   color=_MUTED, fontsize=9)
        ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)
        ax.set_axisbelow(True)
    ax.set_title(f"Target: {target_col}", color=_TEXT, fontsize=10, fontweight="bold")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── EDA: statistical summary ──────────────────────────────────────────────────

def render_statistical_summary(df: pd.DataFrame, feature_cols: list):
    """Extended stats table: mean, median, std, IQR, min, max, skewness, kurtosis."""
    if df is None or not feature_cols:
        return
    numeric = [c for c in feature_cols if c in df.columns
               and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric:
        st.info("No numeric features.")
        return

    try:
        from scipy.stats import skew as _skew, kurtosis as _kurt
    except ImportError:
        _skew = _kurt = None

    rows = []
    for col in numeric:
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        row = {
            "Feature":  col,
            "Count":    len(s),
            "Missing":  int(df[col].isna().sum()),
            "Mean":     round(float(s.mean()), 4),
            "Median":   round(float(s.median()), 4),
            "Std":      round(float(s.std()), 4),
            "IQR":      round(float(q3 - q1), 4),
            "Min":      round(float(s.min()), 4),
            "Max":      round(float(s.max()), 4),
        }
        if _skew:
            try:
                row["Skewness"] = round(float(_skew(s)), 4)
                row["Kurtosis"] = round(float(_kurt(s)), 4)
            except Exception:
                pass
        rows.append(row)

    if rows:
        stat_df = pd.DataFrame(rows).set_index("Feature")
        st.dataframe(
            stat_df.style.background_gradient(
                cmap="Blues", subset=["Std"], axis=0),
            use_container_width=True,
        )


# ── EDA: missing value heatmap ────────────────────────────────────────────────

def render_missing_heatmap(df: pd.DataFrame, feature_cols: list):
    """Binary heatmap showing where missing values occur across rows/columns."""
    if df is None or not feature_cols:
        return
    cols = [c for c in feature_cols if c in df.columns]
    if not cols:
        return

    sub = df[cols].isnull()
    if not sub.any().any():
        st.success("No missing values — heatmap skipped.")
        return

    # Limit rows for performance
    if len(sub) > 500:
        sub = sub.sample(500, random_state=42)

    fig_h = max(2.5, len(cols) * 0.28)
    fig, ax = _dark_fig(min(10, len(sub.columns) * 0.5 + 2), fig_h)
    ax.imshow(sub.T.values.astype(int), aspect="auto",
              cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
    ax.set_yticks(range(len(cols)))
    ax.set_yticklabels(cols, fontsize=7, color=_MUTED)
    ax.set_xlabel("Row index (sample)", color=_MUTED, fontsize=8)
    ax.set_title("Missing Value Pattern (blue = missing)",
                  color=_TEXT, fontsize=9, fontweight="bold")
    plt.tight_layout(pad=0.4)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Vision: class distribution ────────────────────────────────────────────────

def render_vision_class_distribution(class_counts: dict):
    """Horizontal bar chart of image counts per class."""
    if not class_counts:
        return
    labels = list(class_counts.keys())
    values = [class_counts[k] for k in labels]
    total  = sum(values)

    fig, ax = _dark_fig(6, max(2.5, len(labels) * 0.55))
    colors = [_BLUE, _PURPLE, _GREEN, _AMBER, _RED, _TEAL]
    bars   = ax.barh(labels, values,
                     color=[colors[i % len(colors)] for i in range(len(labels))],
                     height=0.6, alpha=0.80)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + total * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val}  ({val/total*100:.1f}%)",
                va="center", color=_TEXT, fontsize=8)
    ax.set_xlabel("Image Count", color=_MUTED, fontsize=9)
    ax.set_title(f"Class Distribution  (total: {total:,})",
                  color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    imbalance = max(values) / (min(values) or 1)
    if imbalance > 3:
        st.warning(
            f"Class imbalance ratio {imbalance:.1f}× detected. "
            "Enable 'Handle class imbalance' in the Vision settings."
        )


# ── Vision: sample image grid ─────────────────────────────────────────────────

def render_vision_sample_grid(samples: dict, n_per_class: int = 4):
    """Show a grid of sample images: one row per class, n_per_class columns."""
    if not samples:
        return
    try:
        from PIL import Image as PILImage
    except ImportError:
        st.info("Pillow not available — skipping image grid.")
        return

    st.markdown("##### Sample Images per Class")
    for class_name, paths in samples.items():
        paths = paths[:n_per_class]
        st.caption(f"**{class_name}** ({len(paths)} samples shown)")
        cols = st.columns(len(paths))
        for col, path in zip(cols, paths):
            try:
                img = PILImage.open(path).convert("RGB")
                img.thumbnail((160, 160))
                col.image(img, use_container_width=True)
            except Exception:
                col.markdown(f"*Error*")


# ── Partial Dependence Plots ──────────────────────────────────────────────────

def render_pdp(pdp_data: dict):
    """Line plots of partial dependence for top features."""
    if not pdp_data:
        return
    features = list(pdp_data.keys())
    n = len(features)
    if n == 0:
        return

    st.markdown("#### Partial Dependence Plots")
    st.caption(
        "Shows how the model's average prediction changes as each feature varies, "
        "holding all other features at their mean values."
    )

    n_cols_g = min(3, n)
    n_rows_g = (n + n_cols_g - 1) // n_cols_g
    fig, axes = plt.subplots(n_rows_g, n_cols_g,
                              figsize=(n_cols_g * 3.8, n_rows_g * 2.8))
    fig.patch.set_facecolor(_BG_DARK)
    axes = np.array(axes).flatten() if n > 1 else [axes]

    colors = [_BLUE, _PURPLE, _GREEN, _AMBER, _TEAL, _RED]
    for i, (feat, (xs, ys)) in enumerate(pdp_data.items()):
        ax = axes[i]
        ax.set_facecolor(_BG_CARD)
        ax.tick_params(colors=_MUTED, labelsize=7)
        ax.spines[:].set_color(_BORDER)
        c = colors[i % len(colors)]
        ax.plot(xs, ys, color=c, linewidth=1.8)
        ax.fill_between(xs, ys, alpha=0.12, color=c)
        ax.axhline(np.mean(ys), color=_MUTED, linewidth=0.8, linestyle="--")
        ax.set_xlabel(feat,          color=_MUTED, fontsize=8)
        ax.set_ylabel("Avg. Prediction", color=_MUTED, fontsize=8)
        ax.set_title(feat,           color=_TEXT,  fontsize=9, fontweight="bold")
        ax.xaxis.grid(True, color=_BORDER, linewidth=0.3)
        ax.yaxis.grid(True, color=_BORDER, linewidth=0.3)
        ax.set_axisbelow(True)

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.tight_layout(pad=0.7)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── SHAP waterfall (single sample) ───────────────────────────────────────────

def render_shap_waterfall(waterfall_data: dict):
    """Horizontal waterfall bar for one sample's SHAP feature contributions."""
    if not waterfall_data:
        return
    features   = waterfall_data.get("features", [])
    values     = waterfall_data.get("shap_values", [])
    base_value = waterfall_data.get("base_value", 0.0)
    if not features or not values:
        return

    st.markdown("#### SHAP Waterfall — Single Sample Explanation")
    st.caption(
        "Shows how each feature pushes the prediction above or below the baseline "
        f"(baseline = {base_value:.4f})."
    )

    # Flatten any nested list/array values (can appear for multi-class SHAP)
    def _scalar(v):
        if isinstance(v, (list, np.ndarray)):
            arr = np.array(v).ravel()
            return float(np.mean(np.abs(arr)))
        return float(v)

    # Sort by absolute contribution
    pairs  = sorted(zip(features, values), key=lambda x: abs(_scalar(x[1])), reverse=True)[:15]
    feats  = [p[0] for p in pairs]
    vals   = [_scalar(p[1]) for p in pairs]

    fig, ax = _dark_fig(7, max(3, len(feats) * 0.42))
    colors  = [_GREEN if v >= 0 else _RED for v in vals]
    ax.barh(range(len(feats)), vals, color=colors, height=0.65, alpha=0.80,
            edgecolor=_BG_DARK, linewidth=0.4)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8, color=_MUTED)
    ax.axvline(0, color=_BORDER, linewidth=1.0)
    ax.set_xlabel("SHAP Value (feature contribution)", color=_MUTED, fontsize=9)
    ax.set_title(f"Waterfall  (base = {base_value:.4f})",
                  color=_TEXT, fontsize=10, fontweight="bold")
    for i, v in enumerate(vals):
        ax.text(v + (0.002 if v >= 0 else -0.002), i,
                f"{v:+.4f}", va="center",
                ha="left" if v >= 0 else "right",
                color=_TEXT, fontsize=7)
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── LIME explanation ──────────────────────────────────────────────────────────

def render_lime_explanation(lime_data: dict):
    """Horizontal bar showing LIME feature weights for one sample."""
    if not lime_data:
        return
    features = lime_data.get("features", [])
    weights  = lime_data.get("weights", [])
    label    = lime_data.get("label", "class")
    if not features or not weights:
        return

    st.markdown(f"#### LIME Explanation — Prediction: `{label}`")
    st.caption(
        "LIME fits a local linear model around this sample to explain why "
        "the model made this specific prediction."
    )
    pairs  = sorted(zip(features, weights), key=lambda x: abs(x[1]), reverse=True)[:15]
    feats  = [p[0] for p in pairs]
    vals   = [p[1] for p in pairs]

    fig, ax = _dark_fig(7, max(3, len(feats) * 0.42))
    colors  = [_GREEN if v >= 0 else _RED for v in vals]
    ax.barh(range(len(feats)), vals, color=colors, height=0.65, alpha=0.80,
            edgecolor=_BG_DARK, linewidth=0.4)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=8, color=_MUTED)
    ax.axvline(0, color=_BORDER, linewidth=1.0)
    ax.set_xlabel("LIME Weight",  color=_MUTED, fontsize=9)
    ax.set_title("LIME Local Explanation", color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Learning curve ────────────────────────────────────────────────────────────

def render_learning_curve(lc_data: dict):
    """Train size vs. CV score: shows if more data would help."""
    if not lc_data:
        return
    train_sizes = lc_data.get("train_sizes", [])
    train_means = lc_data.get("train_means", [])
    train_stds  = lc_data.get("train_stds",  [])
    val_means   = lc_data.get("val_means",   [])
    val_stds    = lc_data.get("val_stds",    [])
    metric_name = lc_data.get("metric_name", "Score")
    if not train_sizes:
        return

    ts = np.array(train_sizes)
    tm = np.array(train_means)
    ts_ = np.array(train_stds)
    vm  = np.array(val_means)
    vs  = np.array(val_stds)

    st.markdown("#### Learning Curve")
    st.caption(
        "If validation score keeps rising with more data, you could benefit from a "
        "larger dataset. A plateau means model capacity is the bottleneck."
    )
    fig, ax = _dark_fig(7, 3.6)
    ax.plot(ts, tm, color=_BLUE,   linewidth=1.8, label="Training score")
    ax.fill_between(ts, tm - ts_, tm + ts_, alpha=0.14, color=_BLUE)
    ax.plot(ts, vm, color=_GREEN,  linewidth=1.8, label="CV score")
    ax.fill_between(ts, vm - vs,  vm + vs,  alpha=0.14, color=_GREEN)
    ax.set_xlabel("Training samples", color=_MUTED, fontsize=9)
    ax.set_ylabel(metric_name,        color=_MUTED, fontsize=9)
    ax.set_title("Learning Curve",    color=_TEXT, fontsize=10, fontweight="bold")
    ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=9)
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Feature ablation chart ────────────────────────────────────────────────────

def render_feature_ablation_chart(ablation_results: list):
    """Bar chart of CV score delta when each feature is dropped."""
    if not ablation_results:
        return
    df = pd.DataFrame(ablation_results)
    if df.empty or "Feature Dropped" not in df.columns:
        return

    df = df.sort_values("Delta (%)", ascending=True)
    st.markdown("#### Feature Ablation")
    st.caption(
        "How much the CV score drops when each feature is removed individually. "
        "Large drops = high importance. Near-zero or positive = possibly noise."
    )

    colors = [_RED if v < -1 else (_AMBER if v < 0 else _GREEN)
              for v in df["Delta (%)"]]
    fig, ax = _dark_fig(7, max(3, len(df) * 0.45))
    ax.barh(df["Feature Dropped"], df["Delta (%)"],
            color=colors, height=0.65, alpha=0.80,
            edgecolor=_BG_DARK, linewidth=0.4)
    ax.axvline(0, color=_BORDER, linewidth=1.0)
    ax.set_xlabel("Score Delta (%)",     color=_MUTED, fontsize=9)
    ax.set_title("Feature Ablation — Drop-One Impact",
                  color=_TEXT, fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Confidence histogram ──────────────────────────────────────────────────────

def render_confidence_histogram(df_preds: pd.DataFrame):
    """Histogram of prediction confidence values with zone annotations."""
    if "Confidence" not in df_preds.columns:
        return
    conf = df_preds["Confidence"].dropna().values
    if len(conf) == 0:
        return

    st.markdown("#### Confidence Distribution")
    fig, ax = _dark_fig(7, 3.0)
    n_bins = min(30, max(10, len(conf) // 10))
    ax.hist(conf, bins=n_bins, color=_BLUE, alpha=0.75,
            edgecolor=_BG_DARK, linewidth=0.3)

    ax.axvspan(0.0, 0.70, alpha=0.08, color=_RED,   label="Low (<70%)")
    ax.axvspan(0.70, 0.90, alpha=0.08, color=_AMBER, label="Medium (70-90%)")
    ax.axvspan(0.90, 1.01, alpha=0.08, color=_GREEN, label="High (≥90%)")
    ax.axvline(0.70, color=_RED,   linewidth=1.0, linestyle="--")
    ax.axvline(0.90, color=_GREEN, linewidth=1.0, linestyle="--")

    ax.set_xlabel("Confidence", color=_MUTED, fontsize=9)
    ax.set_ylabel("Count",      color=_MUTED, fontsize=9)
    ax.set_title("Prediction Confidence Distribution",
                  color=_TEXT, fontsize=10, fontweight="bold")
    ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


# ── Per-class accuracy table ──────────────────────────────────────────────────

def render_per_class_accuracy(report_dict: dict, class_names: Optional[list] = None):
    """Table + bar chart of per-class precision, recall, F1 and support."""
    if not report_dict:
        return
    rows = []
    for key, val in report_dict.items():
        if isinstance(val, dict) and "f1-score" in val:
            rows.append({
                "Class":     str(class_names[int(key)] if (
                                class_names and key.isdigit() and
                                int(key) < len(class_names)) else key),
                "Precision": round(val.get("precision", 0), 4),
                "Recall":    round(val.get("recall", 0), 4),
                "F1":        round(val.get("f1-score", 0), 4),
                "Support":   int(val.get("support", 0)),
            })
    if not rows:
        return

    df = pd.DataFrame(rows)
    st.markdown("#### Per-Class Accuracy")
    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(df, use_container_width=True, hide_index=True)
    with c2:
        fig, ax = _dark_fig(5, max(2.5, len(rows) * 0.55))
        y = range(len(rows))
        ax.barh([r["Class"] for r in rows], df["F1"].values,
                color=_BLUE, height=0.6, alpha=0.80)
        ax.axvline(df["F1"].mean(), color=_AMBER, linewidth=1.0,
                   linestyle="--", label=f"avg={df['F1'].mean():.3f}")
        ax.set_xlabel("F1 Score", color=_MUTED, fontsize=9)
        ax.set_title("Per-Class F1", color=_TEXT, fontsize=10, fontweight="bold")
        ax.legend(facecolor=_BG_DARK, edgecolor=_BORDER, labelcolor=_TEXT, fontsize=8)
        ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ── LaTeX table export ────────────────────────────────────────────────────────

def render_latex_export(results_df: pd.DataFrame):
    """Generate a LaTeX booktabs table from the leaderboard DataFrame."""
    if results_df is None or results_df.empty:
        return
    try:
        latex_str = results_df.to_latex(
            index=False,
            float_format="%.4f",
            bold_rows=False,
            escape=True,
            caption="MLatelier Model Leaderboard",
            label="tab:protoCML_results",
        )
        # Wrap in booktabs-friendly form
        latex_str = latex_str.replace(r"\toprule", r"\hline\hline") \
                             .replace(r"\midrule", r"\hline") \
                             .replace(r"\bottomrule", r"\hline\hline")
    except Exception as e:
        latex_str = f"% LaTeX export failed: {e}"

    st.text_area("LaTeX Table (copy into your paper)", latex_str,
                  height=200, key="_latex_export")
    st.download_button(
        "Download LaTeX (.tex)",
        data=latex_str.encode("utf-8"),
        file_name="MLatelier_Leaderboard.tex",
        mime="text/plain",
    )


# ── Multi-label classification reporting ─────────────────────────────────────

def render_multilabel_report(report_dict: dict):
    """
    Show per-label precision / recall / F1 table plus summary metrics
    for multi-label classification results.
    report_dict must contain 'label_names', 'hamming_loss', 'micro_f1',
    'macro_f1', 'samples_f1' and per-label sub-dicts keyed by label name.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not report_dict or report_dict.get("_task") != "multilabel":
        st.info("No multi-label report available.")
        return

    label_names = report_dict.get("label_names", [])
    if not label_names:
        st.info("No labels found in report.")
        return

    # ── Summary bar ─────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Micro F1",    report_dict.get("micro_f1",    "—"))
    c2.metric("Macro F1",    report_dict.get("macro_f1",    "—"))
    c3.metric("Samples F1",  report_dict.get("samples_f1",  "—"))
    c4.metric("Hamming Loss", report_dict.get("hamming_loss", "—"))

    # ── Per-label table ──────────────────────────────────────────────────────
    rows = []
    for lbl in label_names:
        sub = report_dict.get(lbl, {})
        if isinstance(sub, dict):
            rows.append({
                "Label":     lbl,
                "Precision": round(sub.get("precision", 0.0), 4),
                "Recall":    round(sub.get("recall",    0.0), 4),
                "F1-Score":  round(sub.get("f1-score",  0.0), 4),
                "Support":   int(sub.get("support",     0)),
            })

    if rows:
        df_lbl = pd.DataFrame(rows)
        st.dataframe(df_lbl, use_container_width=True)

        # ── Per-label F1 bar chart ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(max(6, len(rows) * 0.6), 4))
        colors = ["#2196F3" if f >= 0.7 else "#FF9800" if f >= 0.4 else "#F44336"
                  for f in df_lbl["F1-Score"]]
        ax.bar(df_lbl["Label"], df_lbl["F1-Score"], color=colors)
        ax.axhline(0.7, color="green",  linestyle="--", linewidth=0.8, label="0.70 threshold")
        ax.axhline(0.4, color="orange", linestyle="--", linewidth=0.8, label="0.40 threshold")
        ax.set_ylabel("F1-Score")
        ax.set_title("Per-label F1 Score")
        ax.set_ylim(0, 1.05)
        plt.xticks(rotation=30, ha="right", fontsize=8)
        ax.legend(fontsize=7)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render_multilabel_confusion_matrix(cm_data: dict):
    """
    Display per-label 2×2 confusion matrices from multilabel_confusion_matrix output.
    cm_data: {"matrix": [[[TN,FP],[FN,TP]], ...], "labels": [...], "multilabel": True}
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    if not cm_data or not cm_data.get("multilabel"):
        st.info("No multi-label confusion matrix available.")
        return

    matrices = cm_data.get("matrix", [])
    labels   = cm_data.get("labels", [])
    if not matrices or not labels:
        return

    n_labels = len(labels)
    n_cols   = min(4, n_labels)
    n_rows   = (n_labels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3, n_rows * 3))
    axes_flat = np.array(axes).ravel() if n_labels > 1 else [axes]

    for i, (mat, lbl) in enumerate(zip(matrices, labels)):
        ax = axes_flat[i]
        m  = np.array(mat)  # shape (2, 2): [[TN, FP], [FN, TP]]
        im = ax.imshow(m, cmap="Blues")
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(m[r, c]), ha="center", va="center",
                        fontsize=11, color="white" if m[r, c] > m.max() / 2 else "black")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred 0", "Pred 1"], fontsize=7)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["True 0", "True 1"], fontsize=7)
        ax.set_title(lbl, fontsize=8, fontweight="bold")

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── NLP-specific visualisations ───────────────────────────────────────────────

def render_nlp_lime_explanation(lime_data: dict):
    """Horizontal bar chart of LIME word-level explanations."""
    if not lime_data:
        return
    if "error" in lime_data:
        st.warning(f"LIME: {lime_data['error']}")
        return

    explanation = lime_data.get("explanation", [])
    pred_class  = lime_data.get("predicted_class", "?")
    text        = lime_data.get("text", "")

    if not explanation:
        st.info("No LIME explanation available for this sample.")
        return

    st.markdown(f"#### LIME Text Explanation — Prediction: `{pred_class}`")
    if text:
        st.caption(f"Sample: _{text[:200]}{'…' if len(text) > 200 else ''}_")
    st.caption(
        "Green bars indicate words that pushed the model toward this prediction; "
        "red bars indicate words that pushed it away."
    )

    pairs  = sorted(explanation, key=lambda x: abs(x[1]), reverse=True)[:15]
    words  = [p[0] for p in pairs]
    vals   = [p[1] for p in pairs]

    fig, ax = _dark_fig(7, max(3.5, len(words) * 0.44))
    colors  = [_GREEN if v >= 0 else _RED for v in vals]
    ax.barh(range(len(words)), vals, color=colors, height=0.65,
            alpha=0.82, edgecolor=_BG_DARK, linewidth=0.4)
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words, fontsize=9, color=_TEXT)
    ax.axvline(0, color=_BORDER, linewidth=1.0)
    ax.set_xlabel("LIME Weight", color=_MUTED, fontsize=9)
    ax.set_title("Top Word Contributions (LIME)", color=_TEXT,
                 fontsize=10, fontweight="bold")
    ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.6)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_nlp_wordcloud(texts: list, labels=None):
    """Word cloud (or frequency bar chart) for the corpus; per-class breakdown if labels given."""
    import re
    from collections import Counter

    def _tokenize(text: str) -> list:
        return re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())

    try:
        from wordcloud import WordCloud as _WC
        _has_wc = True
    except ImportError:
        _has_wc = False

    all_text = " ".join(str(t) for t in texts)

    if _has_wc:
        st.markdown("#### Word Cloud")
        wc = _WC(
            width=800, height=360,
            background_color=_BG_DARK,
            colormap="Blues",
            max_words=120,
        ).generate(all_text or "no text")
        fig, ax = plt.subplots(figsize=(9, 4))
        fig.patch.set_facecolor(_BG_DARK)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        plt.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.markdown("#### Most Frequent Words")
        st.caption(
            "Install `wordcloud` for a visual cloud: `pip install wordcloud`")
        tokens = _tokenize(all_text)
        top_words = Counter(tokens).most_common(30)
        if top_words:
            words, counts = zip(*top_words)
            fig, ax = _dark_fig(9, 4.5)
            ax.barh(range(len(words)), counts,
                    color=_BLUE, alpha=0.82, height=0.68,
                    edgecolor=_BG_DARK, linewidth=0.4)
            ax.set_yticks(range(len(words)))
            ax.set_yticklabels(words, fontsize=8, color=_TEXT)
            ax.set_xlabel("Frequency", color=_MUTED, fontsize=9)
            ax.set_title("Top 30 Terms (corpus-wide)", color=_TEXT,
                         fontsize=10, fontweight="bold")
            ax.xaxis.grid(True, color=_BORDER, linewidth=0.4)
            ax.set_axisbelow(True)
            plt.tight_layout(pad=0.6)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    # Per-class breakdown
    if labels is not None:
        unique_labels = sorted(set(str(l) for l in labels))
        if 2 <= len(unique_labels) <= 8:
            st.markdown("#### Per-Class Top Terms")
            n_cols = min(len(unique_labels), 4)
            cols = st.columns(n_cols)
            for i, lbl in enumerate(unique_labels):
                class_texts = [
                    str(t) for t, l in zip(texts, labels) if str(l) == lbl]
                tokens = _tokenize(" ".join(class_texts))
                top = Counter(tokens).most_common(10)
                with cols[i % n_cols]:
                    st.caption(f"**{lbl}**")
                    if top:
                        df_lbl = pd.DataFrame(top, columns=["Word", "Count"])
                        st.dataframe(df_lbl, hide_index=True,
                                     use_container_width=True)


def render_nlp_text_stats(texts: list, labels=None):
    """Text length (word-count & char-count) distribution; per-class box plot."""
    word_counts = [len(str(t).split()) for t in texts]
    char_counts  = [len(str(t)) for t in texts]

    st.markdown("#### Text Length Distribution")

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = _dark_fig(4.5, 3.5)
        ax.hist(word_counts, bins=30, color=_BLUE,
                alpha=0.82, edgecolor=_BG_DARK, linewidth=0.4)
        ax.set_xlabel("Word Count", color=_MUTED, fontsize=9)
        ax.set_ylabel("Samples", color=_MUTED, fontsize=9)
        ax.set_title("Word-Count Distribution", color=_TEXT,
                     fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout(pad=0.6)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col2:
        fig, ax = _dark_fig(4.5, 3.5)
        ax.hist(char_counts, bins=30, color=_PURPLE,
                alpha=0.82, edgecolor=_BG_DARK, linewidth=0.4)
        ax.set_xlabel("Character Count", color=_MUTED, fontsize=9)
        ax.set_ylabel("Samples", color=_MUTED, fontsize=9)
        ax.set_title("Char-Count Distribution", color=_TEXT,
                     fontsize=10, fontweight="bold")
        ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)
        ax.set_axisbelow(True)
        plt.tight_layout(pad=0.6)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    stats_rows = [
        ("Min words",    min(word_counts)),
        ("Max words",    max(word_counts)),
        ("Median words", int(np.median(word_counts))),
        ("Mean words",   round(float(np.mean(word_counts)), 1)),
        ("Min chars",    min(char_counts)),
        ("Max chars",    max(char_counts)),
        ("Median chars", int(np.median(char_counts))),
        ("Mean chars",   round(float(np.mean(char_counts)), 1)),
    ]
    st.dataframe(
        pd.DataFrame(stats_rows, columns=["Metric", "Value"]),
        hide_index=True, use_container_width=True,
    )

    # Per-class word-count box plot
    if labels is not None:
        unique_labels = sorted(set(str(l) for l in labels))
        if 2 <= len(unique_labels) <= 12:
            st.markdown("#### Per-Class Word Count")
            class_wc = {lbl: [] for lbl in unique_labels}
            for t, l in zip(texts, labels):
                class_wc[str(l)].append(len(str(t).split()))

            fig, ax = _dark_fig(max(5, len(unique_labels) * 1.3), 3.8)
            data_list = [class_wc[lbl] for lbl in unique_labels]
            bp = ax.boxplot(data_list, patch_artist=True)
            palette = [_BLUE, _GREEN, _PURPLE, _AMBER, _RED, _TEAL] * 4
            for patch, c in zip(bp["boxes"], palette):
                patch.set_facecolor(c)
                patch.set_alpha(0.75)
            for element in ("whiskers", "caps", "medians", "fliers"):
                for item in bp[element]:
                    item.set_color(_MUTED)
            ax.set_xticks(range(1, len(unique_labels) + 1))
            ax.set_xticklabels(
                unique_labels, rotation=30, ha="right",
                fontsize=8, color=_TEXT)
            ax.set_ylabel("Word Count", color=_MUTED, fontsize=9)
            ax.set_title("Word Count by Class", color=_TEXT,
                         fontsize=10, fontweight="bold")
            ax.yaxis.grid(True, color=_BORDER, linewidth=0.4)
            ax.set_axisbelow(True)
            plt.tight_layout(pad=0.6)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


def render_nlp_top_features(top_features: dict):
    """Top TF-IDF coefficient words per class (linear models only)."""
    if not top_features:
        st.info("Top-feature extraction is only available for linear models "
                "(Logistic Regression, Linear SVC, SGD Classifier).")
        return

    st.markdown("#### Top TF-IDF Features per Class")
    st.caption("Words with the highest positive / negative classifier coefficients.")

    for label, sides in top_features.items():
        with st.expander(f"Class: {label}", expanded=False):
            pos = sides.get("positive", [])[:15]
            neg = sides.get("negative", [])[:15]

            if not pos and not neg:
                st.caption("No coefficients extracted.")
                continue

            # Two sub-columns: positive & negative
            c1, c2 = st.columns(2)

            def _bar(ax_local, items, color, title):
                words = [w for w, _ in items]
                scores = [s for _, s in items]
                ax_local.barh(range(len(words)), scores, color=color,
                              height=0.65, alpha=0.82,
                              edgecolor=_BG_DARK, linewidth=0.4)
                ax_local.set_yticks(range(len(words)))
                ax_local.set_yticklabels(words, fontsize=8, color=_TEXT)
                ax_local.set_title(title, color=_TEXT, fontsize=9, fontweight="bold")
                ax_local.tick_params(colors=_MUTED)
                ax_local.spines[:].set_color(_BORDER)
                ax_local.xaxis.grid(True, color=_BORDER, linewidth=0.4)
                ax_local.set_axisbelow(True)

            if pos:
                with c1:
                    fig, ax = plt.subplots(figsize=(3.5, max(2.5, len(pos) * 0.38)))
                    fig.patch.set_facecolor(_BG_DARK)
                    ax.set_facecolor(_BG_CARD)
                    _bar(ax, pos, _GREEN, "Positive (for class)")
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            if neg:
                with c2:
                    fig, ax = plt.subplots(figsize=(3.5, max(2.5, len(neg) * 0.38)))
                    fig.patch.set_facecolor(_BG_DARK)
                    ax.set_facecolor(_BG_CARD)
                    _bar(ax, neg, _RED, "Negative (against class)")
                    plt.tight_layout(pad=0.5)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

