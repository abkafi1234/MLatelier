import os
import json
import datetime
import pandas as pd
import numpy as np
import platform
import sys
import io
import base64
import importlib.metadata
from typing import Optional
import matplotlib

# Force matplotlib to run silently in the background
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def save_minimal_experiment(
    task_type,
    models_raced,
    results_df,
    report_dict,
    best_score,
    winning_curves=None,
    scaler_used="N/A",
    best_params=None,
    label_classes: Optional[list] = None,
    experiment_metadata: Optional[dict] = None,
):
    """
    Saves core scientific files, generated curve images, AND builds the
    fully self-contained HTML optimization report in a single pass.
    """
    now = datetime.datetime.now()
    folder_time = now.strftime("%Y%m%d_%H%M%S") 
    display_time = now.strftime("%Y-%m-%d %H:%M:%S")

    base_dir = os.path.join(os.path.expanduser("~"), "MLatelier", "experiments")
    run_dir = os.path.join(base_dir, f"EXP_{folder_time}")
    os.makedirs(run_dir, exist_ok=True)

    # 1. Save config.json
    config = {
        "task_type": task_type,
        "models_raced": models_raced,
        "scaler_used": scaler_used,
        "timestamp": display_time,
    }
    if experiment_metadata:
        config["random_seed"] = experiment_metadata.get("random_seed")
        config["dataset_fingerprint_sha256"] = experiment_metadata.get(
            "dataset_fingerprint_sha256"
        )
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    # 2. Extract and Save environment.json
    env_data = {
        "platform": platform.system(),
        "python": sys.version.split(" ")[0],
    }

    try:
        import torch

        env_data["torch"] = torch.__version__
        env_data["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        env_data["torch"] = "Not Installed"
        env_data["device"] = "unknown"

    try:
        import sklearn

        env_data["sklearn"] = sklearn.__version__
    except ImportError:
        env_data["sklearn"] = "Not Installed"

    packages_to_track = {
        "numpy": "numpy",
        "pandas": "pandas",
        "scipy": "scipy",
        "xgboost": "xgboost",
        "lightgbm": "lightgbm",
        "catboost": "catboost",
        "shap": "shap",
        "skopt": "scikit-optimize",
        "imbalanced_learn": "imbalanced-learn",
        "streamlit": "streamlit",
        "torchvision": "torchvision",
    }

    for json_key, pip_name in packages_to_track.items():
        try:
            env_data[json_key] = importlib.metadata.version(pip_name)
        except importlib.metadata.PackageNotFoundError:
            env_data[json_key] = "Not Installed"

    with open(os.path.join(run_dir, "environment.json"), "w") as f:
        json.dump(env_data, f, indent=4)

    if experiment_metadata:
        with open(os.path.join(run_dir, "reproducibility_metadata.json"), "w") as f:
            json.dump(experiment_metadata, f, indent=4, default=str)

    # 3. Save leaderboard.csv
    if results_df is not None:
        results_df.to_csv(os.path.join(run_dir, "leaderboard.csv"), index=False)

    # 4. Identify Winning Model
    winning_model = "Unknown"
    if results_df is not None and not results_df.empty:
        winning_model = (
            results_df.iloc[0]["Algorithm"]
            if "Algorithm" in results_df.columns
            else results_df.iloc[0]["Architecture"]
        )

    # 5. Save JSON Reports
    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(report_dict, f, indent=4, default=str)

    with open(os.path.join(run_dir, "best_params.json"), "w") as f:
        json.dump(best_params or {}, f, indent=4, default=str)

    summary = {
        "winning_model": winning_model,
        "best_score": best_score,
        "task_type": task_type,
        "scaler_used": scaler_used,
        "models_raced": models_raced,
        "timestamp": display_time,
    }
    with open(os.path.join(run_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # 6. GENERATE VISUALS (Save PNGs AND build HTML Base64 strings)
    def fig_to_base64(fig):
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    convergence_html = ""
    feature_html = ""
    training_html = ""
    shap_html = ""

    if winning_curves:
        # A. Convergence Curve
        if "bo_history" in winning_curves and winning_curves["bo_history"]:
            with open(os.path.join(run_dir, "optimization_history.json"), "w") as f:
                json.dump(winning_curves["bo_history"], f, indent=4)

            bo_scores = [trial["score"] for trial in winning_curves["bo_history"]]
            best_so_far = np.maximum.accumulate(bo_scores)
            trials = range(1, len(bo_scores) + 1)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(
                trials,
                bo_scores,
                marker="x",
                linestyle="",
                color="#aec7e8",
                alpha=0.7,
                label="Trial Score",
            )
            ax.plot(
                trials,
                best_so_far,
                marker="o",
                linestyle="-",
                color="#1f77b4",
                linewidth=2,
                label="Best Score",
            )
            ax.set_xlabel("Optimization Trial")
            ax.set_ylabel("Metric Score")
            ax.set_title("Bayesian Optimization Convergence")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()

            fig.savefig(
                os.path.join(run_dir, "convergence_curve.png"), bbox_inches="tight"
            )
            img = fig_to_base64(fig)
            plt.close(fig)

            convergence_html = f"""
            <h2>6. Optimization Diagnostics</h2>
            <p>This convergence curve shows how Bayesian Optimization explored the search space and improved performance across successive trials.</p>
            <div class="image-card"><img src="data:image/png;base64,{img}" /></div>
            """

        # B. Feature Importance
        if (
            "feature_importances" in winning_curves
            and winning_curves["feature_importances"]
        ):
            df_imp = pd.DataFrame(
                {
                    "Feature": list(winning_curves["feature_importances"].keys()),
                    "Importance": list(winning_curves["feature_importances"].values()),
                }
            ).sort_values(by="Importance", ascending=True)

            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(df_imp["Feature"], df_imp["Importance"], color="#17becf")
            ax.set_xlabel("Relative Importance / Weight")
            ax.set_title("Drivers of Model Prediction")
            ax.grid(axis="x", linestyle="--", alpha=0.6)

            fig.savefig(
                os.path.join(run_dir, "feature_importance.png"), bbox_inches="tight"
            )
            img = fig_to_base64(fig)
            plt.close(fig)

            feature_html = f"""
            <h2>7. Explainability Analysis</h2>
            <p>Feature importance reveals the exact variables that most strongly influenced the model's final prediction.</p>
            <div class="image-card"><img src="data:image/png;base64,{img}" /></div>
            """

        # B2. SHAP bar chart (tabular)
        shap_html = ""
        if winning_curves.get("shap_values"):
            sv = winning_curves["shap_values"]
            df_shap = pd.DataFrame({"Feature": list(sv.keys()),
                                     "Mean |SHAP|": list(sv.values())}) \
                        .sort_values("Mean |SHAP|", ascending=True)
            fig, ax = plt.subplots(figsize=(7, max(3, len(df_shap) * 0.3)))
            ax.barh(df_shap["Feature"], df_shap["Mean |SHAP|"], color="#8A63FF")
            ax.set_xlabel("Mean |SHAP value|")
            ax.set_title("SHAP Feature Importance")
            ax.grid(axis="x", linestyle="--", alpha=0.5)
            fig.tight_layout()
            img = fig_to_base64(fig)
            plt.close(fig)
            shap_html = (
                f'<h2>7b. SHAP Explainability</h2>'
                f'<p>Mean absolute SHAP values show how much each feature contributes to predictions.</p>'
                f'<div class="image-card"><img src="data:image/png;base64,{img}" /></div>'
            )

        # B3. Save exported model path
        if winning_curves.get("exported_model_path"):
            summary["exported_model_path"] = winning_curves["exported_model_path"]

        # C. Training Curves
        if "train_losses" in winning_curves and "val_f1s" in winning_curves:
            train_losses, val_f1s = (
                winning_curves["train_losses"],
                winning_curves["val_f1s"],
            )
            epochs = range(1, len(train_losses) + 1)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.plot(epochs, train_losses, marker="o", color="#d62728")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Training Loss")
            ax1.set_title("Model Learning")
            ax1.grid(True, linestyle="--", alpha=0.6)

            ax2.plot(epochs, val_f1s, marker="o", color="#2ca02c")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Validation Macro F1")
            ax2.set_title("Generalization")
            ax2.grid(True, linestyle="--", alpha=0.6)

            fig.savefig(
                os.path.join(run_dir, "training_curves.png"), bbox_inches="tight"
            )
            img = fig_to_base64(fig)
            plt.close(fig)

            training_html = f"""
            <h2>7. Deep Learning Training Diagnostics</h2>
            <p>These training curves visualize model learning behavior and generalization quality during training.</p>
            <div class="image-card"><img src="data:image/png;base64,{img}" /></div>
            """

    # 7. ASSEMBLE FULL HTML REPORT
    def _display_class_label(label):
        label_text = str(label)
        if label_classes is None or not label_text.lstrip("-").isdigit():
            return label

        label_index = int(label_text)
        if 0 <= label_index < len(label_classes):
            return str(label_classes[label_index])

        return label

    def _classification_report_html(raw_report):
        # Strip non-standard keys (confusion_matrix, y_true, y_pred, etc.)
        _NON_METRIC_KEYS = {"confusion_matrix", "confusion_matrix_labels",
                             "y_true", "y_pred"}
        clean = {k: v for k, v in raw_report.items()
                 if k not in _NON_METRIC_KEYS and isinstance(v, dict)}
        if not clean:
            return "<p>No per-class metrics available.</p>"
        rep_df = pd.DataFrame(clean).transpose()
        rep_df = rep_df.rename(index=_display_class_label)
        numeric_cols = rep_df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            rep_df[numeric_cols] = rep_df[numeric_cols].round(3)
        return rep_df.to_html(border=0, classes="dataframe")

    # Format Hyperparameters
    if best_params:
        params_df = pd.DataFrame(
            {
                "Hyperparameter": list(best_params.keys()),
                "Value": list(best_params.values()),
            }
        )
        params_html = params_df.to_html(index=False, border=0, classes="dataframe")
    else:
        params_html = "<p>Default Baseline Parameters Used</p>"

    # Format Metrics Section
    if task_type == "regression":
        metric_rows = "".join(
            [
                f"<tr><td>{k}</td><td>{v}</td></tr>"
                for k, v in report_dict.items()
                if k not in ["y_true", "y_pred"]
            ]
        )
        metrics_html = f'<table class="dataframe"><thead><tr><th>Metric</th><th>Value</th></tr></thead><tbody>{metric_rows}</tbody></table>'
    else:
        try:
            metrics_html = _classification_report_html(report_dict)
        except Exception:
            metrics_html = f"<pre>{json.dumps(report_dict, indent=4)}</pre>"

    # Format Leaderboard
    leaderboard_html = (
        results_df.to_html(index=False, border=0, classes="dataframe")
        if results_df is not None
        else "<p>No Leaderboard Available</p>"
    )

    html_content = f"""
    <html>
    <head>
        <title>MLatelier Full Optimization Report</title>
        <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background-color: #F8F9FA; color: #121214; line-height: 1.7; padding: 40px; }}
        .container {{ max-width: 1100px; margin: auto; background: white; border-radius: 16px; padding: 50px; box-shadow: 0 8px 30px rgba(0,0,0,0.08); }}
        h1 {{ border-bottom: 4px solid #5B9AFF; padding-bottom: 12px; margin-bottom: 30px; }}
        h2 {{ margin-top: 40px; border-bottom: 1px solid #EAEAEA; padding-bottom: 8px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; margin-bottom: 25px; }}
        th, td {{ border: 1px solid #EAEAEA; padding: 12px; text-align: left; }}
        th {{ background-color: #F8F9FA; text-transform: uppercase; font-size: 0.85em; }}
        .summary-box {{ background: #F8F9FA; border-left: 5px solid #5B9AFF; border-radius: 10px; padding: 20px; }}
        .badge {{ background-color: #E8F0FE; color: #1967D2; padding: 4px 10px; border-radius: 20px; font-weight: 600; }}
        .image-card {{ background: white; padding: 20px; border-radius: 12px; border: 1px solid #EAEAEA; text-align: center; }}
        img {{ max-width: 100%; border-radius: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>MLatelier Full Optimization Report</h1>
            
            <h2>1. Executive Summary</h2>
            <div class="summary-box">
                <p><b>Winning Model:</b> {winning_model}</p>
                <p><b>Best Score:</b> {best_score}</p>
                <p><b>Task Type:</b> <span class="badge">{task_type.upper()}</span></p>
                <p><b>Scaler Used:</b> <span class="badge">{str(scaler_used).upper()}</span></p>
                <p><b>Candidate Models:</b> {", ".join(models_raced) if models_raced else "N/A"}</p>
                <p><b>Generated:</b> {display_time}</p>
            </div>

            <h2>2. Execution Metadata</h2>
            <table class="dataframe">
                <tr><td>Platform</td><td>{env_data.get("platform", "N/A")}</td></tr>
                <tr><td>Python</td><td>{env_data.get("python", "N/A")}</td></tr>
                <tr><td>PyTorch</td><td>{env_data.get("torch", "N/A")}</td></tr>
                <tr><td>Device</td><td>{env_data.get("device", "N/A").upper()}</td></tr>
                <tr><td>Scikit-Learn</td><td>{env_data.get("sklearn", "N/A")}</td></tr>
                <tr><td>NumPy</td><td>{env_data.get("numpy", "N/A")}</td></tr>
                <tr><td>Pandas</td><td>{env_data.get("pandas", "N/A")}</td></tr>
                <tr><td>SciPy</td><td>{env_data.get("scipy", "N/A")}</td></tr>
                <tr><td>XGBoost</td><td>{env_data.get("xgboost", "N/A")}</td></tr>
                <tr><td>Scikit-Optimize</td><td>{env_data.get("skopt", "N/A")}</td></tr>
                <tr><td>Imbalanced Learn</td><td>{env_data.get("imbalanced_learn", "N/A")}</td></tr>
                <tr><td>Streamlit</td><td>{env_data.get("streamlit", "N/A")}</td></tr>
                <tr><td>Dataset Fingerprint</td><td>{(experiment_metadata or {}).get("dataset_fingerprint_sha256", "N/A")}</td></tr>
            </table>

            <h2>3. Hyperparameter Configuration</h2>
            {params_html}

            <h2>4. Global Leaderboard</h2>
            {leaderboard_html}

            <h2>5. Final Validation Metrics</h2>
            {metrics_html}

            {convergence_html}
            {feature_html}
            {shap_html}
            {training_html}

            <h2>8. Reproducibility Manifest</h2>
            <p>This report was automatically generated by MLatelier and contains experiment metadata, optimization diagnostics, evaluation metrics, and reproducibility information.</p>
        </div>
    </body>
    </html>
    """

    return run_dir, html_content
