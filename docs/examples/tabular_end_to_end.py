"""End-to-end tabular workflow using MLatelier as a library.

Baseline cross-validation -> Bayesian optimisation -> SHAP -> export ->
reload -> predict on new rows.

Run:
    python docs/examples/tabular_end_to_end.py
"""

import os
import tempfile

import pandas as pd
from sklearn.datasets import load_breast_cancer

from mlatelier.tabular_engine import (
    compute_shap_values,
    run_tabular_baseline,
    run_tabular_optimization,
)
from mlatelier.inference_engine import load_tabular_model, predict_tabular

MODELS = ["Random Forest", "Logistic Regression", "XGBoost"]
SEED = 42


def main() -> None:
    # ── Data ──────────────────────────────────────────────────────────────
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame
    features = list(bunch.feature_names)
    target = "target"

    print(f"Dataset: {df.shape[0]} rows x {len(features)} features")

    # ── 1. Baseline cross-validation ──────────────────────────────────────
    # Default hyperparameters, so we have something to measure improvement against.
    baseline, task_type = run_tabular_baseline(
        df,
        features_x=features,
        target_y=target,
        selected_models=MODELS,
        handle_imbalance=True,
        scaler="auto",
        test_size=0.2,
        cv_folds=3,
        random_state=SEED,
    )

    print(f"\nDetected task type: {task_type}")
    print("\nBaseline (3-fold CV):")
    for name, r in sorted(baseline.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {name:22s} {r['mean']:.4f} +/- {r['std']:.4f}  ({r['time_s']}s)")

    # ── 2. Bayesian optimisation ──────────────────────────────────────────
    # The winner is chosen by inner CV score. Test performance is computed and
    # reported for every model, but never used to compare them.
    export_dir = tempfile.mkdtemp(prefix="mlatelier_example_")

    (
        pipeline_results,
        winning_test_score,
        avg_base_score,
        improvement,
        p_val_str,
        winning_params,
        winning_report,
        class_names,
        task_type,
        winning_curves,
        actual_scaler,
        exported_path,
    ) = run_tabular_optimization(
        df,
        features_x=features,
        target_y=target,
        selected_models=MODELS,
        baseline_results=baseline,
        handle_imbalance=True,
        scaler="auto",
        test_size=0.2,
        cv_folds=3,
        n_iter=5,               # small budget so the example runs quickly
        export_model=True,
        export_dir=export_dir,
        random_state=SEED,
    )

    print("\nAfter Bayesian optimisation:")
    print(pd.DataFrame(pipeline_results).to_string(index=False))
    print(f"\nScaler selected      : {actual_scaler}")
    print(f"Winner test score    : {winning_test_score:.4f}")
    print(f"Mean baseline score  : {avg_base_score:.4f}")
    print(f"Improvement          : {improvement:.2f}%")
    print(f"McNemar p-value      : {p_val_str}")
    print(f"Winning params       : {winning_params}")

    # ── 3. SHAP global importance ─────────────────────────────────────────
    # Returns {} rather than raising if shap is unavailable or the model
    # family is unsupported, so this is safe to call unconditionally.
    if exported_path and os.path.exists(exported_path):
        pipeline, metadata = load_tabular_model(exported_path)

        X_test = df[features].tail(100)
        shap_vals = compute_shap_values(
            pipeline, X_test, model_name="winner", feature_names=features
        )

        if shap_vals:
            print("\nTop 10 features by mean |SHAP|:")
            top = sorted(shap_vals.items(), key=lambda kv: -abs(kv[1]))[:10]
            for feat, val in top:
                print(f"  {feat:32s} {val:.5f}")
        else:
            print("\nSHAP unavailable for this model/environment - skipped.")

        # ── 4. Predict on new rows ────────────────────────────────────────
        new_rows = df[features].head(5)
        preds = predict_tabular(
            pipeline,
            new_rows,
            feature_names=metadata.get("feature_names", features),
            class_names=metadata.get("class_names"),
        )
        cols = [c for c in preds.columns if c.startswith(("Prediction", "Confidence", "P("))]
        print("\nPredictions on 5 new rows:")
        print(preds[cols].to_string(index=False))

    print(f"\nArtefacts written to: {export_dir}")


if __name__ == "__main__":
    main()
