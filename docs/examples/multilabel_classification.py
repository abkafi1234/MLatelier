"""Multi-label classification with MLatelier.

Multi-label is the capability that distinguishes MLatelier from the other
no-code tools: each sample carries a label VECTOR rather than one class.
MLatelier wraps the base estimator in MultiOutputClassifier and rewrites the
BayesSearchCV parameter keys into the `model__estimator__` namespace so
hyperparameters route through the wrapper correctly.

Run:
    python docs/examples/multilabel_classification.py
"""

import pandas as pd
from sklearn.datasets import make_multilabel_classification

from mlatelier.tabular_engine import run_tabular_baseline, run_tabular_optimization

SEED = 42
N_LABELS = 4


def build_dataset() -> tuple[pd.DataFrame, list, list]:
    """Synthetic multi-label problem: 20 features, 4 binary label columns."""
    X, Y = make_multilabel_classification(
        n_samples=400,
        n_features=20,
        n_classes=N_LABELS,
        n_labels=2,
        allow_unlabeled=False,
        random_state=SEED,
    )

    feature_cols = [f"feature_{i}" for i in range(X.shape[1])]
    label_cols = [f"label_{j}" for j in range(Y.shape[1])]

    df = pd.concat(
        [pd.DataFrame(X, columns=feature_cols), pd.DataFrame(Y, columns=label_cols)],
        axis=1,
    )
    return df, feature_cols, label_cols


def main() -> None:
    df, feature_cols, label_cols = build_dataset()

    print(f"Dataset: {df.shape[0]} rows, {len(feature_cols)} features, "
          f"{len(label_cols)} labels")
    print("\nLabel cardinality (mean labels per sample): "
          f"{df[label_cols].sum(axis=1).mean():.2f}")
    print("\nPer-label positive rate:")
    for col in label_cols:
        print(f"  {col}: {df[col].mean():.3f}")

    # Passing a LIST of target columns is what triggers multi-label handling.
    # SMOTE is not applicable here and is disabled automatically.
    models = ["Random Forest", "Extra Trees"]

    baseline, task_type = run_tabular_baseline(
        df,
        features_x=feature_cols,
        target_y=label_cols,
        selected_models=models,
        handle_imbalance=False,
        cv_folds=3,
        random_state=SEED,
    )

    print(f"\nDetected task type: {task_type}")   # -> "multilabel"
    print("\nBaseline (micro F1, 3-fold CV):")
    for name, r in sorted(baseline.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {name:18s} {r['mean']:.4f} +/- {r['std']:.4f}")

    results = run_tabular_optimization(
        df,
        features_x=feature_cols,
        target_y=label_cols,
        selected_models=models,
        baseline_results=baseline,
        handle_imbalance=False,
        cv_folds=3,
        n_iter=5,
        export_model=False,
        random_state=SEED,
    )

    pipeline_results = results[0]
    winning_test_score = results[1]
    improvement = results[3]
    winning_report = results[6]

    print("\nAfter optimisation:")
    print(pd.DataFrame(pipeline_results).to_string(index=False))
    print(f"\nWinner micro F1 : {winning_test_score:.4f}")
    print(f"Improvement     : {improvement:.2f}%")

    # The multi-label report carries micro/macro/samples F1, Hamming loss,
    # and per-label precision/recall/F1.
    # NOTE: these summary values are pre-formatted STRINGS, not floats.
    if isinstance(winning_report, dict):
        for key in ("micro_f1", "macro_f1", "samples_f1", "hamming_loss"):
            if key in winning_report:
                print(f"{key:15s}: {winning_report[key]}")

    print(
        "\nNote: SHAP, LIME and PDP are unavailable for multi-label models -\n"
        "MultiOutputClassifier breaks the single-output assumption those\n"
        "explainers rely on. Feature importance is averaged across the\n"
        "per-label estimators instead, which hides label-specific effects."
    )


if __name__ == "__main__":
    main()
