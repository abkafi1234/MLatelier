"""Pipeline ablation: which components actually earn their place?

Two kinds of ablation are demonstrated:

  1. Component ablation - vary scaler / SMOTE / split / CV folds and measure
     each variant against a baseline configuration.
  2. Feature ablation   - drop one feature at a time and measure the cost.

The useful reading of both: any delta smaller than the fold-to-fold standard
deviation is noise, not a result.

Run:
    python docs/examples/ablation_study.py
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

from mlatelier.tabular_engine import run_feature_ablation, run_tabular_ablation

SEED = 42
MODEL = "Random Forest"


def main() -> None:
    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame
    features = list(bunch.feature_names)
    target = "target"

    # ── 1. Component ablation ─────────────────────────────────────────────
    # The FIRST config is the baseline; every Delta(%) is measured against it.
    ablation_configs = [
        {"label": "Baseline (auto scaler + SMOTE)",
         "scaler": "auto", "handle_imbalance": True},
        {"label": "No SMOTE",
         "scaler": "auto", "handle_imbalance": False},
        {"label": "Robust scaler",
         "scaler": "robust", "handle_imbalance": True},
        {"label": "Standard scaler",
         "scaler": "standard", "handle_imbalance": True},
        {"label": "MinMax scaler",
         "scaler": "minmax", "handle_imbalance": True},
        {"label": "Larger test split (0.3)",
         "scaler": "auto", "handle_imbalance": True, "test_size": 0.3},
        {"label": "10-fold CV",
         "scaler": "auto", "handle_imbalance": True, "cv_folds": 10},
    ]

    print(f"Component ablation on {MODEL} ({len(ablation_configs)} configurations)\n")

    component_results = run_tabular_ablation(
        df,
        features_x=features,
        target_y=target,
        model_name=MODEL,
        ablation_configs=ablation_configs,
        random_state=SEED,
    )

    comp_df = pd.DataFrame(component_results)
    print(comp_df.to_string(index=False))

    print(
        "\nInterpretation: a component whose Delta(%) is smaller than the\n"
        "Std(+/-) column is not doing measurable work. Prefer the cheaper\n"
        "configuration in that case."
    )

    # ── 2. Feature ablation ───────────────────────────────────────────────
    # Drop one feature at a time. Requires at least 2 features; cost grows
    # linearly with feature count, so use a shortlist on wide data.
    shortlist = features[:8]

    print(f"\n\nFeature ablation over {len(shortlist)} features\n")

    feature_results = run_feature_ablation(
        df,
        features_x=shortlist,
        target_y=target,
        model_name=MODEL,
        handle_imbalance=True,
        scaler="auto",
        cv_folds=3,
        random_state=SEED,
    )

    feat_df = pd.DataFrame(feature_results)
    if "Delta (%)" in feat_df.columns:
        feat_df = feat_df.sort_values("Delta (%)")
    print(feat_df.to_string(index=False))

    print(
        "\nA strongly NEGATIVE delta means removing that feature hurt - it was\n"
        "carrying signal. A delta near zero or positive means the feature was\n"
        "redundant or actively noisy, and is a candidate for removal."
    )


if __name__ == "__main__":
    main()
