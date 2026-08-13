# Tutorial: Tabular ML

Covers the four tabular task types, the explainability panel, and the ablation
module. Assumes you have been through the [Quickstart](quickstart.md).

## Task detection

You never declare the task type — it is inferred from the target column(s):

| Target | Inferred task | Metric |
|---|---|---|
| Discrete values (2 or more) | `classification` | Macro F1 |
| Continuous numeric | `regression` | R² |
| Several binary columns selected | `multilabel` | Micro F1 |

Binary and multiclass are not distinguished — the engine reports both as
`classification`. Cross-validation follows: `StratifiedKFold` for single-label
classification, plain `KFold` for regression and multi-label.

If detection surprises you, it is almost always the data. An integer-coded
category with 40 levels looks continuous; cast it to string first.

## Preprocessing

The pipeline is assembled as an `ImbPipeline` so that scaling and resampling are
fitted **inside** each CV fold. This is the difference between an honest score
and a leaked one, and it is why you should not pre-scale your data before
uploading it.

**Scaler.** `auto` inspects skew and outlier fraction and picks between
`StandardScaler`, `RobustScaler`, and `MinMaxScaler`. Override it when you know
something the heuristic does not.

**SMOTE.** Applied to the training folds only, never to validation or test data.
Turn it off for regression and multi-label targets — it is disabled for those
automatically, but the checkbox still reflects your choice.

## Bayesian optimisation

The optimiser is `BayesSearchCV` with a Gaussian Process surrogate and Expected
Improvement acquisition, searching a hand-defined space per model family.

The budget you set is multiplied internally:

```
effective_iterations = n_iter × min(len(search_space), 3)
```

so `n_iter=10` on a model with three or more tunable hyperparameters is 30 real
evaluations. Watch the **convergence curve**: if it is still descending at the
end, raise the budget; if it flattened a third of the way in, you are burning
time for nothing.

### How the winner is picked

```
m̂ = argmax_m score_CV(m, D_train)
```

Selection uses the inner cross-validation score — `BayesSearchCV.best_score_`,
or the baseline CV mean for models with no search space. Test performance **is**
computed and displayed for every model, both baseline and optimised, but it
never enters the comparison that picks the winner.

That distinction matters. If you selected the winner by test score, the reported
figure would be the maximum of twenty test evaluations, which is optimistically
biased by however many models you compared. Keeping the test set out of the
selection step is what avoids that.

One residual bias remains and is worth knowing about: the same folds guide both
the hyperparameter search and the ranking, so the CV score itself is slightly
optimistic. Fully nested cross-validation removes this, at several times the
compute. For a headline published figure, hold back a third split scored exactly
once.

## Multi-label classification

Select several binary columns as targets. MLatelier wraps the base estimator in
`MultiOutputClassifier`, training one binary classifier per label, and rewrites
the `BayesSearchCV` parameter keys into the `model__estimator__` namespace so
hyperparameters route through the wrapper correctly.

You get micro / macro / samples-averaged F1, Hamming loss, per-label
precision-recall tables, and per-label confusion matrices.

**Limitation.** SHAP, LIME, and PDP are unavailable here. Feature importance
comes from averaging the per-label estimators:

```
φ̄_k = (1/L) · Σ_j φ_k^(j)
```

That average is a real loss of information — a feature can matter enormously for
one rare label and not at all elsewhere, and averaging hides it. Read per-label
importances when a specific label is what you care about.

## The explainability panel

| View | Scope | Answers |
|---|---|---|
| SHAP global (beeswarm) | Whole model | Which features matter, and in which direction |
| SHAP waterfall | One sample | Why did *this row* get *this* prediction |
| LIME | One sample | Same question, via an independent local surrogate |
| PDP | Whole model | How the prediction moves as one feature varies |
| Calibration curve | Whole model | Does "0.9" actually mean 90 % |
| Learning curve | Whole model | Would more data help |
| Feature ablation | Whole model | What does dropping each feature cost |

Two habits worth forming:

**Cross-check SHAP against LIME.** They are methodologically independent. When
they agree on the top features, the ranking is worth considerably more than
either alone. When they disagree sharply, treat both as unreliable for that
model and investigate.

**Read the calibration curve before trusting probabilities.** A model with
excellent F1 can still be badly calibrated, which matters the moment anyone
thresholds the probability to make a decision.

SHAP is computed on at most 300 test rows for tractability; it returns the top
20 features.

## The ablation module

Ablation answers "which parts of my pipeline actually earn their place?"

Define a baseline, then variants that each change one component:

```python
ablation_configs = [
    {"label": "Baseline (Robust + SMOTE)", "scaler": "robust",   "handle_imbalance": True},
    {"label": "No SMOTE",                  "scaler": "robust",   "handle_imbalance": False},
    {"label": "Standard scaler",           "scaler": "standard", "handle_imbalance": True},
    {"label": "MinMax scaler",             "scaler": "minmax",   "handle_imbalance": True},
    {"label": "10-fold CV",                "scaler": "robust",   "cv_folds": 10},
]
```

Valid `scaler` values are `"auto"`, `"robust"`, `"standard"`, and `"minmax"`.
Anything else silently falls back to `StandardScaler`, so a mistyped key
produces a duplicate arm rather than an error — check that your labels and
scores actually differ.

Results are reported as a delta against the first configuration. Components that
cost time and deliver a delta inside the fold-to-fold standard deviation are
components you can drop.

**Feature ablation** is separate: it drops one feature at a time and reports the
CV cost of each. It needs at least two features.

## Working with wide data

The `n_features > n_samples` regime is where zero-code tooling most easily
misleads. Practical guidance:

- Prefer regularised linear models and tree ensembles over SVMs with RBF kernels
- Raise the test split — a 20 % split of 200 rows is 40 rows and will not
  separate models reliably
- Treat differences smaller than the CV standard deviation as noise, not ranking
- Feature ablation gets expensive fast; run it on a shortlist

## Next

- [Tutorial: NLP](tutorial-nlp.md)
- [Examples](examples/) — this workflow as a script
