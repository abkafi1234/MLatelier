# Quickstart — your first experiment in five minutes

This walks through a complete tabular classification experiment in the
dashboard, from raw file to explained result. No code.

## 0. Get a dataset

Any CSV with one row per observation, feature columns, and one target column.
If you want a known-good file to practise on:

```python
from sklearn.datasets import load_breast_cancer
df = load_breast_cancer(as_frame=True).frame
df.to_csv("breast_cancer.csv", index=False)
```

## 1. Launch

```bash
mlatelier
```

The browser opens on the **Tabular ML** tab.

## 2. Upload

Drop `breast_cancer.csv` onto the upload panel. CSV, Excel, a direct HTTP(S)
URL, or a local path all work; multi-sheet workbooks prompt for a sheet.

An exploratory panel appears immediately: shape, dtypes, missingness, summary
statistics, correlation heatmap, and per-feature distributions. Read it before
training — this is where you notice a constant column or 60 % missingness in a
feature, which no amount of hyperparameter search will fix.

## 3. Choose the target

Pick `target` from the target dropdown. MLatelier infers the task type:

| What it sees | Task inferred | Metric |
|---|---|---|
| Discrete values (2 or more) | `classification` | Macro F1 |
| Continuous numeric | `regression` | R² |
| Multiple binary label columns | `multilabel` | Micro F1 |

Binary and multiclass are not distinguished — both are `classification` and both
are scored with macro F1.

## 4. Configure

Sensible defaults are pre-filled; the ones worth understanding:

| Setting | Default | When to change it |
|---|---|---|
| Scaler | `auto` | `auto` inspects skew and outliers and picks. Force `RobustScaler` if you know you have heavy outliers |
| Handle imbalance (SMOTE) | on | Turn off for regression, multi-label, or balanced data |
| Test split | 0.2 | Raise for small datasets where 20 % is too few rows to trust |
| CV folds | 3 | Raise to 5 or 10 for a more stable estimate |
| BO iterations | 10 | Raise to 30+ for a serious run; this is the main speed/quality dial |

Select several models. Start with Random Forest, XGBoost, and Logistic
Regression — fast, and they cover three quite different model families.

## 5. Train

Press **Run Baseline**. Each model is cross-validated with default
hyperparameters and the leaderboard fills in live.

Then press **Run Optimisation**. Bayesian search runs per model, and each is
refitted on the full training set and scored on the held-out test set.

> The winner is chosen by **inner cross-validation score**. Test performance is
> shown for every model, but it never enters the selection — so the winner is
> not simply whichever model got lucky on the test split.

## 6. Read the diagnostics

- **Leaderboard** — every model, baseline vs optimised, with timings
- **Classification report** — per-class precision, recall, F1
- **Confusion matrix** and **ROC curves**
- **SHAP global** — which features drive the model overall
- **SHAP waterfall** and **LIME** — why *this one sample* got *this* prediction
- **PDP** — how the prediction moves as one feature varies
- **Calibration curve** — whether a "0.9" really means 90 % of the time
- **Convergence curve** — did the optimiser plateau, or would more iterations help?
- **Learning curve** — would more *data* help?

A useful habit: check whether SHAP and LIME agree on the top features. When two
methodologically independent explainers rank features the same way, the ranking
is worth more than either alone.

## 7. Export

The HTML report and generated curves are archived under
`~/MLatelier/experiments/EXP_<timestamp>/`. Models go separately to
`~/MLatelier/models/` — a `.joblib` per model with a JSON metadata sidecar. The
leaderboard CSV and a LaTeX `booktabs` table are offered as download buttons in
the UI rather than written to disk.

Under **Download All Models** you also get an **ONNX** section. Those `.onnx`
files run under any ONNX runtime
— C++, C#, Java, JavaScript, mobile — with the scaling and encoding baked into
the graph, so you do not have to reimplement preprocessing on the target
platform. CatBoost has no converter and stays `.joblib`-only.

> The models directory is flat, so re-running overwrites the previous export of
> the same model. Copy out anything you need to keep.

## 8. Predict on new data

Go to **Predict / Inference**, upload the `.joblib`, then supply new rows by
paste, file, or URL. You get predictions with per-class probabilities and a
confidence histogram. Missing feature columns are filled with 0 and you are
warned which ones — check that warning rather than ignoring it.

## Next

- [Tutorial: Tabular ML](tutorial-tabular.md) — multi-label, ablation, the XAI panel in depth
- [Examples](examples/) — the same workflow as a Python script
