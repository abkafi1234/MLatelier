# MLatelier Documentation

MLatelier is a zero-code machine learning workshop: a Streamlit dashboard that
runs complete ML experiments across tabular, vision, and text data without
requiring the user to write model-training code.

These pages cover both ways of using the project:

- **As an application** — launch the dashboard and drive it through the browser.
- **As a library** — import the engines directly and script experiments in
  Python. Every engine is a plain function that takes a DataFrame (or list of
  texts, or image directory) and returns a result dictionary; nothing depends on
  Streamlit being running.

## Contents

| Page | What it covers |
|---|---|
| [Installation](installation.md) | Install, Python versions, GPU setup, troubleshooting |
| [Quickstart](quickstart.md) | First experiment in the dashboard, in five minutes |
| [Tutorial: Tabular ML](tutorial-tabular.md) | Classification, regression, multi-label, XAI, ablation |
| [Tutorial: NLP](tutorial-nlp.md) | The three text tracks and when to use each |
| [Tutorial: Vision](tutorial-vision.md) | Transfer learning and Grad-CAM |
| [API Reference](api-reference.md) | Every public function, with signatures |
| [Examples](examples/) | Runnable scripts using the engines as a library |
| [Contributing](../CONTRIBUTING.md) | Dev setup, tests, coding conventions |

## The six tabs

| Tab | Purpose |
|---|---|
| Tabular ML | Upload a table, pick a target, train and optimise 20+ models |
| Vision DL | Transfer learning over twelve pre-trained architectures |
| Ablation Study | Measure what each pipeline component contributes |
| Predict / Inference | Load an exported model and score new data |
| NLP | Text classification via TF-IDF, embeddings, or fine-tuning |
| AI Assistant | Gemini-backed interpretation of your actual results |

## Design in one paragraph

The Streamlit UI layer (`app.py`) collects configuration from widgets and
dispatches it to six independent engines. The engines are stateless: they accept
plain arguments and return plain dictionaries, holding no Streamlit state and
performing no I/O beyond reading input data and writing export artefacts. That
separation is why the engines are usable as a library and why the test suite can
exercise the full pipeline without launching a browser.

## Where results go

Results land in two places:

```
~/MLatelier/experiments/EXP_20260811_143205/   # archived per run
├── report.html                                # self-contained
└── *.png                                      # generated curves

~/MLatelier/models/                            # flat, shared
├── MLatelier_Random_Forest.joblib
├── MLatelier_Random_Forest_metadata.json
├── MLatelier_Random_Forest.onnx               # every convertible model
├── MLatelier_XGBoost.joblib
└── MLatelier_XGBoost_metadata.json
```

Every model that was trained is exported, not just the winner, so you can
compare or deploy any of them.

> **Note:** the models directory is flat and filenames are keyed by model name,
> so a later run overwrites an earlier run's export of the same model. Copy
> models you want to keep out of that directory before re-running.

## Citing MLatelier

If MLatelier contributes to published work, please cite the accompanying paper.
A `CITATION.cff` file is provided in the repository root and is picked up
automatically by GitHub's "Cite this repository" button.
