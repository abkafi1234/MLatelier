# Contributing to MLatelier

Contributions are welcome — bug reports, documentation fixes, new models, new
explainability views, or new task types.

## Getting set up

```bash
git clone https://github.com/abkafi1234/MLatelier.git
cd MLatelier
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

The full suite is 390 tests and runs in about a minute. Everything should pass on
a clean checkout; if it does not, that is a bug worth reporting on its own.

```bash
pytest tests/test_tabular_engine.py -v    # one module
pytest -k "multilabel"                    # by keyword
pytest --cov=src/mlatelier --cov-report=term-missing
```

Note the `src/mlatelier` **path**, not the package name. The test modules
`sys.path`-insert `src/mlatelier` and import flat (`from nlp_engine import ...`),
so `--cov=mlatelier` reports near-zero regardless of what the tests actually
exercise.

## Architecture in one paragraph

`app.py` is the Streamlit UI layer: it collects widget state and dispatches to
the engines. The engines (`tabular_engine`, `vision_engine`, `nlp_engine`,
`inference_engine`) are **stateless functions** — they take plain arguments and
return plain dictionaries, import no Streamlit, and hold no session state.
`reporting.py` is the only module that renders to Streamlit. `file_utils.py` and
`tracker.py` are shared utilities.

**Keep engines Streamlit-free.** This is the constraint that makes the test
suite possible without a browser, and it is the one architectural rule that
matters. Where an engine needs to report progress, it takes a callback
(`progress_bar`, `status_text`, `model_callback`) and calls it defensively —
never assume the callback exists or that it will not raise.

## Where things go

| Change | Location |
|---|---|
| New tabular model | `get_classification_models` / `get_regression_models`, plus a space in `get_search_spaces` |
| New vision architecture | `get_vision_model` and `replace_classification_head` |
| New NLP classifier | `get_nlp_ml_models` or `get_nlp_dl_classifiers` |
| New plot or panel | `reporting.py` as a `render_*` function |
| New explainability method | Compute in the engine (returns a dict), render in `reporting.py` |

Adding a model means adding a test that it builds, fits, and predicts. The
existing parametrised tests over the model catalogue will pick it up
automatically in some cases — check whether that already covers you.

## Conventions

- Follow the surrounding style. The codebase uses type hints on public
  signatures, a short docstring stating what is returned, and `snake_case`.
- **Degrade, don't crash.** An optional dependency that is missing, a model
  family an explainer cannot handle, or one model failing to fit should produce
  a `UserWarning` and an empty/`None` result — not an exception that loses the
  whole run. This matters more than usual here: users cannot read the traceback,
  because they are not programmers.
- Guard optional imports with a `_HAS_X` flag at module level, as the existing
  code does for SHAP, LIME, and XGBoost.
- Keep docstrings honest about return shapes. Several are load-bearing for the
  API reference.

## Pull requests

1. Branch from `main`.
2. Make the change, with tests.
3. Run `pytest` locally — CI runs the same suite on Python 3.9 through 3.14,
   so a local failure is a guaranteed CI failure.
4. Update the docs if you changed public behaviour. `docs/api-reference.md` is
   maintained by hand.
5. Open the PR describing what changed and why. Link the issue if there is one.

CI must be green before merge. There is no separate style gate — matching the
surrounding code is the standard.

## Reporting bugs

Open an issue at https://github.com/abkafi1234/MLatelier/issues with:

- What you did, what happened, what you expected
- The full traceback or the on-screen error
- OS, Python version, and `pip show mlatelier` output
- The dataset shape and task type (the data itself is rarely needed — please
  don't attach sensitive data)

## Reporting a security issue

Please do not open a public issue for anything exploitable. Email
kafi.cse@diu.edu.bd directly.

## Licence

By contributing you agree that your contributions are licensed under the MIT
Licence, the same terms that cover the project.
