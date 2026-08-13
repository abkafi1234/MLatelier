"""Compare MLatelier's NLP tracks on the same corpus.

Track "ml" - TF-IDF features + classic classifiers. Fast, CPU-only,
             explainable at the term level.
Track "dl" - Sentence Transformer embeddings + the same classifiers.
             Captures semantics that bag-of-words cannot; no term-level
             explanation is possible.

The HuggingFace fine-tuning track (run_transformer_finetune) is shown at the
bottom but not executed by default - it needs several minutes and, realistically,
a GPU.

Run:
    python docs/examples/nlp_track_comparison.py
"""

from sklearn.datasets import fetch_20newsgroups

from mlatelier.nlp_engine import get_nlp_device, get_nlp_ml_models, run_nlp_baseline

SEED = 42
CATEGORIES = ["rec.sport.hockey", "sci.med", "comp.graphics"]


def load_corpus() -> tuple[list[str], list[str]]:
    """Three well-separated newsgroups; headers/footers stripped to avoid leakage.

    Note: this is the one example that downloads data (~14 MB, cached by
    scikit-learn afterwards).
    """
    bunch = fetch_20newsgroups(
        subset="train",
        categories=CATEGORIES,
        remove=("headers", "footers", "quotes"),
        random_state=SEED,
    )
    texts = [t for t in bunch.data if t.strip()]
    labels = [bunch.target_names[y] for t, y in zip(bunch.data, bunch.target) if t.strip()]

    # Keep it small so the example finishes quickly.
    return texts[:600], labels[:600]


def report(title: str, results: dict) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for name, r in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
        print(f"  {name:22s} {r['mean']:.4f} +/- {r['std']:.4f}   ({r['time_s']}s)")


def main() -> None:
    print(f"Device: {get_nlp_device()}")

    texts, labels = load_corpus()
    print(f"Corpus: {len(texts)} documents across {len(set(labels))} classes")

    model_names = list(get_nlp_ml_models().keys())

    # ── ML track: TF-IDF ──────────────────────────────────────────────────
    ml_results, class_names = run_nlp_baseline(
        texts,
        labels,
        model_names=model_names,
        track="ml",
        use_stopwords=True,
        use_stemming=False,
        ngram_max=2,          # bigrams capture negation; unigrams cannot
        max_features=5000,
        min_df=2,
        cv_folds=3,
        random_state=SEED,
    )
    report("ML track (TF-IDF)", ml_results)

    # ── DL track: Sentence Transformer embeddings ─────────────────────────
    # Texts are encoded ONCE, then the same classifiers train on the vectors.
    # The encoder itself is not fine-tuned.
    try:
        dl_results, _ = run_nlp_baseline(
            texts,
            labels,
            model_names=["Logistic Regression", "Linear SVC", "Random Forest"],
            track="dl",
            st_model_name="all-MiniLM-L6-v2",
            device="cpu",     # set "cuda" if a GPU is available
            cv_folds=3,
            random_state=SEED,
        )
        report("DL track (Sentence Transformers)", dl_results)
    except Exception as exc:
        print(f"\nDL track skipped: {exc}")

    print(f"\nClasses: {class_names}")

    print(
        "\n"
        "Reading the comparison:\n"
        "  - If TF-IDF is close to the embedding track, your classes are\n"
        "    separable by vocabulary alone. Prefer TF-IDF: it is far cheaper\n"
        "    and you get term-level explanations.\n"
        "  - If the embedding track wins clearly, word ORDER and semantics\n"
        "    matter for your problem, and fine-tuning is worth considering.\n"
    )

    # ── HF track (not executed) ───────────────────────────────────────────
    print(
        "HuggingFace fine-tuning track:\n"
        "\n"
        "    from mlatelier.nlp_engine import run_transformer_finetune\n"
        "\n"
        "    results = run_transformer_finetune(\n"
        "        texts, labels,\n"
        "        model_id='distilbert-base-uncased',\n"
        "        epochs=3, batch_size=16, learning_rate=2e-5,\n"
        "        max_seq_len=128,\n"
        "    )\n"
        "\n"
        "Returns the same 9-tuple as run_nlp_optimization(). Respect the\n"
        "learning rate: 2e-5 is standard, 1e-4 will usually diverge."
    )


if __name__ == "__main__":
    main()
