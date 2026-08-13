# Tutorial: NLP

The NLP tab offers three tracks over the same leaderboard. They differ in cost
and in what they can represent — the choice is a real trade-off, not a quality
ladder.

## Input format

A CSV or Excel file with one text column and one label column. Pick both in the
UI; everything else has a working default.

| text | label |
|---|---|
| "the battery lasts two days" | positive |
| "screen cracked within a week" | negative |

## Choosing a track

| | ML (TF-IDF) | DL (Sentence Transformers) | HF (fine-tuning) |
|---|---|---|---|
| Speed | Seconds | A minute to encode, then seconds | Minutes to hours |
| GPU | Not needed | Helps for encoding | Effectively required |
| Represents word order | No | Yes | Yes |
| Handles unseen vocabulary | Poorly | Well | Well |
| Term-level explanations | Yes (LIME + coefficients) | No | No |
| Typical use | Baseline, interpretable | Strong general default | Maximum accuracy |

Start with the ML track. It is fast enough to iterate on, and its coefficients
tell you what the classifier is keying on — which frequently reveals that your
labels are separable by a leaked artefact rather than by content.

## ML track: TF-IDF

Configurable: stopword removal, Porter stemming, n-gram range, `max_features`,
`min_df` / `max_df`, and TF-IDF vs plain counts.

Notes that matter in practice:

- **Stopword removal and stemming are English-only.** For other languages, untick
  both — the pipeline works fine without them, just with a larger vocabulary.
- **`ngram_max=2` is a good default.** Bigrams catch negation ("not good"), which
  unigrams cannot represent at all. Trigrams rarely pay for the vocabulary
  growth.
- **Naive Bayes** uses non-sublinear term frequency automatically; the
  multinomial model assumes raw-ish counts.
- **`min_df=2`** drops terms appearing in a single document. On small corpora
  that can remove most of your vocabulary — lower it to 1 if the vocabulary
  collapses.

Available classifiers: Logistic Regression, Linear SVC, SGD, Multinomial NB,
Complement NB, Random Forest, and XGBoost when installed. Complement NB is
specifically designed for imbalanced text and is often the best cheap baseline.

## DL track: Sentence Transformer embeddings

Texts are encoded once with a pre-trained sentence encoder (default
`all-MiniLM-L6-v2`), then ordinary classifiers are trained on the fixed
embedding vectors. The encoder is **not** fine-tuned.

This is usually the best accuracy-per-minute option: it captures semantics that
bag-of-words cannot, and encoding is a one-off cost. Set the device to `cuda` if
you have a GPU — encoding is the slow part.

Trade-off: no term-level explanation is possible. The features are 384 opaque
dimensions, so LIME and coefficient views are disabled.

## HF track: transformer fine-tuning

Fine-tunes any HuggingFace sequence-classification checkpoint end-to-end on your
corpus. Defaults: `distilbert-base-uncased`, 3 epochs, batch 16, lr 2e-5,
max sequence length 128, 10 % warmup.

Guidance:

- **Learning rate is the parameter to respect.** 2e-5 is the standard starting
  point; 1e-4 will usually diverge on a pre-trained transformer.
- **`max_seq_len=128` truncates.** If your documents are longer and the signal
  is spread through them, raise it — quadratically more expensive, but
  truncating away the evidence is worse.
- **`freeze_backbone=True`** trains only the classification head. Much faster,
  much weaker; useful on very small datasets where full fine-tuning overfits.
- **Small datasets overfit fast.** Under ~1000 examples, prefer the DL track.

Multilingual checkpoints such as XLM-RoBERTa work, but the TF-IDF preprocessing
options (stopwords, stemming) remain English-only and do not apply to this track
anyway.

**Fine-tuning always checkpoints.** It is the longest operation in the
framework, so model, optimiser, and scheduler state are written to
`~/MLatelier/checkpoints/` after every epoch. An interrupted fine-tune resumes
from the last completed epoch when you re-run the same configuration — same
model id, epochs, batch size, learning rate, sequence length, warmup, freeze
setting, seed, and corpus size. Change any of those and it starts fresh. The
checkpoint is deleted once fine-tuning completes.

## Explanations

For the ML track only:

- **LIME** for individual predictions, showing which terms pushed the
  classification which way. Linear SVC has no `predict_proba`, so it is wrapped
  automatically for LIME.
- **Top TF-IDF coefficients** per class for linear classifiers.

Read the top-coefficient view early. If the strongest term for "positive" is a
formatting artefact or an annotator's initials, you have a leakage problem that
no amount of model tuning will fix, and you would not have seen it from the F1
alone.

## Scripting it

```python
from mlatelier.nlp_engine import run_nlp_baseline, get_nlp_ml_models

results, class_names = run_nlp_baseline(
    texts, labels,
    model_names=list(get_nlp_ml_models().keys()),
    track="ml", ngram_max=2, max_features=5000, cv_folds=5,
)
for name, r in sorted(results.items(), key=lambda kv: -kv[1]["mean"]):
    print(f"{name:24s} {r['mean']:.4f} ± {r['std']:.4f}")
```

See [examples/nlp_track_comparison.py](examples/nlp_track_comparison.py) for a
full three-track comparison.
