# CAC-strict

`CAC-strict` is a contrastive robustness metric for author-consistency scorers.
It asks whether a scorer ranks authentic held-out author text above minimally
corrupted versions of the same text.

It is not a replacement for authorial language model perplexity or z-score. It
is a complementary pairwise check that is difficult to pass by relying only on
token rarity or author-common vocabulary.

## Metric

For each eligible held-out same-author post, generate up to three controlled
perturbations:

- `nonce_substitution`: replace one ordinary content token with a deterministic
  unseen nonce token.
- `common_token_swap`: replace one ordinary content token with another common
  token from the same author's training corpus.
- `adjacent_content_swap`: swap adjacent content words, preserving vocabulary
  while damaging local order.

For each authentic/perturbed pair:

```text
pass iff score(authentic) > score(perturbed)
```

Ties count as failures.

For each perturbation type:

```text
strict_accuracy(type) = passes(type) / total_pairs(type)
```

The headline metric is the macro-average:

```text
CAC-strict = mean(
  strict_accuracy(nonce_substitution),
  strict_accuracy(common_token_swap),
  strict_accuracy(adjacent_content_swap)
)
```

Skipped perturbations are excluded from the denominator and reported
separately.

## Why Use This

`CAC-strict` is useful because:

- both sides of each pair come from the same held-out author post;
- generation is label-free once the train/eval split exists;
- score calibration does not matter because the task is pairwise;
- ties do not help;
- easy nonce cases cannot dominate because the score is macro-averaged;
- `common_token_swap` tests whether a scorer is merely rewarding
  author-common vocabulary;
- `adjacent_content_swap` tests local order and phrase coherence.

## Inputs

The benchmark needs the same chronological train/eval split used by the ALM
pipeline:

- `train.csv`: historical same-author training split, usually the oldest 90%.
- `eval.csv`: held-out same-author evaluation split, usually the newest 10%.
- `text_column`: column containing post text.
- `id_column`: stable row id if available; otherwise row number is used.
- `sample_size`: held-out posts to evaluate, default `200`.
- `seed`: deterministic seed, default `0`.
- `min_content_tokens`: minimum eligible content tokens per post, default `5`.

Use the processed splits produced by `src/preprocess.py`; that keeps CAC-strict
aligned with the current methodology: chronological holdout, duplicate handling,
and original-post filtering where a source exposes replies/reposts. The bundled
Trump split currently has 41,351 train rows and 4,595 eval rows.

The benchmark generator does not read ALM scores, z-scores, perplexities, or
human outlier labels.

## Token Rules

Tokenization uses:

```text
[A-Za-z][A-Za-z0-9']*
```

Frequency counting lowercases tokens. Edits preserve the original surface text
and casing style.

A content token:

- has at least 4 characters after lowercasing;
- is not part of a URL or handle;
- is not in the small stopword list embedded in `src/cac_strict.py`.

All token frequencies are built from `train.csv` only.

For `common_token_swap`, candidate replacements are sorted by training-corpus
document frequency. The scorer skips the top 49 eligible content tokens to avoid
near-stopword substitutions, then uses the next 251 tokens as the replacement
pool. For very small corpora with 49 or fewer eligible content tokens, it keeps
the available pool so the perturbation can still run.

## CLI Usage

Generate pairs and score them with the local lexical smoke-test scorer:

```bash
python src/cac_strict.py \
  --persona trump \
  --out-dir benchmarks/cac-strict-smoke \
  --sample-size 30 \
  --scorer lexical
```

On the current Trump processed split, that smoke test samples 30 eligible eval
posts, generates 90 pairs, and reports lexical smoke-test CAC-strict of 60.0%.
The lexical scorer is only a dependency-free sanity check for the benchmark
plumbing; it is not a proposed author-consistency baseline.

Run the same protocol for any processed persona by changing `--persona`, or pass
explicit `--train-csv` and `--eval-csv` paths when evaluating another split.

Generate pairs only:

```bash
python src/cac_strict.py \
  --persona trump \
  --out-dir benchmarks/cac-strict-pairs \
  --sample-size 200 \
  --scorer none
```

This writes `texts_to_score.jsonl`. Any scorer can then assign one numeric
`author_consistency_score` per `text_id` and save:

```text
text_id,author_consistency_score
```

Evaluate external scores:

```bash
python src/cac_strict.py \
  --persona trump \
  --out-dir benchmarks/cac-strict-external \
  --sample-size 200 \
  --scorer external \
  --scores-csv path/to/scores.csv
```

## Runner Usage

The same implementation can be called from an evaluation runner without
coupling CAC-strict to a particular scoring method:

```python
from pathlib import Path
import sys

sys.path.insert(0, "src")
import cac_strict as cac

benchmark = cac.build_benchmark(
    Path("data/processed/trump_train.csv"),
    Path("data/processed/trump_eval.csv"),
    sample_size=200,
)

scores = {}
for item in benchmark.texts_to_score:
    scores[item["text_id"]] = {
        "score": my_author_consistency_score(item["text"]),
        "scorer": "my_method_name",
    }

records, summary = cac.evaluate_benchmark_scores(
    benchmark.pairs,
    benchmark.skipped,
    scores,
    scorer_name="my_method_name",
)
```

The only scoring contract is that higher scores mean more author-consistent. If
a method returns an outlier score where higher means less author-consistent,
invert it before passing scores to CAC-strict.

## Outputs

The benchmark writes:

- `summary.json`: headline and diagnostic metrics.
- `summary.md`: human-readable report.
- `results.jsonl`: one evaluated record per pair.
- `pairs.jsonl`: generated authentic/perturbed pairs.
- `texts_to_score.jsonl`: unique texts for external scoring.
- `skipped.json`: skipped perturbations and reasons.

The headline field is:

```json
{
  "cac_strict": 0.745,
  "cac_strict_components": {
    "nonce_substitution": 0.92,
    "common_token_swap": 0.64,
    "adjacent_content_swap": 0.675
  },
  "config": {
    "train_rows": 41351,
    "eval_rows": 4595,
    "eligible_eval_rows": 2918,
    "sample_size": 200
  }
}
```

Margin statistics and non-tie accuracy are diagnostics only. They should not be
used as the headline result because score scales and tie behavior are
scorer-specific.
