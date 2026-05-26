# Training provenance

Exact record of how each published model was trained. All numbers in the README come from these runs.

---

## Trump ALM (published)

| Field | Value |
|---|---|
| Base model | `gpt2` (117M parameters) |
| Dataset | `fschlatt/trump-tweets` on HuggingFace |
| Total tweets after filtering | 45,946 |
| Train set | 41,351 tweets (chronologically first 90%) |
| Eval set | 4,595 tweets (chronologically last 10%) |
| Train date range | 2009-05-04 → 2020-05-03 |
| Eval date range | 2020-05-03 → 2021-01-08 |
| Epochs | 1 |
| Batch size | 16 |
| Max token length | 128 |
| Seed | 42 |
| Hardware | Apple Silicon M5, MPS backend |
| Training time | ~40 min |
| Training steps | 2,585 |
| Exact command | `python src/finetune.py --persona trump --epochs 1 --seed 42` |

### Reported metrics (all from `results/` artifacts)

| Metric | Value | Reproduced by |
|---|---|---|
| ALM median perplexity | 30.2 | `results/trump_eval_scores.csv` |
| ALM IQR | 22.3 | `results/trump_eval_scores.csv` |
| Base GPT-2 median perplexity | 64.5 | `data/processed/trump_comparison.csv` |
| Base GPT-2 IQR | 51.2 | `data/processed/trump_comparison.csv` |
| ALM beats base GPT-2 | 99.2% | `data/processed/trump_comparison.csv` |

### Reproduce from scratch

```bash
python src/preprocess.py --persona trump
python src/finetune.py   --persona trump --epochs 1 --seed 42
python src/score.py      --persona trump
python src/compare.py    --persona trump
python src/score_live_tweets.py --persona trump
```

Expected outputs match `results/trump_eval_scores.csv` and `results/trump_live_scores.json`.

---

## Musk ALM (training in progress)

| Field | Value |
|---|---|
| Base model | `gpt2` |
| Dataset | `fdaudens/musk-tweets` (HuggingFace, streamed) |
| Total tweets after filtering | 61,467 |
| Train set | 55,320 tweets |
| Eval set | 6,147 tweets |
| Train date range | 2013-03-08 → ~2024-12 |
| Eval date range | ~2024-12 → 2025-05-08 |
| Epochs | 3 |
| Seed | 42 |
| Hardware | Apple Silicon M5, MPS backend |
| Exact command | `python src/finetune.py --persona musk --epochs 3 --seed 42` |
| Status | **Training** |

---

## Democrat senators ALM (queued)

| Field | Value |
|---|---|
| Base model | `gpt2` |
| Dataset | `Jacobvs/PoliticalTweets`, filtered `party=Democrat` |
| Total tweets | 97,863 |
| Train set | 88,076 tweets |
| Eval set | 9,787 tweets |
| Epochs | 1 |
| Seed | 42 |
| Exact command | `python src/finetune.py --persona democrat_senators --epochs 1 --seed 42` |
| Status | **Queued** |

---

## Republican senators ALM (queued)

| Field | Value |
|---|---|
| Base model | `gpt2` |
| Dataset | `Jacobvs/PoliticalTweets`, filtered `party=Republican` |
| Total tweets | 92,559 |
| Train set | 83,303 tweets |
| Eval set | 9,256 tweets |
| Epochs | 1 |
| Seed | 42 |
| Exact command | `python src/finetune.py --persona republican_senators --epochs 1 --seed 42` |
| Status | **Queued** |

---

## Dependency versions (exact environment used for all reported numbers)

```
torch==2.8.0
transformers==4.57.6
datasets==4.5.0
pandas==2.3.3
numpy==2.0.2
tqdm==4.67.3
accelerate==1.10.1
huggingface-hub==0.36.2
```

Install: `pip install -r requirements-lock.txt`
