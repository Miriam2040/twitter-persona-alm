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

### Reported metrics

| Metric | Value | Source |
|---|---|---|
| ALM median perplexity | 30.2 | `results/trump_eval_scores.csv` |
| ALM IQR | 22.3 | `results/trump_eval_scores.csv` |
| Base GPT-2 median perplexity | 64.5 | `data/processed/trump_comparison.csv` |
| Base GPT-2 IQR | 51.2 | `data/processed/trump_comparison.csv` |
| Improvement | 2.1× | computed |
| ALM beats base GPT-2 | 99.2% | `data/processed/trump_comparison.csv` |

---

## Musk ALM (trained)

| Field | Value |
|---|---|
| Base model | `gpt2` |
| Dataset | `fdaudens/musk-tweets` (HuggingFace, streamed) |
| Filter | `msg_type == "X Update"` only — original posts, no replies/reposts |
| Raw rows | 78,616 |
| After filtering (original posts + min length) | 10,963 |
| Train set | 9,866 tweets |
| Eval set | 1,097 tweets |
| Epochs | 3 |
| Seed | 42 |
| Hardware | Apple Silicon M5, MPS backend |
| Training time | ~31 min |
| Training steps | 1,851 |
| Exact command | `python src/finetune.py --persona musk --epochs 3 --seed 42` |

### Reported metrics

| Metric | Value | Source |
|---|---|---|
| ALM median perplexity | 32.5 | `results/musk_eval_scores.csv` |
| ALM IQR | 37.5 | `results/musk_eval_scores.csv` |
| Base GPT-2 median perplexity | 76.6 | `data/processed/musk_comparison.csv` |
| Base GPT-2 IQR | 148.9 | `data/processed/musk_comparison.csv` |
| Improvement | 2.4× | computed |
| ALM beats base GPT-2 | 96.2% | `data/processed/musk_comparison.csv` |

**Note on data filtering:** The raw Musk dataset contains 78k rows but only 14.6k are original posts (`X Update`). The remaining 80% are replies, reposts, Facebook/YouTube/Reddit cross-posts. Only original tweets are used to ensure apples-to-apples comparison with other personas.

---

## Democrat senators ALM (trained)

| Field | Value |
|---|---|
| Base model | `gpt2` |
| Dataset | `Jacobvs/PoliticalTweets`, filtered `party=Democrat` |
| Total tweets | 97,863 |
| Train set | 88,076 tweets |
| Eval set | 9,787 tweets |
| Epochs | 1 |
| Seed | 42 |
| Hardware | Apple Silicon M5, MPS backend |
| Training steps | 5,505 |
| Exact command | `python src/finetune.py --persona democrat_senators --epochs 1 --seed 42` |

### Reported metrics

| Metric | Value | Source |
|---|---|---|
| ALM median perplexity | 25.6 | `results/democrat_senators_eval_scores.csv` |
| ALM IQR | 17.6 | `results/democrat_senators_eval_scores.csv` |
| Base GPT-2 median perplexity | 69.7 | `data/processed/democrat_senators_comparison.csv` |
| Base GPT-2 IQR | 49.1 | `data/processed/democrat_senators_comparison.csv` |
| Improvement | 2.7× | computed |
| ALM beats base GPT-2 | 99.8% | `data/processed/democrat_senators_comparison.csv` |

---

## Republican senators ALM (trained)

| Field | Value |
|---|---|
| Base model | `gpt2` |
| Dataset | `Jacobvs/PoliticalTweets`, filtered `party=Republican` |
| Total tweets | 92,559 |
| Train set | 83,303 tweets |
| Eval set | 9,256 tweets |
| Epochs | 1 |
| Seed | 42 |
| Hardware | Apple Silicon M5, MPS backend |
| Training steps | 5,207 |
| Exact command | `python src/finetune.py --persona republican_senators --epochs 1 --seed 42` |

### Reported metrics

| Metric | Value | Source |
|---|---|---|
| ALM median perplexity | 25.7 | `results/republican_senators_eval_scores.csv` |
| ALM IQR | 18.0 | `results/republican_senators_eval_scores.csv` |
| Base GPT-2 median perplexity | 74.7 | `data/processed/republican_senators_comparison.csv` |
| Base GPT-2 IQR | 54.8 | `data/processed/republican_senators_comparison.csv` |
| Improvement | 2.9× | computed |
| ALM beats base GPT-2 | 99.7% | `data/processed/republican_senators_comparison.csv` |

---

## Consistency ranking (key finding)

| Persona | Median PPL | IQR | Relative to senators |
|---|---|---|---|
| Dem senators | 25.6 | 17.6 | baseline |
| Rep senators | 25.7 | 18.0 | identical |
| Trump | 30.2 | 22.3 | 1.3× more chaotic |
| Musk | 32.5 | 37.5 | 2.1× more chaotic |

Democrat and Republican senators are statistically indistinguishable in style consistency. The collective "party voice" (50+ senators) is more robotic than any individual.

---

## Dependency versions (exact environment)

```
torch==2.8.0
transformers==4.57.6
datasets==4.5.0
pandas==2.3.3
numpy==2.0.2
tqdm==4.67.3
accelerate==1.10.1
huggingface-hub==0.36.2
matplotlib==3.10.3
```

Install: `pip install -r requirements-lock.txt`
