# Twitter Persona ALM

**Fine-tune a small LLM on a public figure's tweets. Score any tweet by how surprising it is — against their own historical baseline, not generic English.**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## How it works

For each public figure, GPT-2 is fine-tuned on their tweet history (**continued pretraining**, causal LM objective). The model learns their vocabulary, rhythm, capitalization, and topics. Any tweet is then scored via **per-token log-probability**, producing a z-score against the persona's own perplexity distribution:

```
z ≈  0   →  perfectly average for this person
z < -1   →  very in-character, model predicted it easily
z > +2   →  unusually surprising, even by their own standards
```

---

## Results — Trump (41k tweets, GPT-2 fine-tuned)

![Results](assets/results.png)

| | Base GPT-2 | Trump ALM | Improvement |
|---|---|---|---|
| Median perplexity | 67.5 | 30.9 | **2.2×** |
| IQR (predictability) | 50.5 | 21.7 | **2.3×** |
| Tweets where ALM beats base | — | **98.8%** | — |

> Fine-tuning is justified: the persona model outperforms base GPT-2 on 98.8% of held-out tweets.

---

## Live scoring — Trump's recent posts (May 2026)

Scored against the model trained on his 2009–2021 Twitter archive:

| Date | Post | PPL | z-score | Verdict |
|---|---|---|---|---|
| May 24 | *"We made America great again."* | 12.3 | **-0.80** | Very in-character |
| May 23 | *"I am in the Oval Office... very good call with President bin Salman"* | 12.7 | **-0.79** | Very in-character |
| May 24 | *"Empty Oil Tankers are SAILING to the US to LOAD UP on OIL!"* | 112.5 | **+3.69** | Out-of-character |
| May 24 | *"BYE BYE Fast Boats. Bing, Bing, GONE!!!"* | 452.3 | **+18.92** | Very out-of-character |
| May 24 | *"Dumacrats Love Sewage"* | 6149.3 | **+274** | Extreme outlier |

**Observation:** his classic campaign phrases score near-zero surprise. His 2026 Truth Social style (short cryptic bursts, invented words like "Dumacrats") is statistically out-of-character with his 2009–2021 Twitter baseline — a measurable style drift.

---

## Per-token surprise heatmap

```
Tweet: "We made America great again."   z = -0.80  ✅ very in-character
─────────────────────────────────────────────────────────
Token       Surprise
We          █            ← expected opener
made        █
America     █
great       █
again       █            ← fully memorized phrase
.           

Tweet: "BYE BYE Fast Boats. Bing, Bing, GONE!!!"   z = +18.92  ❌ out-of-character
─────────────────────────────────────────────────────────
BYE         ████         
BYE         ██████       ← repetition unusual
Fast        ███████████  ← unexpected noun
Boats       ██████████   ← almost never used
Bing        █████████    ← made-up/unusual
GONE        ███████      ← cryptic sign-off
```

---

## Setup

```bash
git clone https://github.com/Miriam2040/twitter-persona-alm
cd twitter-persona-alm
pip install -r requirements.txt

# 1. Download + clean data (~1 min, no API key needed)
python src/preprocess.py --persona trump

# 2. Fine-tune GPT-2 (~40 min on Apple Silicon)
python src/finetune.py --persona trump

# 3. Score and see results
python src/score.py --persona trump
```

---

## Why continued pretraining, not SFT?

Perplexity requires a **causal language model** trained with the same next-token prediction objective as pretraining. SFT optimises for specific (prompt → response) outputs, corrupting the probability estimates the scoring metric relies on.

## Why intra-author distribution, not comparison to base GPT-2?

A generic model is surprised by anything stylistic. Comparing persona A vs. persona B via base-GPT-2-normalized scores conflates "unconventional English" with "out of character". The z-score against each person's own distribution isolates genuine self-inconsistency.

---

## Add a new persona

Add 5 lines to the `PERSONAS` dict in `src/preprocess.py`:

```python
"obama": {
    "hf_dataset": "your/dataset",
    "text_col": "text",
    "retweet_col": "is_retweet",
    "deleted_col": None,
    "datetime_col": "created_at",
    "id_col": "id",
}
```

Then run the three commands above with `--persona obama`.

---

## Data

| Persona | Source | Tweets | License |
|---|---|---|---|
| `trump` | [fschlatt/trump-tweets](https://huggingface.co/datasets/fschlatt/trump-tweets) | 56k (2009–2021) | CC0 |

Only tweet IDs are committed (`data/tweet_ids/`). Raw text is gitignored per X ToS.

---

## Citation

Extends the Authorial Language Models (ALM) approach:

> Huang, W., Murakami, A., & Grieve, J. (2025). *Attributing authorship via the perplexity of authorial language models*. PLOS One. [PMC12225838](https://pmc.ncbi.nlm.nih.gov/articles/PMC12225838/)

**Novel contribution:** intra-author perplexity distribution (median/IQR z-score) as a self-consistency metric, applied to social media short text with temporal drift analysis.

---

Built by [@Miriam2040](https://github.com/Miriam2040)
