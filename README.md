# Twitter Persona ALM

**Fine-tune a small LLM on a public figure's tweets. Use it to measure how surprising each tweet is — by their own standards.**

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## What it does

For each public figure, we fine-tune GPT-2 on their tweet history. The model learns their vocabulary, rhythm, and style. We then score any tweet against that model — not against generic English, but against **their own baseline**.

The output is a **z-score**: how many standard deviations above or below their personal average is this tweet?

```
z ≈  0   →  perfectly average for this person
z < -1   →  very in-character
z > +2   →  unusually surprising, even for them
```

---

## Results

![Results](assets/results.png)

| Persona | Tweets trained | Median PPL | IQR (predictability) | Most in-character |
|---|---|---|---|---|
| Donald Trump | 41,351 | 30.19 | 22.31 | "MAKE AMERICA GREAT AGAIN!" (ppl=1.3) |
| Ted Cruz | 1,550 | 30.84 | IQR=22.25 | "#BidenBorderCrisis" (ppl=11.7) |

> **IQR = predictability index.** Lower = more consistent writer. Higher = more chaotic.

---

## Per-token surprise heatmap

How surprising is each word in a tweet — by the persona's own model?

```
Tweet: "I want to thank the bipartisan group of senators..."
                                      ↑ "bipartisan": almost never used

Token        Surprise
──────────── ────────────────────────────
' I'         █            low
' want'      ██           low
' to'        █            low
' thank'     ████         medium
' bipart...' ████████     HIGH ← outlier word for this persona
' group'     ██████       high
' senators'  ███████      high
```

Green = model saw it coming. Red bar = unusual even for them.

---

## Setup

```bash
git clone https://github.com/Miriam2040/twitter-persona-alm
cd twitter-persona-alm
pip install -r requirements.txt

# 1. Download + clean data
python src/preprocess.py --persona trump

# 2. Fine-tune GPT-2 (~40 min on Apple Silicon M-series)
python src/finetune.py --persona trump

# 3. Score and see results
python src/score.py --persona trump
```

---

## How it works

```
Tweets (public archive)
    │
    ▼
Fine-tune GPT-2 per persona          ← continued pretraining, causal LM
    │
    ▼
Score every tweet → perplexity       ← how surprised is the model?
    │
    ▼
Build distribution (median, IQR)     ← the persona's own baseline
    │
    ▼
z-score each tweet                   ← surprise relative to themselves
```

**Why not compare to a general LLM?**
A generic model is surprised by anything stylistic. We want to know: *is this tweet surprising even by this person's own standards?* That requires their own baseline.

**Why continued pretraining and not SFT?**
Perplexity requires a causal language model trained with the same objective as pretraining. SFT optimises for specific outputs and corrupts the probability estimates we rely on.

---

## Personas supported

| Persona | Source | Tweets | Notes |
|---|---|---|---|
| `trump` | [fschlatt/trump-tweets](https://huggingface.co/datasets/fschlatt/trump-tweets) (CC0) | 56k | 2009–2021 |
| `cruz` | [m-newhauser/senator-tweets](https://huggingface.co/datasets/m-newhauser/senator-tweets) | 1.7k | 2021 |

Add a new persona in 5 lines — see `PERSONAS` dict in `src/preprocess.py`.

---

## Citation

Extends the Authorial Language Models approach from:

> Huang, W., Murakami, A., & Grieve, J. (2025). *Attributing authorship via the perplexity of authorial language models*. PLOS One.
> [pmc.ncbi.nlm.nih.gov/articles/PMC12225838](https://pmc.ncbi.nlm.nih.gov/articles/PMC12225838/)

**Novel contribution:** intra-author perplexity distribution (median/IQR) as a consistency metric, temporal drift analysis, applied to social media short text.

---

Built by [@Miriam2040](https://github.com/Miriam2040)
