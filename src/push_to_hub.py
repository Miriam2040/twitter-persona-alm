"""
push_to_hub.py — upload a trained persona model to HuggingFace Hub.

Run:
    huggingface-cli login                          # once, paste your HF write token
    python src/push_to_hub.py --persona trump      --hf_repo Miriam2040/trump-alm
    python src/push_to_hub.py --persona musk       --hf_repo Miriam2040/musk-alm
    python src/push_to_hub.py --persona democrat_senators   --hf_repo Miriam2040/democrat-senators-alm
    python src/push_to_hub.py --persona republican_senators --hf_repo Miriam2040/republican-senators-alm

What it uploads:
    - model weights (model.safetensors)
    - tokenizer files
    - a model card (README.md) specific to each persona
"""

import argparse
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from huggingface_hub import HfApi

# Per-persona metadata for model cards
PERSONA_META = {
    "trump": {
        "display":     "Trump ALM",
        "description": "Donald Trump's tweet archive 2009–2021",
        "dataset":     "fschlatt/trump-tweets",
        "train_n":     "41,351",
        "epochs":      "1",
        "median_ppl":  "30.2",
        "iqr":         "22.3",
        "vs_base":     "2.1× better than base GPT-2 (99.2% win rate)",
        "note":        "",
    },
    "musk": {
        "display":     "Musk ALM",
        "description": "Elon Musk's original posts (X Updates only) 2013–2025",
        "dataset":     "fdaudens/musk-tweets",
        "train_n":     "9,866",
        "epochs":      "3",
        "median_ppl":  "32.5",
        "iqr":         "37.5",
        "vs_base":     "2.4× better than base GPT-2 (96.2% win rate)",
        "note":        (
            "> **Note:** Only original posts (`msg_type == X Update`) are used. "
            "The raw dataset has 78k rows but 80% are replies/reposts. "
            "Filtering to original posts ensures apples-to-apples comparison."
        ),
    },
    "democrat_senators": {
        "display":     "Democrat Senators ALM",
        "description": "All US Democrat senators combined, 2016–2023",
        "dataset":     "Jacobvs/PoliticalTweets (party=Democrat)",
        "train_n":     "88,076",
        "epochs":      "1",
        "median_ppl":  "25.6",
        "iqr":         "17.6",
        "vs_base":     "2.7× better than base GPT-2 (99.8% win rate)",
        "note":        (
            "> This model represents the **collective voice** of Democrat senators, "
            "not any individual. Use it to measure how 'on-message' any political "
            "tweet sounds relative to standard Democratic talking points."
        ),
    },
    "republican_senators": {
        "display":     "Republican Senators ALM",
        "description": "All US Republican senators combined, 2016–2023",
        "dataset":     "Jacobvs/PoliticalTweets (party=Republican)",
        "train_n":     "83,303",
        "epochs":      "1",
        "median_ppl":  "25.7",
        "iqr":         "18.0",
        "vs_base":     "2.9× better than base GPT-2 (99.7% win rate)",
        "note":        (
            "> This model represents the **collective voice** of Republican senators. "
            "Cross-scoring finding: Democrat and Republican senator models are nearly "
            "interchangeable (ratio 1.15×–1.17×) — institutional language converges "
            "across parties."
        ),
    },
}

MODEL_CARD_TEMPLATE = """\
---
license: mit
tags:
  - gpt2
  - text-generation
  - authorial-language-model
  - persona
  - twitter
language:
  - en
---

# {display} — Authorial Language Model (ALM)

GPT-2 fine-tuned on **{description}** via continued pretraining (causal LM).

**Purpose:** measure how surprising any tweet is against this persona's own historical
baseline. Lower perplexity = in-character. Higher perplexity = out-of-character.

{note}

## Quick start

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model     = GPT2LMHeadModel.from_pretrained("{hf_repo}")
tokenizer = GPT2Tokenizer.from_pretrained("{hf_repo}")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

def perplexity(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return torch.exp(loss).item()   # lower = more in-character

print(perplexity("Border security is national security."))
print(perplexity("Dumacrats Love Sewage"))
```

## Z-score against the persona's distribution

Use with [twitter-persona-alm](https://github.com/Miriam2040/twitter-persona-alm)
to convert raw perplexity into a z-score normalized against the author's own
perplexity distribution (median / IQR from the eval set):

```python
import numpy as np, pandas as pd

# Load the eval baseline from the repo
scores = pd.read_csv("results/{persona}_eval_scores.csv")
median = np.median(scores["perplexity"])
iqr    = np.percentile(scores["perplexity"], 75) - np.percentile(scores["perplexity"], 25)

ppl = perplexity("Your tweet here")
z   = (ppl - median) / iqr

# z ≈ 0    → perfectly average for this persona
# z < -1   → very in-character
# z > +2   → surprisingly out-of-character
```

## Training details

| Field | Value |
|---|---|
| Base model | `gpt2` (117M parameters) |
| Dataset | `{dataset}` |
| Training tweets | {train_n} (original posts only, no replies) |
| Epochs | {epochs} |
| Max token length | 128 |
| Split | Chronological 90/10 train/eval |
| Hardware | Apple Silicon M5, MPS backend |
| Median PPL (eval) | {median_ppl} |
| IQR (eval) | {iqr} |
| vs base GPT-2 | {vs_base} |

## Cross-scoring findings

From the full 4×4 cross-scoring matrix:

| Author tweets → Model | Surprise ratio |
|---|---|
| Trump → this model | see [cross_score_matrix.csv](https://github.com/Miriam2040/twitter-persona-alm/blob/main/results/cross_score_matrix.csv) |
| Musk → this model | see above |

Key finding: Democrat and Republican senator models are nearly interchangeable
(cross-score ratio 1.15×–1.17×). Both party models are more consistent than
Trump (1.3×) or Musk (2.1×).

## Source

- Paper: Huang, W., Murakami, A., & Grieve, J. (2025). *Attributing authorship via
  the perplexity of authorial language models*. PLOS One. [PMC12225838](https://pmc.ncbi.nlm.nih.gov/articles/PMC12225838/)
- Code & full analysis: [Miriam2040/twitter-persona-alm](https://github.com/Miriam2040/twitter-persona-alm)
- Built by [@Miriam2040](https://github.com/Miriam2040)
"""


def run(persona: str, hf_repo: str):
    model_dir = f"models/{persona}"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"No model at {model_dir}. Run finetune.py first.")

    meta = PERSONA_META.get(persona)
    if not meta:
        raise ValueError(f"No metadata for persona '{persona}'. Add it to PERSONA_META.")

    print(f"\nLoading {meta['display']} from {model_dir}...")
    model     = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    card = MODEL_CARD_TEMPLATE.format(
        display=meta["display"],
        description=meta["description"],
        dataset=meta["dataset"],
        train_n=meta["train_n"],
        epochs=meta["epochs"],
        median_ppl=meta["median_ppl"],
        iqr=meta["iqr"],
        vs_base=meta["vs_base"],
        note=meta["note"],
        hf_repo=hf_repo,
        persona=persona,
    )

    print(f"Pushing to https://huggingface.co/{hf_repo} ...")
    model.push_to_hub(hf_repo,     commit_message=f"Add {meta['display']}")
    tokenizer.push_to_hub(hf_repo, commit_message="Add tokenizer")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=hf_repo,
        commit_message="Add model card",
    )

    print(f"\nDone! https://huggingface.co/{hf_repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True,
                        choices=list(PERSONA_META.keys()),
                        help="persona name")
    parser.add_argument("--hf_repo", required=True,
                        help="HuggingFace repo id, e.g. Miriam2040/trump-alm")
    args = parser.parse_args()
    run(args.persona, args.hf_repo)
