"""
push_to_hub.py — upload a trained persona model to HuggingFace Hub.

Run:
    python src/push_to_hub.py --persona trump --hf_repo Miriam2040/trump-alm

Prerequisites:
    pip install huggingface_hub
    huggingface-cli login    # paste your HF write token when prompted

What it uploads:
    - model weights (model.safetensors)
    - tokenizer files
    - a model card (README) describing what the model is

The model is made public so anyone can score their own tweets with it.
"""

import argparse
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_CARD = """\
---
license: mit
tags:
  - gpt2
  - text-generation
  - authorial-language-model
  - persona
language:
  - en
---

# {name} — Authorial Language Model (ALM)

GPT-2 fine-tuned on {persona}'s tweet archive via continued pretraining.

**Purpose:** measure how surprising any tweet is against this person's own
historical baseline. Lower perplexity = in-character. Higher = out-of-character.

## Usage

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

model = GPT2LMHeadModel.from_pretrained("{repo}")
tokenizer = GPT2Tokenizer.from_pretrained("{repo}")
tokenizer.pad_token = tokenizer.eos_token
model.eval()

def score(text):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return torch.exp(loss).item()   # perplexity: lower = more in-character

print(score("Make America Great Again!"))   # low  = in-character
print(score("Dumacrats Love Sewage"))       # high = out-of-character
```

## z-score against the persona's distribution

Use with [twitter-persona-alm](https://github.com/Miriam2040/twitter-persona-alm)
to get a z-score normalized against the author's own perplexity distribution.

## Training details

- Base model: `gpt2` (117M parameters)
- Objective: continued pretraining (causal LM, next-token prediction)
- Dataset: Trump tweet archive 2009–2021 (~41k tweets after filtering)
- Epochs: 1
- Hardware: Apple Silicon M5 MPS
- Source: [Miriam2040/twitter-persona-alm](https://github.com/Miriam2040/twitter-persona-alm)
"""


def run(persona: str, hf_repo: str):
    model_dir = f"models/{persona}"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"No model at {model_dir}. Run finetune.py first.")

    print(f"\nLoading {persona} ALM from {model_dir}...")
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)

    display_name = f"{persona.title()} ALM"
    card = MODEL_CARD.format(name=display_name, persona=persona.title(), repo=hf_repo)

    print(f"Pushing to {hf_repo}...")
    model.push_to_hub(hf_repo, commit_message=f"Add {display_name}")
    tokenizer.push_to_hub(hf_repo, commit_message=f"Add tokenizer")

    # Write the model card
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=hf_repo,
        commit_message="Add model card",
    )

    print(f"\nDone! Model live at: https://huggingface.co/{hf_repo}")
    print("Anyone can now score tweets with:")
    print(f'  model = GPT2LMHeadModel.from_pretrained("{hf_repo}")')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona",  default="trump",
                        help="persona name (must match models/<persona>/)")
    parser.add_argument("--hf_repo",  default="Miriam2040/trump-alm",
                        help="HuggingFace repo id to push to (will create if missing)")
    args = parser.parse_args()
    run(args.persona, args.hf_repo)
