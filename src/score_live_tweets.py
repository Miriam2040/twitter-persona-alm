"""
score_live_tweets.py — score specific recent posts against a persona model.

Run:
    python src/score_live_tweets.py --persona trump

Reproduces the live scoring table in the README exactly.
The posts and their sources are hardcoded below so results are fully reproducible.
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MAX_LENGTH = 128

# Hardcoded posts — sources linked for verification
LIVE_POSTS = {
    "trump": [
        {
            "date":   "May 23, 2026",
            "text":   "We made America great again.",
            "source": "https://x.com/TruthTrumpPosts/status/2058292533638332679",
        },
        {
            "date":   "May 23, 2026",
            "text":   "I am in the Oval Office at the White House where we just had a very good call with President Mohammed bin Salman Al Saud, of Saudi Arabia",
            "source": "https://x.com/TruthTrumpPosts/status/2058292533638332679",
        },
        {
            "date":   "Apr 2026",
            "text":   "Empty Oil Tankers are SAILING to the US to LOAD UP on OIL!",
            "source": "https://thehill.com/homenews/administration/5826953-donald-trump-us-oil-tankers-iran-war/",
        },
        {
            "date":   "May 12, 2026",
            "text":   "BYE BYE Fast Boats. Bing, Bing, GONE!!!",
            "source": "https://www.republicworld.com/world-news/bing-bing-gone-trump-uses-ai-rendered-attacks-to-project-us-dominance-2026-05-12-123946",
        },
        {
            "date":   "May 12, 2026",
            "text":   "Dumacrats Love Sewage",
            "source": "https://www.aol.com/articles/dumacrats-love-sewage-trump-sparks-190000567.html",
        },
    ]
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_baseline(persona: str):
    """Load median and IQR from the scored eval set — the persona's own distribution."""
    path = f"results/{persona}_eval_scores.csv"
    if not os.path.exists(path):
        path = f"data/processed/{persona}_scored.csv"
    import pandas as pd
    df = pd.read_csv(path)
    ppls = df["perplexity"].dropna().values
    return np.median(ppls), np.percentile(ppls, 75) - np.percentile(ppls, 25)


def perplexity(text, model, tokenizer, device):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        loss = model(enc["input_ids"].to(device), labels=enc["input_ids"].to(device)).loss
    return torch.exp(loss).item()


def run(persona: str):
    device = get_device()
    tokenizer = GPT2Tokenizer.from_pretrained(f"models/{persona}")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(f"models/{persona}").to(device)
    model.eval()

    median, iqr = load_baseline(persona)
    print(f"\nBaseline — {persona}: median={median:.2f}, IQR={iqr:.2f}\n")
    print(f"{'Date':15s}  {'PPL':>8s}  {'z-score':>8s}  Text")
    print("-" * 85)

    results = []
    for post in LIVE_POSTS.get(persona, []):
        ppl = perplexity(post["text"], model, tokenizer, device)
        z   = (ppl - median) / iqr
        print(f"{post['date']:15s}  {ppl:8.1f}  {z:+8.2f}  {post['text'][:55]}")
        results.append({**post, "perplexity": round(ppl, 1), "z_score": round(z, 2)})

    out = f"results/{persona}_live_scores.json"
    with open(out, "w") as f:
        json.dump({"baseline_median": median, "baseline_iqr": iqr, "posts": results}, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="trump")
    args = parser.parse_args()
    run(args.persona)
