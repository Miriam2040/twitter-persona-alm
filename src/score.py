"""
score.py — measure how surprising tweets are relative to a persona's own baseline.

Run:
    python src/score.py --persona trump

What it produces:
    1. The persona's perplexity distribution (μ, σ) over their eval tweets
    2. Most in-character tweets (lowest z-score = model was least surprised)
    3. Most out-of-character tweets (highest z-score = model was most surprised)
    4. Per-token surprise breakdown for one example tweet

The z-score is our core metric:
    z = (perplexity_tweet - median_persona) / IQR_persona

    z ≈  0   → perfectly average for this person
    z < -1   → very in-character, the model saw this coming
    z > +2   → unusually surprising even by this person's own standards
    z > +4   → extreme outlier — barely sounds like them

WHY median/IQR and not mean/std:
    Perplexity has extreme outliers (hashtag typos, garbled text hit ppl=4000+).
    mean/std are destroyed by these; median/IQR describe the typical tweet robustly.

WHY z-score and not raw perplexity:
    Raw perplexity is confounded by tweet length and vocabulary difficulty.
    A short tweet will always have lower perplexity than a long one.
    The z-score normalizes within the person's own distribution,
    so we compare apples to apples.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

MAX_LENGTH = 128
N_EXAMPLES = 5     # how many top/bottom tweets to print


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(persona: str, device: str):
    model_dir = f"models/{persona}"
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"No model found at {model_dir}. Run finetune.py first.")
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer


def perplexity(text: str, model, tokenizer, device: str) -> float:
    """
    Perplexity = exp( mean negative log-likelihood per token ).
    Lower = model predicted this text well = tweet is in-character.
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss   # mean NLL across tokens

    return torch.exp(loss).item()


def per_token_surprise(text: str, model, tokenizer, device: str):
    """
    Returns (tokens, nll_per_token) for a single tweet.
    Used to build the heatmap: which specific words surprised the model?
    """
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        logits = model(input_ids).logits   # [1, seq_len, vocab_size]

    # To score token[i], we look at logits[i-1] (predict next from previous)
    shift_logits = logits[0, :-1, :]          # [seq_len-1, vocab_size]
    shift_ids    = input_ids[0, 1:]           # [seq_len-1]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_nll = -log_probs[range(len(shift_ids)), shift_ids]  # NLL per token

    tokens = [tokenizer.decode([t]) for t in shift_ids]
    return tokens, token_nll.cpu().tolist()


def score_all(persona: str, model, tokenizer, device: str):
    """Score every tweet in the eval set; return (dataframe, mu, sigma, median, iqr)."""
    df = pd.read_csv(f"data/processed/{persona}_eval.csv")

    ppls = []
    for text in tqdm(df["text"].dropna(), desc="Scoring tweets"):
        ppls.append(perplexity(text, model, tokenizer, device))

    df = df.copy()
    df["perplexity"] = ppls

    mu     = np.mean(ppls)
    sigma  = np.std(ppls)
    median = np.median(ppls)
    iqr    = np.percentile(ppls, 75) - np.percentile(ppls, 25)

    # Robust z-score: median/IQR instead of mean/std
    # WHY: mean/std are destroyed by hashtag/typo outliers (ppl=4000+)
    df["z_score"] = (df["perplexity"] - median) / iqr

    return df, mu, sigma, median, iqr


def print_distribution(persona: str, mu: float, sigma: float, ppls):
    median = np.median(ppls)
    p25, p75 = np.percentile(ppls, 25), np.percentile(ppls, 75)
    iqr = p75 - p25   # interquartile range — robust alternative to std

    print(f"\n{'='*55}")
    print(f"  {persona.upper()} — Perplexity Distribution")
    print(f"{'='*55}")
    print(f"  μ (mean)    : {mu:.2f}")
    print(f"  median      : {median:.2f}   ← robust center (use this)")
    print(f"  σ (std dev) : {sigma:.2f}   ← inflated by typo/hashtag outliers")
    print(f"  IQR         : {iqr:.2f}   ← robust predictability index (use this)")
    print(f"  p25 / p75   : {p25:.2f} / {p75:.2f}")
    print(f"  min / max   : {np.min(ppls):.2f} / {np.max(ppls):.2f}")
    print(f"\n  Predictability index (IQR): low = robot-consistent, high = chaotic")


def print_examples(df: pd.DataFrame, persona: str, model, tokenizer, device: str):
    print(f"\n--- Most IN-CHARACTER tweets (z < 0, model saw them coming) ---")
    top_in = df.nsmallest(N_EXAMPLES, "z_score")
    for _, row in top_in.iterrows():
        print(f"  z={row['z_score']:+.2f} | ppl={row['perplexity']:.1f} | {row['text'][:90]}")

    print(f"\n--- Most OUT-OF-CHARACTER tweets (z > 0, model was surprised) ---")
    top_out = df.nlargest(N_EXAMPLES, "z_score")
    for _, row in top_out.iterrows():
        print(f"  z={row['z_score']:+.2f} | ppl={row['perplexity']:.1f} | {row['text'][:90]}")

    # Per-token breakdown on the most out-of-character tweet
    outlier = top_out.iloc[0]["text"]
    tokens, nlls = per_token_surprise(outlier, model, tokenizer, device)
    max_nll = max(nlls) if nlls else 1

    print(f"\n--- Per-token surprise: most out-of-character tweet ---")
    print(f"  Tweet: {outlier[:120]}")
    print(f"  (token : surprise level)")
    for tok, nll in zip(tokens, nlls):
        bar = "█" * int(8 * nll / max_nll)
        print(f"  {repr(tok):15s} {bar:<8} {nll:.2f}")


def run(persona: str):
    device = get_device()
    print(f"\n=== Scoring: {persona} | device: {device} ===")

    model, tokenizer = load_model(persona, device)
    df, mu, sigma, median, iqr = score_all(persona, model, tokenizer, device)

    print_distribution(persona, mu, sigma, df["perplexity"].values)
    print_examples(df, persona, model, tokenizer, device)

    # Save full scored eval set for later analysis / paper figures
    out_path = f"data/processed/{persona}_scored.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  Full results saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="trump")
    args = parser.parse_args()
    run(args.persona)
