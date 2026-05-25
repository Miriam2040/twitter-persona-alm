"""
compare.py — prove fine-tuning worked by comparing the persona model to base GPT-2.

Run:
    python src/compare.py --persona trump

What it does:
    Scores every eval tweet with TWO models:
      1. Base GPT-2   — the off-the-shelf model, knows nothing about this person
      2. Persona ALM  — fine-tuned on this person's tweets

    If fine-tuning actually learned the persona's style, the persona model should
    have LOWER perplexity on their tweets than base GPT-2. We report:
      - How much lower the median perplexity is (2× lower = fine-tuning clearly helped)
      - What % of tweets the persona model beats base GPT-2 on
        (expect 95%+ if fine-tuning worked; ~50% = useless)

    This is the justification for the whole approach:
    "We fine-tuned GPT-2 on X's tweets — here's proof it actually learned their style."
"""

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MAX_LENGTH = 128


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_path_or_name: str, device: str):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path_or_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path_or_name).to(device)
    model.eval()
    return model, tokenizer


def perplexity(text: str, model, tokenizer, device: str) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    input_ids = enc["input_ids"].to(device)
    with torch.no_grad():
        loss = model(input_ids, labels=input_ids).loss
    return torch.exp(loss).item()


def score_corpus(texts, model, tokenizer, device: str, label: str):
    ppls = [perplexity(t, model, tokenizer, device) for t in tqdm(texts, desc=label)]
    return np.array(ppls)


def run(persona: str):
    device = get_device()
    print(f"\n=== Comparing base GPT-2 vs {persona} ALM | device: {device} ===\n")

    df = pd.read_csv(f"data/processed/{persona}_eval.csv")
    texts = df["text"].dropna().tolist()

    print("Loading base GPT-2 (no fine-tuning)...")
    base_model, base_tok = load_model("gpt2", device)
    base_ppls = score_corpus(texts, base_model, base_tok, device, "Base GPT-2")
    del base_model  # free memory before loading second model

    print("\nLoading persona ALM (fine-tuned)...")
    alm_model, alm_tok = load_model(f"models/{persona}", device)
    alm_ppls = score_corpus(texts, alm_model, alm_tok, device, "Persona ALM")

    base_med = np.median(base_ppls)
    alm_med  = np.median(alm_ppls)
    base_iqr = np.percentile(base_ppls, 75) - np.percentile(base_ppls, 25)
    alm_iqr  = np.percentile(alm_ppls,  75) - np.percentile(alm_ppls,  25)
    pct_wins = np.mean(alm_ppls < base_ppls) * 100

    print(f"\n{'='*50}")
    print(f"  {persona.upper()} — Base GPT-2 vs Fine-tuned ALM")
    print(f"{'='*50}")
    print(f"  {'':25s}  {'Base GPT-2':>12s}  {'ALM':>12s}  {'Improvement':>12s}")
    print(f"  {'-'*63}")
    print(f"  {'Median perplexity':25s}  {base_med:12.1f}  {alm_med:12.1f}  {base_med/alm_med:11.1f}x")
    print(f"  {'IQR (spread)':25s}  {base_iqr:12.1f}  {alm_iqr:12.1f}  {base_iqr/alm_iqr:11.1f}x")
    print(f"  {'ALM beats base':25s}  {'—':>12s}  {pct_wins:11.1f}%  {'':>12s}")
    print(f"{'='*50}")

    if pct_wins >= 90:
        verdict = "Fine-tuning clearly worked — ALM has internalized this person's style."
    elif pct_wins >= 70:
        verdict = "Fine-tuning helped but more data or epochs would improve it further."
    else:
        verdict = "Fine-tuning had limited effect — consider more data or more epochs."
    print(f"\n  Verdict: {verdict}\n")

    out = pd.DataFrame({"text": texts, "base_ppl": base_ppls, "alm_ppl": alm_ppls})
    out["alm_wins"] = out["alm_ppl"] < out["base_ppl"]
    out_path = f"data/processed/{persona}_comparison.csv"
    out.to_csv(out_path, index=False)
    print(f"  Full comparison saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="trump")
    args = parser.parse_args()
    run(args.persona)
