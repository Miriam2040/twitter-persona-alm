"""
score_alm.py — score texts from texts_to_score.jsonl using a fine-tuned GPT-2 ALM.

Reads: benchmarks/<dir>/texts_to_score.jsonl
Writes: benchmarks/<dir>/alm_scores.csv  (text_id, author_consistency_score)

score = -perplexity  (higher = more in-character, as required by cac_strict external scorer)

Usage:
    python src/score_alm.py --persona trump --bench-dir benchmarks/cac-strict-alm-trump
"""
import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"


def perplexity(text, model, tokenizer, device, max_length=128):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        loss = model(**enc, labels=enc["input_ids"]).loss
    return torch.exp(loss).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True)
    parser.add_argument("--bench-dir", required=True)
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()

    model_dir = args.model_dir or f"models/{args.persona}"
    bench_dir = Path(args.bench_dir)
    texts_path = bench_dir / "texts_to_score.jsonl"
    out_path = bench_dir / "alm_scores.csv"

    device = get_device()
    print(f"Loading {model_dir} on {device}...")
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    texts = [json.loads(l) for l in texts_path.read_text().splitlines() if l.strip()]
    print(f"Scoring {len(texts)} texts...")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text_id", "author_consistency_score"])
        for i, item in enumerate(texts):
            if i % 100 == 0:
                print(f"  {i}/{len(texts)}...", end="\r", flush=True)
            score = -perplexity(item["text"], model, tokenizer, device)
            writer.writerow([item["text_id"], score])

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
