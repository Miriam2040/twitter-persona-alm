"""
cross_score.py — score persona A's tweets through persona B's model.

Answers: "How much does Musk tweet like Trump?" and vice versa.
Also builds the full N×N cross-scoring matrix for all trained personas.

Run:
    python src/cross_score.py                    # full matrix
    python src/cross_score.py --author musk --model trump   # one pair

Output:
    results/cross_score_matrix.csv   — N×N table of median perplexities
    results/cross_score_matrix.json  — same + z-scores + interesting examples
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

MAX_LENGTH = 128
N_EXAMPLES = 3


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(persona: str, device: str):
    model_dir = f"models/{persona}"
    tokenizer = GPT2Tokenizer.from_pretrained(model_dir)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_dir).to(device)
    model.eval()
    return model, tokenizer


def perplexity(text: str, model, tokenizer, device: str) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    with torch.no_grad():
        loss = model(enc["input_ids"].to(device), labels=enc["input_ids"].to(device)).loss
    return torch.exp(loss).item()


def score_tweets_with_model(tweets: list, model, tokenizer, device: str, desc: str) -> list:
    ppls = []
    for text in tqdm(tweets, desc=desc, leave=False):
        ppls.append(perplexity(text, model, tokenizer, device))
    return ppls


def load_eval_tweets(persona: str, n: int = 500) -> list:
    """Load a sample of eval tweets for cross-scoring. Use 500 for speed."""
    path = f"data/processed/{persona}_eval.csv"
    df = pd.read_csv(path)
    texts = df["text"].dropna().tolist()
    # Use every Nth tweet to get a representative sample
    step = max(1, len(texts) // n)
    return texts[::step][:n]


def load_home_baseline(persona: str):
    """Load the persona's own median and IQR from their scored eval CSV."""
    path = f"data/processed/{persona}_scored.csv"
    if not os.path.exists(path):
        path = f"results/{persona}_eval_scores.csv"
    df = pd.read_csv(path)
    ppls = df["perplexity"].dropna().values
    return np.median(ppls), np.percentile(ppls, 75) - np.percentile(ppls, 25)


def run_pair(author: str, model_persona: str, device: str, n_tweets: int = 500):
    """Score author's tweets through model_persona's model."""
    tweets = load_eval_tweets(author, n=n_tweets)
    model, tokenizer = load_model(model_persona, device)
    desc = f"{author} tweets → {model_persona} model"
    ppls = score_tweets_with_model(tweets, model, tokenizer, device, desc)
    median = np.median(ppls)
    iqr = np.percentile(ppls, 75) - np.percentile(ppls, 25)

    # Load home baseline to compute z-score relative to model's own distribution
    home_median, home_iqr = load_home_baseline(model_persona)
    z = (median - home_median) / home_iqr

    return {
        "author": author,
        "model": model_persona,
        "n_tweets": len(tweets),
        "median_ppl": round(median, 1),
        "iqr": round(iqr, 1),
        "home_median": round(home_median, 1),
        "home_iqr": round(home_iqr, 1),
        "z_vs_home": round(z, 2),
        # Most in-character examples (lowest PPL)
        "most_in_character": [
            tweets[i] for i in np.argsort(ppls)[:N_EXAMPLES]
        ],
        # Most out-of-character examples (highest PPL)
        "most_out_of_character": [
            tweets[i] for i in np.argsort(ppls)[-N_EXAMPLES:][::-1]
        ],
    }


def run_matrix(personas: list, device: str, n_tweets: int = 500):
    """Build full N×N cross-scoring matrix."""
    results = []
    matrix_rows = []

    for author in personas:
        row = {"author": author}
        for model_p in personas:
            print(f"\n  [{author} → {model_p}]")
            result = run_pair(author, model_p, device, n_tweets)
            results.append(result)
            row[model_p] = result["median_ppl"]
        matrix_rows.append(row)

    matrix_df = pd.DataFrame(matrix_rows).set_index("author")
    os.makedirs("results", exist_ok=True)
    matrix_df.to_csv("results/cross_score_matrix.csv")
    print(f"\n  Matrix saved → results/cross_score_matrix.csv")

    with open("results/cross_score_matrix.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Details saved → results/cross_score_matrix.json")

    # Pretty print
    print(f"\n{'='*65}")
    print(f"  CROSS-SCORING MATRIX  (median perplexity)")
    print(f"  Row = whose tweets | Column = whose model reads them")
    print(f"  Lower = 'sounds more like them'")
    print(f"{'='*65}")
    print(matrix_df.to_string())
    print(f"\n  Diagonal = home score (author read by their own model)")

    # Highlight interesting off-diagonal findings
    print(f"\n--- Key findings ---")
    for r in results:
        if r["author"] != r["model"]:
            direction = "sounds like" if r["z_vs_home"] < 0 else "doesn't sound like"
            print(f"  {r['author']:25s} → {r['model']:25s} model: "
                  f"median={r['median_ppl']:6.1f}  z={r['z_vs_home']:+.2f}  ({direction} {r['model']})")

    return matrix_df, results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--author", default=None, help="whose tweets to score")
    parser.add_argument("--model",  default=None, help="which persona's model to use")
    parser.add_argument("--personas", nargs="+",
                        default=["trump", "musk"],
                        help="list of personas for full matrix (default: trump musk)")
    parser.add_argument("--n_tweets", type=int, default=500,
                        help="tweets to sample per author (default: 500)")
    args = parser.parse_args()

    device = get_device()
    print(f"\n=== Cross-scoring | device: {device} ===")

    if args.author and args.model:
        result = run_pair(args.author, args.model, device, args.n_tweets)
        print(f"\nResult: {args.author} tweets → {args.model} model")
        print(f"  Median PPL : {result['median_ppl']}")
        print(f"  z vs home  : {result['z_vs_home']:+.2f}")
        print(f"\nMost in-character {args.author} tweets (by {args.model} model):")
        for t in result["most_in_character"]:
            print(f"  • {t[:100]}")
        print(f"\nMost out-of-character:")
        for t in result["most_out_of_character"]:
            print(f"  • {t[:100]}")
    else:
        run_matrix(args.personas, device, args.n_tweets)
