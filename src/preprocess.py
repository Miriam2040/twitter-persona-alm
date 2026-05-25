"""
preprocess.py — download and clean tweet data for one persona.

Run:
    python src/preprocess.py --persona trump

What it does:
    1. Downloads the tweet dataset from HuggingFace (no login needed for Trump CC0 data)
    2. Filters out noise (retweets, deleted tweets, very short tweets)
    3. Splits chronologically: 90% train / 10% eval
       WHY chronological not random: we want eval tweets to be "newer" than training,
       which is the real use case — scoring a new tweet against a model trained on the past.
    4. Saves two CSVs and also exports tweet IDs for git (raw text is gitignored)
"""

import os
import argparse
from typing import Tuple
import pandas as pd
from datasets import load_dataset

PERSONAS = {
    "trump": {
        "hf_dataset": "fschlatt/trump-tweets",
        "text_col": "text",
        "retweet_col": "is_retweet",
        "deleted_col": "is_deleted",
        "datetime_col": "datetime",
        "id_col": "id",
    },
    "cruz": {
        "hf_dataset": "m-newhauser/senator-tweets",
        "hf_filter": ("username", "SenTedCruz"),  # filter to one senator
        "text_col": "text",
        "retweet_col": None,
        "deleted_col": None,
        "datetime_col": "date",
        "id_col": "id",
    },
}

MIN_CHARS = 15      # tweets shorter than this are noise (just a mention or emoji)
TRAIN_RATIO = 0.90


def load_raw(persona: str) -> pd.DataFrame:
    cfg = PERSONAS[persona]
    print(f"Downloading {persona} tweets from HuggingFace ({cfg['hf_dataset']})...")
    ds = load_dataset(cfg["hf_dataset"])
    df = ds["train"].to_pandas()
    if "hf_filter" in cfg:
        col, val = cfg["hf_filter"]
        df = df[df[col] == val]
    print(f"  Raw rows: {len(df):,}")
    return df, cfg


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    original = len(df)

    # Drop retweets — they reflect someone else's style, not the persona's
    if cfg["retweet_col"] in df.columns:
        df = df[~df[cfg["retweet_col"]].fillna(False)]

    # Drop deleted tweets — the person chose to remove them, respect that
    if cfg["deleted_col"] in df.columns:
        df = df[~df[cfg["deleted_col"]].fillna(False)]

    # Drop tweets starting with RT @ (retweet prefix not caught by flag)
    df = df[~df[cfg["text_col"]].str.startswith("RT @", na=False)]

    # Drop very short tweets (just mentions, links, or single words)
    df = df[df[cfg["text_col"]].str.len() >= MIN_CHARS]

    # Deduplicate on tweet ID only — repeated identical text is real signal
    # (e.g. "MAKE AMERICA GREAT AGAIN!" tweeted 50 times = very in-character)
    df = df.drop_duplicates(subset=[cfg["id_col"]])

    # Strip internal newlines so each tweet is one line in the text file
    df[cfg["text_col"]] = df[cfg["text_col"]].str.replace(r"\n+", " ", regex=True).str.strip()

    print(f"  After filtering: {len(df):,}  (removed {original - len(df):,})")
    return df


def split(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Sort chronologically so eval = newest tweets
    df = df.sort_values(cfg["datetime_col"]).reset_index(drop=True)
    split_idx = int(len(df) * TRAIN_RATIO)
    train, eval_ = df.iloc[:split_idx], df.iloc[split_idx:]
    print(f"  Train: {len(train):,} | Eval: {len(eval_):,}")
    return train, eval_


def save(train: pd.DataFrame, eval_: pd.DataFrame, cfg: dict, persona: str):
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("data/tweet_ids", exist_ok=True)

    train.to_csv(f"data/processed/{persona}_train.csv", index=False)
    eval_.to_csv(f"data/processed/{persona}_eval.csv", index=False)

    # Tweet IDs only go to git (ToS compliant)
    all_ids = pd.concat([train, eval_])[cfg["id_col"]]
    all_ids.to_csv(f"data/tweet_ids/{persona}_ids.txt", index=False, header=False)

    print(f"  Saved → data/processed/{persona}_train.csv")
    print(f"  Saved → data/processed/{persona}_eval.csv")
    print(f"  Saved → data/tweet_ids/{persona}_ids.txt  (commit this to git)")


def run(persona: str):
    print(f"\n=== Preprocessing: {persona} ===")
    df, cfg = load_raw(persona)
    df = clean(df, cfg)
    train, eval_ = split(df, cfg)
    save(train, eval_, cfg, persona)
    print("Done.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="trump", choices=list(PERSONAS.keys()))
    args = parser.parse_args()
    run(args.persona)
