"""
preprocess.py — download and clean tweet data for one persona.

Run:
    python src/preprocess.py --persona trump
    python src/preprocess.py --persona musk
    python src/preprocess.py --persona democrat_senators
    python src/preprocess.py --persona republican_senators

What it does:
    1. Downloads the tweet dataset (HuggingFace or local CSV)
    2. Filters out noise (retweets, deleted tweets, very short tweets)
    3. Splits chronologically: 90% train / 10% eval
       WHY chronological not random: we want eval tweets to be "newer" than training,
       which is the real use case — scoring a new tweet against a model trained on the past.
    4. Saves two CSVs and also exports tweet IDs for git (raw text is gitignored)

Data sources:
    trump              — HuggingFace: fschlatt/trump-tweets (CC0, 56k tweets 2009-2021)
    musk               — HuggingFace: fdaudens/musk-tweets (streamed, 14.6k original tweets 2013-2025)
                         Run once: python src/preprocess.py --persona musk --download_musk
                         This streams the raw data and caches it at data/processed/musk_raw.csv
    democrat_senators  — HuggingFace: Jacobvs/PoliticalTweets filtered party=Democrat (97k tweets 2016-2023)
    republican_senators— HuggingFace: Jacobvs/PoliticalTweets filtered party=Republican (92k tweets 2016-2023)
    obama              — Kaggle: neelgajare/all-12000-president-obama-tweets (~12k tweets)
                         Requires: pip install kaggle + kaggle.json credentials
"""

import os
import argparse
from typing import Tuple
import pandas as pd
from datasets import load_dataset

PERSONAS = {
    "trump": {
        "source":       "hf",
        "hf_dataset":   "fschlatt/trump-tweets",
        "text_col":     "text",
        "retweet_col":  "is_retweet",       # True = retweet, drop these
        "deleted_col":  "is_deleted",
        "datetime_col": "datetime",
        "id_col":       "id",
    },
    "musk": {
        # Raw data pre-downloaded via --download_musk flag (HF streaming workaround)
        # Source: fdaudens/musk-tweets on HuggingFace (~78k rows, 14.6k original tweets 2013-2025)
        # IMPORTANT: musk_raw.csv has msg_type column (string: "X Update", "X Reply", "X Repost", etc.)
        # We keep ONLY "X Update" (original tweets) to match Trump/senators filtering (no replies).
        "source":         "local_csv",
        "local_csv":      "data/processed/musk_raw.csv",
        "text_col":       "text",
        "retweet_col":    None,             # handled via msg_type_keep below
        "msg_type_col":   "msg_type",       # filter to original tweets only
        "msg_type_keep":  "X Update",       # drop replies, reposts, FB/YT/Reddit
        "deleted_col":    None,
        "datetime_col":   "created_at",
        "id_col":         "id",
    },
    "democrat_senators": {
        # All US Democrat senators, 2016-2023. 97k tweets combined.
        # Source: Jacobvs/PoliticalTweets on HuggingFace
        "source":       "hf",
        "hf_dataset":   "Jacobvs/PoliticalTweets",
        "hf_filter":    ("party", "Democrat"),
        "text_col":     "text",
        "retweet_col":  None,
        "deleted_col":  None,
        "datetime_col": "date",
        "id_col":       "id",
    },
    "republican_senators": {
        # All US Republican senators, 2016-2023. 92k tweets combined.
        # Source: Jacobvs/PoliticalTweets on HuggingFace
        "source":       "hf",
        "hf_dataset":   "Jacobvs/PoliticalTweets",
        "hf_filter":    ("party", "Republican"),
        "text_col":     "text",
        "retweet_col":  None,
        "deleted_col":  None,
        "datetime_col": "date",
        "id_col":       "id",
    },
    "obama": {
        # Barack Obama tweets — requires Kaggle credentials
        # Dataset: neelgajare/all-12000-president-obama-tweets (~12k tweets)
        # Setup: pip install kaggle, place kaggle.json at ~/.kaggle/kaggle.json
        # Then: kaggle datasets download neelgajare/all-12000-president-obama-tweets
        "source":       "kaggle",
        "kaggle_dataset": "neelgajare/all-12000-president-obama-tweets",
        "text_col":     "Tweet",
        "retweet_col":  None,
        "deleted_col":  None,
        "datetime_col": "Timestamp",
        "id_col":       "Tweet Id",
    },
}

MIN_CHARS   = 15      # tweets shorter than this are noise (just a mention or emoji)
TRAIN_RATIO = 0.90


def download_musk_raw():
    """
    Stream the Musk dataset from HuggingFace and save locally.
    The dataset has a broken parquet schema (Latitude field), so we use streaming.
    Only needs to run once — result cached at data/processed/musk_raw.csv.
    """
    from datasets import load_dataset as _load
    print("Streaming Musk tweets from HuggingFace (one-time download, ~2 min)...")
    ds = _load("fdaudens/musk-tweets", streaming=True)
    rows = []
    for i, row in enumerate(ds["train"]):
        rows.append({
            "text":       row.get("Message", ""),
            "created_at": row.get("CreatedTime", ""),
            "is_retweet": row.get("MessageType", "") != "X Update",
            "id":         row.get("UniversalMessageId", ""),
        })
        if i % 10000 == 0:
            print(f"  {i:,} rows streamed...")
    df = pd.DataFrame(rows)
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/musk_raw.csv", index=False)
    print(f"  Saved {len(df):,} rows → data/processed/musk_raw.csv")
    orig = df[~df["is_retweet"]]
    print(f"  Original tweets (not reposts): {len(orig):,}")


def load_raw(persona: str) -> Tuple[pd.DataFrame, dict]:
    cfg = PERSONAS[persona]
    source = cfg.get("source", "hf")

    if source == "local_csv":
        path = cfg["local_csv"]
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Local CSV not found: {path}\n"
                f"Run first: python src/preprocess.py --persona {persona} --download_musk"
            )
        print(f"Loading {persona} from local CSV ({path})...")
        df = pd.read_csv(path)

    elif source == "hf":
        print(f"Downloading {persona} from HuggingFace ({cfg['hf_dataset']})...")
        ds = load_dataset(cfg["hf_dataset"])
        df = ds["train"].to_pandas()
        if "hf_filter" in cfg:
            col, val = cfg["hf_filter"]
            df = df[df[col] == val].copy()

    elif source == "kaggle":
        raise NotImplementedError(
            f"Kaggle download not automated. Steps:\n"
            f"  1. pip install kaggle\n"
            f"  2. Place kaggle.json at ~/.kaggle/kaggle.json\n"
            f"  3. kaggle datasets download {cfg['kaggle_dataset']} -p data/processed/ --unzip\n"
            f"  4. Re-run this script"
        )
    else:
        raise ValueError(f"Unknown source: {source}")

    print(f"  Raw rows: {len(df):,}")
    return df, cfg


def clean(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    original = len(df)

    # Drop retweets — they reflect someone else's style, not the persona's
    if cfg.get("retweet_col") and cfg["retweet_col"] in df.columns:
        df = df[~df[cfg["retweet_col"]].fillna(False)]

    # For sources with a msg_type string column (e.g. Musk): keep only original posts.
    # This ensures apples-to-apples comparison with Trump/senators (no replies).
    if cfg.get("msg_type_col") and cfg["msg_type_col"] in df.columns:
        keep_val = cfg["msg_type_keep"]
        before = len(df)
        df = df[df[cfg["msg_type_col"]] == keep_val].copy()
        print(f"  msg_type filter (keep='{keep_val}'): {before:,} → {len(df):,} rows")

    # Drop deleted tweets — the person chose to remove them, respect that
    if cfg["deleted_col"] and cfg["deleted_col"] in df.columns:
        df = df[~df[cfg["deleted_col"]].fillna(False)]

    # Drop tweets starting with RT @ (retweet prefix not caught by flag)
    df = df[~df[cfg["text_col"]].str.startswith("RT @", na=False)]

    # Drop very short tweets (just mentions, links, or single words)
    df = df[df[cfg["text_col"]].str.len() >= MIN_CHARS]

    # Deduplicate on tweet ID only — repeated identical text is real signal
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
    parser.add_argument("--persona", default="trump", choices=list(PERSONAS.keys()),
                        help="Persona to preprocess.")
    parser.add_argument("--download_musk", action="store_true",
                        help="Stream and cache Musk raw data from HuggingFace (run once before --persona musk).")
    args = parser.parse_args()

    if args.download_musk:
        download_musk_raw()
    else:
        run(args.persona)
