# Data Sources

All raw tweet files are **gitignored** — only tweet IDs are committed (X/Twitter ToS compliance).  
Raw files live in `data/raw/{persona}/`, cleaned files in `data/processed/{persona}_tweets.csv`.

---

## Persona 1 — Donald Trump

### Source A: HuggingFace `fschlatt/trump-tweets` (primary)

| Field        | Detail |
|---|---|
| URL          | https://huggingface.co/datasets/fschlatt/trump-tweets |
| Origin       | Clone of the original Kaggle Trump Twitter Archive (headsortails) |
| License      | **CC0 1.0 — public domain.** No restrictions on use, including ML training. |
| Format       | Parquet (6.28 MB) |
| Total rows   | **56,571 tweets** |
| Date range   | August 2011 – January 5, 2021 |

**Columns:**

| Column       | Type      | Description |
|---|---|---|
| `id`         | int64     | Unique tweet ID |
| `text`       | string    | Raw tweet content |
| `is_retweet` | bool      | True if it is a retweet (RT @...) |
| `is_deleted` | bool      | True if tweet was deleted before account suspension |
| `device`     | string    | Client used: "Twitter for iPhone", "TweetDeck", etc. |
| `favorites`  | int64     | Like count at time of archiving |
| `retweets`   | int64     | Retweet count at time of archiving |
| `datetime`   | timestamp | UTC datetime of posting |
| `is_flagged` | bool      | True if Twitter added a fact-check/warning label |

**How to download:**

```python
from datasets import load_dataset
ds = load_dataset("fschlatt/trump-tweets")
df = ds["train"].to_pandas()
df.to_csv("data/raw/trump/tweets_raw.csv", index=False)
```

### Source B: GitHub `MarkHershey/CompleteTrumpTweetsArchive` (supplement)

| Field      | Detail |
|---|---|
| URL        | https://github.com/MarkHershey/CompleteTrumpTweetsArchive |
| License    | Not explicitly stated — archival/research use standard in NLP community |
| Format     | Two CSV files |
| File 1     | `realDonaldTrump_bf_office.csv` — **31,249 tweets**, May 2009 – Jan 19 2017 |
| File 2     | `realDonaldTrump_in_office.csv` — **23,075 tweets**, Jan 20 2017 – Jan 8 2021 |
| Total      | **54,324 tweets** |
| Why use it | Covers 2009–2011 which HuggingFace source misses |

**How to download:**

```bash
git clone https://github.com/MarkHershey/CompleteTrumpTweetsArchive.git
cp CompleteTrumpTweetsArchive/*.csv data/raw/trump/
```

### What we actually use from Trump data

- Merge both sources, deduplicate on tweet ID
- Drop all rows where `is_retweet = True`
- Drop rows where `text` length < 15 characters
- Drop rows where `text` starts with `RT @`
- Drop deleted tweets (`is_deleted = True`) — we only use tweets the person chose to keep public
- Keep replies (they carry style signal)
- **Expected usable tweets after filtering: ~38,000–42,000**
- Columns we keep: `id`, `text`, `datetime`, `device`, `favorites`, `retweets`, `is_flagged`

---

## Persona 2 — Elon Musk

### Source: Kaggle `dadalyndell/elon-musk-tweets-2010-to-2025-march` (primary)

| Field      | Detail |
|---|---|
| URL        | https://www.kaggle.com/datasets/dadalyndell/elon-musk-tweets-2010-to-2025-march |
| Description | Self-described as "probably the most complete archive of tweets published under the handle elonmusk" |
| License    | Community Data License Agreement — Permissive (standard Kaggle research license) |
| Format     | CSV |
| Date range | 2010 – April 2025 |
| Total rows | ~25,000–35,000 (dataset is regularly updated) |

**How to download:**

```bash
# requires Kaggle CLI configured with your API key
pip install kaggle
kaggle datasets download -d dadalyndell/elon-musk-tweets-2010-to-2025-march
unzip elon-musk-tweets-2010-to-2025-march.zip -d data/raw/musk/
```

### Backup source: HuggingFace `fdaudens/musk-tweets`

| Field  | Detail |
|---|---|
| URL    | https://huggingface.co/datasets/fdaudens/musk-tweets |
| Status | **Has a schema bug** — `Latitude`/`Longitude` field type mismatch causes load failure as of May 2026. Do not use as primary source. Use only if Kaggle source becomes unavailable. |

### Existing model (reference only, do NOT use as our model)

| Field  | Detail |
|---|---|
| URL    | https://huggingface.co/huggingtweets/elonmusk |
| What   | GPT-2 fine-tuned on Musk tweets by the HuggingTweets project |
| Use    | Sanity check / qualitative comparison against our ALM — not our model |

### What we actually use from Musk data

- Drop retweets (rows where text starts with `RT @`)
- Drop tweets < 15 characters
- Drop pure URL tweets
- Deduplicate on tweet ID
- **Expected usable tweets after filtering: ~18,000–25,000**
- Note: the 2022 period (Twitter acquisition) is particularly interesting for temporal drift analysis

---

## Persona 3 — US Politician (Bernie Sanders / AOC / Ted Cruz)

### Source: Kaggle `mrmorj/us-politicians-twitter-dataset`

| Field      | Detail |
|---|---|
| URL        | https://www.kaggle.com/datasets/mrmorj/us-politicians-twitter-dataset |
| Description | Tweets from US politicians collected via Twitter API |
| Format     | CSV |
| Coverage   | Multiple senators and representatives |

**How to download:**

```bash
kaggle datasets download -d mrmorj/us-politicians-twitter-dataset
unzip us-politicians-twitter-dataset.zip -d data/raw/politician/
```

### Which politician to pick

We pick **one** politician for Phase 1 based on tweet volume. Target minimum: 15,000 usable tweets.

Priority order:
1. Bernie Sanders — high volume, strong distinctive style, long history
2. Alexandria Ocasio-Cortez (AOC) — very active, distinct voice
3. Ted Cruz — high volume, conservative counterpoint to Trump comparison

**Why add a politician at all:** Trump and Musk are both heads of major entities. Adding a senator creates a three-way comparison that strengthens the paper — different platforms of power, different communication styles, different consistency patterns.

---

## Summary Table

| Persona | Source | Raw tweets | Usable after filter | Date range | License |
|---|---|---|---|---|---|
| Trump | HuggingFace CC0 + GitHub | ~56,500 | ~40,000 | 2009–2021 | CC0 / archival |
| Musk | Kaggle | ~30,000 | ~20,000 | 2010–2025 | CDLA Permissive |
| Politician (TBD) | Kaggle politicians dataset | ~20,000 | ~15,000 | varies | research use |

---

## Legal & Ethical Notes

| Question | Answer |
|---|---|
| Can we train on these tweets? | Yes for research. All three are public figures, tweets were public, CC0 or permissive licenses apply. Standard in NLP literature. |
| Can we redistribute raw tweets? | No — X ToS prohibits redistribution of raw tweet content. We commit only tweet IDs in `data/tweet_ids/`. |
| Can we publish perplexity scores? | Yes — scores are derived metrics, not the content itself. |
| Can we publish the fine-tuned model? | Yes — model weights do not contain the original tweets. Upload to HuggingFace Hub. |
| GDPR / EU concerns? | Public figures acting in public roles have reduced privacy expectations. Still: do not include any private/DM content (none exists in these archives). |
| Right of publicity? | We score real tweets, we do not generate fake ones. This is the safe framing. If we add generation, label outputs clearly as AI-generated. |

---

## File Naming Convention

```
data/
  raw/
    trump/
      tweets_raw.csv            # merged from both sources
    musk/
      tweets_raw.csv
    politician/
      tweets_raw.csv
  processed/
    trump_tweets.csv            # after filtering, columns selected
    musk_tweets.csv
    {name}_tweets.csv
  tweet_ids/
    trump_ids.txt               # committed to git
    musk_ids.txt
    {name}_ids.txt
```

Raw and processed folders are in `.gitignore`. Only `tweet_ids/` is committed.
