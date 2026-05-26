#!/usr/bin/env python3
"""CAC-strict contrastive author-consistency benchmark.

This module is intentionally self-contained. It uses only the Python standard
library, generates deterministic authentic/perturbed text pairs, and evaluates
any scorer that returns one numeric author-consistency score per text.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Optional


TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9']*")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
INVENTED_TOKEN_RE = re.compile(r"(?=.*[a-z])(?=.*\d)[a-z0-9]{7,}", re.I)
PERTURBATION_TYPES = (
    "nonce_substitution",
    "common_token_swap",
    "adjacent_content_swap",
)
COMMON_TOKEN_POOL_SKIP_TOP = 49
COMMON_TOKEN_POOL_SIZE = 251
CONTENT_STOPWORDS = {
    "about",
    "after",
    "also",
    "been",
    "being",
    "from",
    "have",
    "into",
    "just",
    "more",
    "only",
    "over",
    "some",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "very",
    "were",
    "what",
    "when",
    "with",
    "your",
}


@dataclass(frozen=True)
class TokenSpan:
    raw: str
    token: str
    start: int
    end: int
    content: bool


@dataclass
class CorpusIndex:
    row_count: int
    token_document_frequency: dict[str, int]
    phrase_document_frequency: dict[str, int]


@dataclass
class CACBenchmark:
    pairs: list[dict[str, Any]]
    skipped: list[dict[str, Any]]
    texts_to_score: list[dict[str, str]]
    index: CorpusIndex
    train_row_count: int
    eval_row_count: int
    eligible_eval_row_count: int
    sampled_eval_row_count: int
    common_token_pool_size: int


def normalize_token(raw: str) -> str:
    return raw.lower().strip("'")


def in_any_range(start: int, end: int, ranges: list[tuple[int, int]]) -> bool:
    return any(range_start <= start and end <= range_end for range_start, range_end in ranges)


def is_url_or_handle(
    text: str,
    start: int,
    end: int,
    raw: str,
    url_ranges: list[tuple[int, int]],
) -> bool:
    prefix = text[max(0, start - 8) : start].lower()
    if start > 0 and text[start - 1] == "@":
        return True
    return (
        in_any_range(start, end, url_ranges)
        or raw.lower().startswith("http")
        or prefix.endswith("https://")
        or prefix.endswith("http://")
    )


def token_spans(text: str) -> list[TokenSpan]:
    spans: list[TokenSpan] = []
    url_ranges = [(match.start(), match.end()) for match in URL_RE.finditer(text)]
    for match in TOKEN_RE.finditer(text):
        raw = match.group(0)
        token = normalize_token(raw)
        content = (
            len(token) >= 4
            and token not in CONTENT_STOPWORDS
            and not is_url_or_handle(text, match.start(), match.end(), raw, url_ranges)
        )
        spans.append(TokenSpan(raw, token, match.start(), match.end(), content))
    return spans


def content_spans(text: str) -> list[TokenSpan]:
    return [span for span in token_spans(text) if span.content]


def content_tokens(text: str) -> list[str]:
    return [span.token for span in content_spans(text)]


def phrase_keys(tokens: list[str], max_phrases: int = 24) -> list[str]:
    phrases: list[str] = []
    for n in (2, 3):
        for index in range(0, max(0, len(tokens) - n + 1)):
            phrases.append(" ".join(tokens[index : index + n]))
            if len(phrases) >= max_phrases:
                return phrases
    return phrases


def row_id(row: dict[str, str], row_number: int, id_column: str) -> str:
    if id_column and row.get(id_column):
        return str(row[id_column])
    return str(row_number)


def read_rows(path: Path, text_column: str, id_column: str) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        if text_column not in reader.fieldnames:
            raise ValueError(f"CSV {path} is missing required text column: {text_column}")
        rows: list[dict[str, Any]] = []
        for index, row in enumerate(reader):
            rows.append(
                {
                    "id": row_id(row, index, id_column),
                    "row_number": index,
                    "text": row[text_column] or "",
                    "raw": row,
                }
            )
    return rows


def build_corpus_index(rows: list[dict[str, Any]]) -> CorpusIndex:
    token_df: dict[str, int] = {}
    phrase_df: dict[str, int] = {}
    for row in rows:
        tokens = content_tokens(str(row["text"]))
        for token in set(tokens):
            token_df[token] = token_df.get(token, 0) + 1
        for phrase in set(phrase_keys(tokens)):
            phrase_df[phrase] = phrase_df.get(phrase, 0) + 1
    return CorpusIndex(
        row_count=len(rows),
        token_document_frequency=token_df,
        phrase_document_frequency=phrase_df,
    )


def evenly_spaced(rows: list[dict[str, Any]], sample_size: int) -> list[dict[str, Any]]:
    if sample_size >= len(rows):
        return rows
    step = len(rows) / sample_size
    return [rows[min(int(i * step), len(rows) - 1)] for i in range(sample_size)]


def sample_eval_rows(
    rows: list[dict[str, Any]],
    sample_size: int,
    min_content_tokens: int,
) -> list[dict[str, Any]]:
    return evenly_spaced(eligible_eval_rows(rows, min_content_tokens), sample_size)


def eligible_eval_rows(
    rows: list[dict[str, Any]],
    min_content_tokens: int,
) -> list[dict[str, Any]]:
    return [row for row in rows if len(content_tokens(str(row["text"]))) >= min_content_tokens]


def common_token_pool(index: CorpusIndex) -> list[str]:
    max_docs = max(1, math.floor(index.row_count * 0.20))
    tokens = [
        token
        for token, count in index.token_document_frequency.items()
        if len(token) >= 4 and count <= max_docs
    ]
    tokens.sort(key=lambda token: (-index.token_document_frequency[token], token))
    # Drop the most frequent content tokens so swaps do not become stopword-like,
    # but keep small corpora usable instead of producing an empty pool.
    if len(tokens) <= COMMON_TOKEN_POOL_SKIP_TOP:
        return tokens
    return tokens[
        COMMON_TOKEN_POOL_SKIP_TOP : COMMON_TOKEN_POOL_SKIP_TOP + COMMON_TOKEN_POOL_SIZE
    ]


def digest_hex(seed: int, post_id: str, perturbation_type: str) -> str:
    payload = f"{seed}:{post_id}:{perturbation_type}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def digest_int(seed: int, post_id: str, perturbation_type: str) -> int:
    return int(digest_hex(seed, post_id, perturbation_type), 16)


def text_id(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


def apply_case(replacement: str, original: str) -> str:
    if original.isupper():
        return replacement.upper()
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


def replace_span(text: str, span: TokenSpan, replacement: str) -> str:
    return text[: span.start] + replacement + text[span.end :]


def swap_spans(text: str, first: TokenSpan, second: TokenSpan) -> str:
    if first.start > second.start:
        first, second = second, first
    return (
        text[: first.start]
        + second.raw
        + text[first.end : second.start]
        + first.raw
        + text[second.end :]
    )


def lowest_frequency_spans(text: str, index: CorpusIndex) -> list[TokenSpan]:
    return sorted(
        content_spans(text),
        key=lambda span: (index.token_document_frequency.get(span.token, 0), span.start),
    )


def nonce_substitution(
    example: dict[str, Any],
    index: CorpusIndex,
    seed: int,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    candidates = lowest_frequency_spans(str(example["text"]), index)
    if not candidates:
        return None, "no eligible content token"
    target = candidates[0]
    digest = digest_hex(seed, str(example["id"]), "nonce_substitution")
    replacement = apply_case(f"zzq{digest[:8]}", target.raw)
    return (
        {
            "perturbation_type": "nonce_substitution",
            "perturbed_text": replace_span(str(example["text"]), target, replacement),
            "target_tokens": [target.raw],
            "replacement_tokens": [replacement],
        },
        None,
    )


def common_token_swap(
    example: dict[str, Any],
    index: CorpusIndex,
    pool: list[str],
    seed: int,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    if not pool:
        return None, "empty common-token pool"
    candidates = lowest_frequency_spans(str(example["text"]), index)
    if not candidates:
        return None, "no eligible content token"
    target = candidates[0]
    start = digest_int(seed, str(example["id"]), "common_token_swap") % len(pool)
    replacement_token = pool[start]
    for offset in range(len(pool)):
        candidate = pool[(start + offset) % len(pool)]
        if candidate != target.token:
            replacement_token = candidate
            break
    if replacement_token == target.token:
        return None, "no different common-token replacement"
    replacement = apply_case(replacement_token, target.raw)
    return (
        {
            "perturbation_type": "common_token_swap",
            "perturbed_text": replace_span(str(example["text"]), target, replacement),
            "target_tokens": [target.raw],
            "replacement_tokens": [replacement],
        },
        None,
    )


def adjacent_content_swap(
    example: dict[str, Any],
    index: CorpusIndex,
) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    original_text = str(example["text"])
    spans = content_spans(original_text)
    if len(spans) < 2:
        return None, "fewer than two content tokens"
    scored_pairs: list[tuple[int, int, TokenSpan, TokenSpan]] = []
    for left, right in zip(spans, spans[1:]):
        if left.token == right.token:
            continue
        combined = index.token_document_frequency.get(left.token, 0) + index.token_document_frequency.get(
            right.token, 0
        )
        scored_pairs.append((combined, left.start, left, right))
    if not scored_pairs:
        return None, "no different adjacent content tokens"
    _, _, first, second = sorted(scored_pairs, key=lambda item: (item[0], item[1]))[0]
    perturbed_text = swap_spans(original_text, first, second)
    if perturbed_text == original_text:
        return None, "adjacent swap made no change"
    return (
        {
            "perturbation_type": "adjacent_content_swap",
            "perturbed_text": perturbed_text,
            "target_tokens": [first.raw, second.raw],
            "replacement_tokens": [second.raw, first.raw],
        },
        None,
    )


def generate_pairs(
    examples: list[dict[str, Any]],
    index: CorpusIndex,
    pool: list[str],
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pairs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    generators = [
        (PERTURBATION_TYPES[0], lambda example: nonce_substitution(example, index, seed)),
        (PERTURBATION_TYPES[1], lambda example: common_token_swap(example, index, pool, seed)),
        (PERTURBATION_TYPES[2], lambda example: adjacent_content_swap(example, index)),
    ]
    for example in examples:
        for perturbation_type, generator in generators:
            perturbation, skip_reason = generator(example)
            if perturbation is None:
                skipped.append(
                    {
                        "post_id": str(example["id"]),
                        "perturbation_type": perturbation_type,
                        "reason": skip_reason or "unknown",
                    }
                )
                continue
            authentic_text = str(example["text"])
            perturbed_text = str(perturbation["perturbed_text"])
            pairs.append(
                {
                    "pair_id": f"{example['id']}:{perturbation_type}",
                    "post_id": str(example["id"]),
                    "perturbation_type": perturbation_type,
                    "authentic_text_id": text_id(authentic_text),
                    "perturbed_text_id": text_id(perturbed_text),
                    "authentic_text": authentic_text,
                    "perturbed_text": perturbed_text,
                    "target_tokens": perturbation["target_tokens"],
                    "replacement_tokens": perturbation["replacement_tokens"],
                }
            )
    return pairs, skipped


def lexical_evidence(text: str, index: CorpusIndex) -> dict[str, Any]:
    tokens = content_tokens(text)
    unique_tokens = sorted(set(tokens))
    token_items = [
        {"token": token, "document_frequency": index.token_document_frequency.get(token, 0)}
        for token in unique_tokens
    ]
    phrase_items = [
        {"phrase": phrase, "document_frequency": index.phrase_document_frequency.get(phrase, 0)}
        for phrase in phrase_keys(tokens)
    ]
    row_count = max(1, index.row_count)
    common_cutoff = max(2, math.floor(row_count * 0.02))
    return {
        "content_token_count": len(tokens),
        "token_document_frequencies": token_items,
        "phrase_document_frequencies": phrase_items,
        "unseen_content_tokens": [
            item["token"] for item in token_items if int(item["document_frequency"]) == 0
        ],
        "rare_content_tokens": [
            item["token"] for item in token_items if 0 < int(item["document_frequency"]) <= 3
        ],
        "common_content_tokens": [
            item["token"] for item in token_items if int(item["document_frequency"]) >= common_cutoff
        ],
        "invented_like_tokens": [token for token in unique_tokens if INVENTED_TOKEN_RE.search(token)],
    }


def lexical_score(text: str, index: CorpusIndex) -> dict[str, Any]:
    """Small dependency-free scorer for smoke tests, not a proposed baseline."""
    evidence = lexical_evidence(text, index)
    token_count = max(1, int(evidence["content_token_count"]))
    token_items = evidence["token_document_frequencies"]
    phrase_items = evidence["phrase_document_frequencies"]
    unseen = len(evidence["unseen_content_tokens"])
    rare = len(evidence["rare_content_tokens"])
    invented_unseen = len(
        set(evidence["invented_like_tokens"]) & set(evidence["unseen_content_tokens"])
    )
    token_item_count = max(1, len(token_items))
    common_ratio = len(evidence["common_content_tokens"]) / token_item_count
    uncommon_ratio = (unseen + rare) / token_item_count
    unseen_ratio = unseen / token_item_count
    rare_ratio = rare / token_item_count
    invented_unseen_ratio = invented_unseen / token_item_count
    phrase_support = sum(1 for item in phrase_items if int(item["document_frequency"]) > 0)
    phrase_ratio = phrase_support / max(1, len(phrase_items))
    phrase_weight = sum(math.log1p(int(item["document_frequency"])) for item in phrase_items)
    token_weight = sum(
        math.log1p(int(item["document_frequency"])) for item in token_items
    ) / token_count
    score = (
        0.18 * token_weight
        + 0.20 * common_ratio
        + 1.20 * phrase_ratio
        + 0.10 * phrase_weight
        - 0.65 * uncommon_ratio
        - 0.45 * unseen_ratio
        - 0.15 * rare_ratio
        - 1.00 * invented_unseen_ratio
    )
    return {
        "score": round(score, 6),
        "scorer": "lexical_overlap_v1",
        "evidence": evidence,
    }


def unique_text_records(pairs: list[dict[str, Any]]) -> list[dict[str, str]]:
    seen: set[str] = set()
    records: list[dict[str, str]] = []
    for pair in pairs:
        for role, text_key, id_key in (
            ("authentic", "authentic_text", "authentic_text_id"),
            ("perturbed", "perturbed_text", "perturbed_text_id"),
        ):
            item_id = str(pair[id_key])
            if item_id in seen:
                continue
            seen.add(item_id)
            records.append({"text_id": item_id, "role_hint": role, "text": str(pair[text_key])})
    return records


def score_texts_lexical(pairs: list[dict[str, Any]], index: CorpusIndex) -> dict[str, dict[str, Any]]:
    scores: dict[str, dict[str, Any]] = {}
    for item in unique_text_records(pairs):
        scores[item["text_id"]] = lexical_score(item["text"], index)
    return scores


def build_benchmark(
    train_csv: Path,
    eval_csv: Path,
    text_column: str = "text",
    id_column: str = "id",
    sample_size: int = 200,
    seed: int = 0,
    min_content_tokens: int = 5,
) -> CACBenchmark:
    """Build deterministic CAC-strict pairs for any downstream scorer."""
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if min_content_tokens < 0:
        raise ValueError("min_content_tokens must be non-negative")
    train_rows = read_rows(train_csv, text_column, id_column)
    eval_rows = read_rows(eval_csv, text_column, id_column)
    index = build_corpus_index(train_rows)
    eligible_rows = eligible_eval_rows(eval_rows, min_content_tokens)
    sampled_rows = evenly_spaced(eligible_rows, sample_size)
    pool = common_token_pool(index)
    pairs, skipped = generate_pairs(sampled_rows, index, pool, seed=seed)
    return CACBenchmark(
        pairs=pairs,
        skipped=skipped,
        texts_to_score=unique_text_records(pairs),
        index=index,
        train_row_count=len(train_rows),
        eval_row_count=len(eval_rows),
        eligible_eval_row_count=len(eligible_rows),
        sampled_eval_row_count=len(sampled_rows),
        common_token_pool_size=len(pool),
    )


def read_external_scores(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"Scores CSV has no header: {path}")
        score_column = None
        for candidate in ("author_consistency_score", "score"):
            if candidate in reader.fieldnames:
                score_column = candidate
                break
        if "text_id" not in reader.fieldnames or score_column is None:
            raise ValueError(
                "Scores CSV must contain text_id and author_consistency_score "
                "(or score) columns."
            )
        scores: dict[str, dict[str, Any]] = {}
        for row in reader:
            item_id = str(row["text_id"])
            try:
                score = float(row[score_column])
            except ValueError as exc:
                raise ValueError(f"Non-numeric score for text_id {item_id}: {row[score_column]}") from exc
            if not math.isfinite(score):
                raise ValueError(f"Non-finite score for text_id {item_id}: {row[score_column]}")
            scores[item_id] = {
                "score": score,
                "scorer": "external_scores_csv",
            }
    return scores


def decision_for_margin(margin: float) -> str:
    if margin > 0:
        return "pass"
    if margin < 0:
        return "fail"
    return "tie"


def evaluate_pairs(
    pairs: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    scorer_name: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for pair in pairs:
        missing = [
            item_id
            for item_id in (pair["authentic_text_id"], pair["perturbed_text_id"])
            if item_id not in scores
        ]
        if missing:
            raise ValueError(f"Missing scores for text_id(s): {', '.join(missing[:5])}")
        authentic = scores[pair["authentic_text_id"]]
        perturbed = scores[pair["perturbed_text_id"]]
        margin = round(float(authentic["score"]) - float(perturbed["score"]), 6)
        records.append(
            {
                **pair,
                "scorer": scorer_name,
                "authentic_score": authentic["score"],
                "perturbed_score": perturbed["score"],
                "margin": margin,
                "decision": decision_for_margin(margin),
                "authentic_score_details": authentic,
                "perturbed_score_details": perturbed,
            }
        )
    return records


def calculate_cac_strict(records: list[dict[str, Any]]) -> tuple[Optional[float], dict[str, Optional[float]]]:
    by_type: dict[str, Optional[float]] = {}
    for perturbation_type in PERTURBATION_TYPES:
        subset = [record for record in records if record["perturbation_type"] == perturbation_type]
        by_type[perturbation_type] = (
            sum(record["decision"] == "pass" for record in subset) / len(subset)
            if subset
            else None
        )
    values = [value for value in by_type.values() if value is not None]
    return (statistics.fmean(values) if values else None, by_type)


def summarize(records: list[dict[str, Any]], skipped: list[dict[str, Any]]) -> dict[str, Any]:
    passes = [r for r in records if r["decision"] == "pass"]
    failures = [r for r in records if r["decision"] == "fail"]
    ties = [r for r in records if r["decision"] == "tie"]
    margins = [float(r["margin"]) for r in records]
    by_type: dict[str, dict[str, Any]] = {}
    for perturbation_type in sorted({r["perturbation_type"] for r in records}):
        subset = [r for r in records if r["perturbation_type"] == perturbation_type]
        subset_passes = [r for r in subset if r["decision"] == "pass"]
        subset_failures = [r for r in subset if r["decision"] == "fail"]
        subset_ties = [r for r in subset if r["decision"] == "tie"]
        subset_margins = [float(r["margin"]) for r in subset]
        by_type[perturbation_type] = {
            "pairs": len(subset),
            "passes": len(subset_passes),
            "failures": len(subset_failures),
            "ties": len(subset_ties),
            "strict_accuracy": len(subset_passes) / len(subset) if subset else None,
            "non_tie_accuracy": len(subset_passes) / (len(subset_passes) + len(subset_failures))
            if subset_passes or subset_failures
            else None,
            "tie_rate": len(subset_ties) / len(subset) if subset else None,
            "mean_margin": statistics.fmean(subset_margins) if subset_margins else None,
            "median_margin": statistics.median(subset_margins) if subset_margins else None,
        }
    skipped_by_reason: dict[str, int] = {}
    for item in skipped:
        key = f"{item['perturbation_type']}:{item['reason']}"
        skipped_by_reason[key] = skipped_by_reason.get(key, 0) + 1
    cac_strict, components = calculate_cac_strict(records)
    return {
        "pairs": len(records),
        "passes": len(passes),
        "failures": len(failures),
        "ties": len(ties),
        "cac_strict": cac_strict,
        "cac_strict_components": components,
        "pairwise_accuracy": len(passes) / len(records) if records else None,
        "non_tie_accuracy": len(passes) / (len(passes) + len(failures))
        if passes or failures
        else None,
        "tie_rate": len(ties) / len(records) if records else None,
        "mean_margin": statistics.fmean(margins) if margins else None,
        "median_margin": statistics.median(margins) if margins else None,
        "by_perturbation": by_type,
        "skipped": {"count": len(skipped), "by_reason": skipped_by_reason},
    }


def evaluate_benchmark_scores(
    pairs: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    scores: dict[str, dict[str, Any]],
    scorer_name: str,
    config: Optional[dict[str, Any]] = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Evaluate method-provided scores where higher means more author-consistent."""
    records = evaluate_pairs(pairs, scores, scorer_name=scorer_name)
    summary = summarize(records, skipped)
    summary["scorer"] = scorer_name
    if config is not None:
        summary["config"] = config
    return records, summary


def pct(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def num(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def format_summary(summary: dict[str, Any], records: list[dict[str, Any]]) -> str:
    lines = [
        "# CAC-strict Benchmark",
        "",
        f"- Scorer: `{summary['scorer']}`",
        f"- CAC-strict: {pct(summary['cac_strict'])}",
        f"- Pairs: {summary['pairs']}",
        f"- Pairwise accuracy: {pct(summary['pairwise_accuracy'])}",
        f"- Non-tie accuracy: {pct(summary['non_tie_accuracy'])}",
        f"- Tie rate: {pct(summary['tie_rate'])}",
        f"- Mean margin: {num(summary['mean_margin'])}",
        f"- Median margin: {num(summary['median_margin'])}",
    ]
    config = summary.get("config") or {}
    if config:
        lines.extend(
            [
                f"- Train rows: {config.get('train_rows', 'n/a')}",
                f"- Eval rows: {config.get('eval_rows', 'n/a')}",
                f"- Eligible eval rows: {config.get('eligible_eval_rows', 'n/a')}",
                f"- Sampled eval rows: {config.get('eligible_eval_rows_sampled', 'n/a')}",
            ]
        )
    lines.extend(
        [
            "",
            "## By Perturbation",
            "",
            "| Perturbation | Pass | Fail | Tie | Pairs | Strict Acc. | Non-Tie Acc. | Mean Margin |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for perturbation_type, stats in summary["by_perturbation"].items():
        lines.append(
            "| {kind} | {passes} | {failures} | {ties} | {pairs} | {acc} | {non_tie} | {margin} |".format(
                kind=perturbation_type,
                passes=stats["passes"],
                failures=stats["failures"],
                ties=stats["ties"],
                pairs=stats["pairs"],
                acc=pct(stats["strict_accuracy"]),
                non_tie=pct(stats["non_tie_accuracy"]),
                margin=num(stats["mean_margin"]),
            )
        )
    failures = [r for r in records if r["decision"] != "pass"]
    lines.extend(["", "## Hard Cases", ""])
    if not failures:
        lines.append("- none")
    else:
        for record in sorted(failures, key=lambda item: float(item["margin"]))[:20]:
            lines.append(
                "- {pair_id}: {decision} margin={margin:.3f} authentic={authentic:.3f} perturbed={perturbed:.3f} text={text}".format(
                    pair_id=record["pair_id"],
                    decision=record["decision"],
                    margin=float(record["margin"]),
                    authentic=float(record["authentic_score"]),
                    perturbed=float(record["perturbed_score"]),
                    text=record["perturbed_text"][:120].replace("\n", " "),
                )
            )
    lines.append("")
    return "\n".join(lines)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def write_benchmark_files(out_dir: Path, benchmark: CACBenchmark) -> None:
    write_jsonl(out_dir / "pairs.jsonl", benchmark.pairs)
    write_jsonl(out_dir / "texts_to_score.jsonl", benchmark.texts_to_score)
    write_json(out_dir / "skipped.json", {"skipped": benchmark.skipped})


def write_evaluation_files(
    out_dir: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    write_json(out_dir / "summary.json", summary)
    write_jsonl(out_dir / "results.jsonl", records)
    (out_dir / "summary.md").write_text(format_summary(summary, records), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate CAC-strict pairs.")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--persona", default="trump")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--eval-csv", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--out-dir", default="benchmarks/cac-strict-001")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-content-tokens", type=int, default=5)
    parser.add_argument("--scorer", choices=["lexical", "external", "none"], default="lexical")
    parser.add_argument(
        "--scores-csv",
        default=None,
        help="CSV with text_id and author_consistency_score columns for --scorer external.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train_csv is None:
        args.train_csv = f"data/processed/{args.persona}_train.csv"
    if args.eval_csv is None:
        args.eval_csv = f"data/processed/{args.persona}_eval.csv"
    repo = Path(args.repo).resolve()
    train_path = repo / args.train_csv
    eval_path = repo / args.eval_csv
    out_dir = repo / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        benchmark = build_benchmark(
            train_path,
            eval_path,
            text_column=args.text_column,
            id_column=args.id_column,
            sample_size=args.sample_size,
            seed=args.seed,
            min_content_tokens=args.min_content_tokens,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    write_benchmark_files(out_dir, benchmark)

    if args.scorer == "none":
        print(
            "Wrote {pairs} pairs and {texts} unique texts to {out_dir}".format(
                pairs=len(benchmark.pairs),
                texts=len(benchmark.texts_to_score),
                out_dir=out_dir,
            )
        )
        return

    try:
        if args.scorer == "lexical":
            scores = score_texts_lexical(benchmark.pairs, benchmark.index)
            scorer_name = "lexical_overlap_v1"
        else:
            if not args.scores_csv:
                raise SystemExit("--scores-csv is required when --scorer external")
            scores = read_external_scores(repo / args.scores_csv)
            scorer_name = "external_scores_csv"

        records, summary = evaluate_benchmark_scores(
            benchmark.pairs,
            benchmark.skipped,
            scores,
            scorer_name=scorer_name,
            config={
                "train_csv": args.train_csv,
                "eval_csv": args.eval_csv,
                "text_column": args.text_column,
                "id_column": args.id_column,
                "sample_size": args.sample_size,
                "seed": args.seed,
                "min_content_tokens": args.min_content_tokens,
                "scorer": args.scorer,
                "scores_csv": args.scores_csv,
                "train_rows": benchmark.train_row_count,
                "eval_rows": benchmark.eval_row_count,
                "eligible_eval_rows": benchmark.eligible_eval_row_count,
                "eligible_eval_rows_sampled": benchmark.sampled_eval_row_count,
                "common_token_pool_size": benchmark.common_token_pool_size,
            },
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    write_evaluation_files(out_dir, records, summary)
    print((out_dir / "summary.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
