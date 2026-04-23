"""
RAG-based few-shot example retrieval for the Market Prompt Ambiguity Risk Scoring System.

Retrieves semantically similar positive (low-risk) and negative (high-risk) examples
from prediction_market_200_examples.json using TF-IDF cosine similarity, then formats
them for injection into the LLM analysis prompt.
"""

import json
import math
import re
from collections import Counter
from typing import List, Tuple

from config import PREDICTION_MARKET_EXAMPLES_PATH

# Score thresholds for splitting examples into positive/negative pools
POSITIVE_MAX_SCORE = 40   # risk_score <= 40: low-risk (positive) examples
NEGATIVE_MIN_SCORE = 60   # risk_score >= 60: high-risk (negative) examples


def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _build_idf(docs: List[List[str]]) -> dict:
    N = len(docs)
    df: Counter = Counter()
    for doc in docs:
        for term in set(doc):
            df[term] += 1
    return {t: math.log(N / df[t]) for t in df}


def _cosine_sim(query_tokens: List[str], doc_tokens: List[str], idf: dict) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0

    qtf = Counter(query_tokens)
    dtf = Counter(doc_tokens)
    shared = set(qtf) & set(dtf)
    if not shared:
        return 0.0

    dot = sum(
        (qtf[t] / len(query_tokens)) * idf.get(t, 0)
        * (dtf[t] / len(doc_tokens)) * idf.get(t, 0)
        for t in shared
    )

    def norm(tf, n):
        return math.sqrt(sum(((tf[t] / n) * idf.get(t, 0)) ** 2 for t in tf))

    denom = norm(qtf, len(query_tokens)) * norm(dtf, len(doc_tokens))
    return dot / denom if denom else 0.0


def _to_few_shot_format(entry: dict) -> dict:
    """Convert a dataset entry to the few-shot {"question", "result"} format."""
    return {
        "question": entry["prompt"],
        "result": {
            "risk_score": entry["risk_score"],
            "risk_tags": [],
            "rationale": entry["reasons"],
        },
    }


def retrieve_few_shot_examples(
    question: str,
    n_positive: int = 5,
    n_negative: int = 5,
) -> List[dict]:
    """
    Retrieve similar few-shot examples from the prediction market dataset via TF-IDF.

    Splits the 200-example dataset into low-risk (positive) and high-risk (negative)
    pools, ranks each by cosine similarity to the query, and returns the top-n from
    each pool interleaved for balanced representation.

    Args:
        question: The market question being analyzed.
        n_positive: Number of low-risk examples to include (default 5).
        n_negative: Number of high-risk examples to include (default 5).

    Returns:
        List of up to n_positive + n_negative few-shot examples in
        {"question", "result"} format, ordered positive-negative-positive-...
    """
    with open(PREDICTION_MARKET_EXAMPLES_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    all_tokens = [_tokenize(e["prompt"]) for e in dataset]
    idf = _build_idf(all_tokens)
    query_tokens = _tokenize(question)

    positives: List[Tuple[dict, List[str]]] = [
        (e, tok)
        for e, tok in zip(dataset, all_tokens)
        if e["risk_score"] <= POSITIVE_MAX_SCORE
    ]
    negatives: List[Tuple[dict, List[str]]] = [
        (e, tok)
        for e, tok in zip(dataset, all_tokens)
        if e["risk_score"] >= NEGATIVE_MIN_SCORE
    ]

    def rank(pool: List[Tuple[dict, List[str]]]) -> List[dict]:
        scored = sorted(
            pool,
            key=lambda x: _cosine_sim(query_tokens, x[1], idf),
            reverse=True,
        )
        return [e for e, _ in scored]

    top_pos = rank(positives)[:n_positive]
    top_neg = rank(negatives)[:n_negative]

    # Interleave positive and negative so the LLM sees both sides evenly
    few_shot: List[dict] = []
    for pos, neg in zip(top_pos, top_neg):
        few_shot.append(_to_few_shot_format(pos))
        few_shot.append(_to_few_shot_format(neg))
    for e in top_pos[len(top_neg):]:
        few_shot.append(_to_few_shot_format(e))
    for e in top_neg[len(top_pos):]:
        few_shot.append(_to_few_shot_format(e))

    return few_shot
