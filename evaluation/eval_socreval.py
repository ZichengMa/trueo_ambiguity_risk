"""
SocREval-based evaluation orchestrator for the Market Prompt Ambiguity Risk Scoring System.

Compares three conditions:
  A — Bare GLM baseline (minimal prompt, no system prompt, no few-shot, no web search)
  B — Full system without web search
  C — Full system with web search

Uses the combined SocREval "All" strategy (Dialectic + Maieutics + Definition) with
the same GLM model as evaluator.

Usage:
    cd evaluation
    python eval_socreval.py                # Run full evaluation (first 50 examples)
    python eval_socreval.py --resume       # Resume from cache
    python eval_socreval.py --report-only  # Regenerate report/plots from cached results
    python eval_socreval.py --n 10         # Run only first 10 examples
"""

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths & import setup
# ---------------------------------------------------------------------------
EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
DATASET_PATH = PROJECT_ROOT / "prediction_market_200_examples.json"
RESULTS_PATH = EVAL_DIR / "eval_results.json"
REPORT_PATH = EVAL_DIR / "eval_report.txt"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from zhipuai import ZhipuAI

from config import ZHIPU_API_KEY, ZHIPU_MODEL
from eval_prompts import BASELINE_PROMPT_TEMPLATE, SOCREVAL_EVALUATOR_PROMPT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONDITIONS = ["A_baseline", "B_system_basic", "C_system_full"]
CONDITION_LABELS = {
    "A_baseline": "A: Baseline",
    "B_system_basic": "B: System (no search)",
    "C_system_full": "C: System (full)",
}
DIMENSIONS = [
    "score_accuracy",
    "rationale_quality",
    "tag_correctness",
    "overall_quality",
]


# ========================== Generation helpers =============================


def _get_glm_client() -> ZhipuAI:
    return ZhipuAI(
        api_key=ZHIPU_API_KEY,
        base_url="https://open.bigmodel.cn/api/coding/paas/v4",
    )


def _parse_json_response(content: str) -> Optional[dict]:
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        return None
    try:
        return json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return None


def _call_glm(
    client: ZhipuAI, messages: list, temperature: float = 0.3, timeout: float = 180
) -> str:
    response = client.chat.completions.create(
        model=ZHIPU_MODEL,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
    )
    return response.choices[0].message.content


def generate_baseline(question: str, client: ZhipuAI) -> dict:
    prompt = BASELINE_PROMPT_TEMPLATE.format(question=question)
    content = _call_glm(client, [{"role": "user", "content": prompt}])
    parsed = _parse_json_response(content)
    if parsed is None:
        return {
            "risk_score": -1,
            "risk_tags": [],
            "rationale": f"PARSE_ERROR: {content[:200]}",
        }
    return {
        "risk_score": max(0, min(100, int(parsed.get("risk_score", -1)))),
        "risk_tags": parsed.get("risk_tags", []),
        "rationale": parsed.get("rationale", ""),
    }


def generate_system_basic(question: str, agent=None, client=None) -> dict:
    from prompts import build_analysis_prompt, SYSTEM_PROMPT

    if client is None:
        client = _get_glm_client()
    prompt = build_analysis_prompt(question, include_few_shot=True)
    content = _call_glm(
        client,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    parsed = _parse_json_response(content)
    if parsed is None:
        return {
            "risk_score": -1,
            "risk_tags": [],
            "rationale": f"PARSE_ERROR: {content[:200]}",
        }
    return {
        "risk_score": max(0, min(100, int(parsed.get("risk_score", -1)))),
        "risk_tags": parsed.get("risk_tags", []),
        "rationale": parsed.get("rationale", ""),
    }


def generate_system_full(question: str, agent=None, client=None) -> dict:
    import json as _json
    from prompts import build_analysis_prompt, SYSTEM_PROMPT
    from main import merge_analysis_context
    from rag import (
        retrieve_few_shot_examples,
        _tokenize,
        _build_idf,
        _cosine_sim,
        _to_few_shot_format,
        POSITIVE_MAX_SCORE,
        NEGATIVE_MIN_SCORE,
    )
    from search import WebSearchClient
    from config import PREDICTION_MARKET_EXAMPLES_PATH

    if client is None:
        client = _get_glm_client()
    search_client = WebSearchClient()
    web_ctx = search_client.build_context(question)
    ctx = merge_analysis_context(web_search_context=web_ctx)

    # RAG retrieval with leak exclusion
    with open(PREDICTION_MARKET_EXAMPLES_PATH, "r", encoding="utf-8") as f:
        dataset = _json.load(f)
    filtered = [e for e in dataset if e["prompt"] != question]
    all_tokens = [_tokenize(e["prompt"]) for e in filtered]
    idf = _build_idf(all_tokens)
    query_tokens = _tokenize(question)
    positives = [
        (e, tok)
        for e, tok in zip(filtered, all_tokens)
        if e["risk_score"] <= POSITIVE_MAX_SCORE
    ]
    negatives = [
        (e, tok)
        for e, tok in zip(filtered, all_tokens)
        if e["risk_score"] >= NEGATIVE_MIN_SCORE
    ]

    def rank(pool):
        scored = sorted(
            pool, key=lambda x: _cosine_sim(query_tokens, x[1], idf), reverse=True
        )
        return [e for e, _ in scored]

    top_pos = rank(positives)[:5]
    top_neg = rank(negatives)[:5]
    rag_examples = []
    for pos, neg in zip(top_pos, top_neg):
        rag_examples.append(_to_few_shot_format(pos))
        rag_examples.append(_to_few_shot_format(neg))
    for e in top_pos[len(top_neg) :]:
        rag_examples.append(_to_few_shot_format(e))
    for e in top_neg[len(top_pos) :]:
        rag_examples.append(_to_few_shot_format(e))

    prompt = build_analysis_prompt(
        question, context=ctx, few_shot_examples=rag_examples, include_few_shot=True
    )
    content = _call_glm(
        client,
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    parsed = _parse_json_response(content)
    if parsed is None:
        return {
            "risk_score": -1,
            "risk_tags": [],
            "rationale": f"PARSE_ERROR: {content[:200]}",
        }
    return {
        "risk_score": max(0, min(100, int(parsed.get("risk_score", -1)))),
        "risk_tags": parsed.get("risk_tags", []),
        "rationale": parsed.get("rationale", ""),
    }


# ========================== SocREval evaluator ==============================


def socreval_evaluate(
    question: str,
    ground_truth: dict,
    condition_output: dict,
    client: ZhipuAI,
) -> Optional[dict]:
    prompt = SOCREVAL_EVALUATOR_PROMPT.format(
        question=question,
        gt_risk_score=ground_truth.get("risk_score", ""),
        gt_reasons=ground_truth.get("reasons", ""),
        pred_risk_score=condition_output.get("risk_score", ""),
        pred_risk_tags=json.dumps(
            condition_output.get("risk_tags", []), ensure_ascii=False
        ),
        pred_rationale=condition_output.get("rationale", ""),
    )
    content = _call_glm(client, [{"role": "user", "content": prompt}], temperature=0.2)
    parsed = _parse_json_response(content)
    if parsed is None:
        return None
    for dim in DIMENSIONS:
        if dim in parsed:
            parsed[dim] = max(1, min(5, int(parsed[dim])))
    return parsed


# ========================== Run evaluation ==================================


def load_dataset(n: int) -> List[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data[:n]


def load_stratified_dataset(per_tier: int = 10, seed: int = 42) -> List[dict]:
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    tiers = [
        ("low", 0, 22),
        ("mid", 23, 84),
        ("high", 85, 100),
    ]
    rng = random.Random(seed)
    selected = []
    for label, lo, hi in tiers:
        bucket = [d for d in data if lo <= d["risk_score"] <= hi]
        sample = rng.sample(bucket, min(per_tier, len(bucket)))
        selected.extend(sample)
    return selected


def load_cache() -> Dict[str, dict]:
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: Dict[str, dict]) -> None:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def run_evaluation(
    n_samples: int = 50, resume: bool = False, stratified: int = 0
) -> Dict[str, dict]:
    if stratified > 0:
        dataset = load_stratified_dataset(per_tier=stratified)
        cache_path = EVAL_DIR / "eval_results_stratified.json"
    else:
        dataset = load_dataset(n_samples)
        cache_path = RESULTS_PATH

    cache: Dict[str, dict] = {}
    if resume and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)
    client = _get_glm_client()

    total = len(dataset)
    for i, example in enumerate(dataset):
        key = example["prompt"]
        question = example["prompt"]
        gt = {"risk_score": example["risk_score"], "reasons": example["reasons"]}

        if key in cache and resume:
            entry = cache[key]
            all_complete = all(
                cond in entry and "error" not in entry.get(cond, {})
                for cond in CONDITIONS
            ) and all(f"{cond}_eval" in entry for cond in CONDITIONS)
            if all_complete:
                print(f"[{i + 1}/{total}] cached — skipping", flush=True)
                continue
            result_entry = entry
            print(f"[{i + 1}/{total}] {question[:70]}... (partial)", flush=True)
        else:
            result_entry: Dict[str, Any] = {"question": question, "gt": gt}
            print(f"[{i + 1}/{total}] {question[:70]}...", flush=True)

        # Condition A — Baseline
        if "A_baseline" not in result_entry:
            t0 = time.time()
            try:
                result_entry["A_baseline"] = generate_baseline(question, client)
                print(f"  A baseline: {time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                result_entry["A_baseline"] = {"error": str(e)}
                print(f"  A baseline: FAILED {e}", flush=True)

        # Condition B — System Basic
        if "B_system_basic" not in result_entry:
            t0 = time.time()
            try:
                result_entry["B_system_basic"] = generate_system_basic(
                    question, client=client
                )
                print(f"  B basic:    {time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                result_entry["B_system_basic"] = {"error": str(e)}
                print(f"  B basic:    FAILED {e}", flush=True)

        # Condition C — System Full (RAG few-shot + web search)
        if "C_system_full" not in result_entry:
            t0 = time.time()
            try:
                result_entry["C_system_full"] = generate_system_full(
                    question, client=client
                )
                print(f"  C full:     {time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                result_entry["C_system_full"] = {"error": str(e)}
                print(f"  C full:     FAILED {e}", flush=True)

        # SocREval evaluations
        for cond in CONDITIONS:
            eval_key = f"{cond}_eval"
            if eval_key in result_entry:
                continue
            cond_output = result_entry.get(cond, {})
            if "error" in cond_output:
                result_entry[eval_key] = None
                continue
            t0 = time.time()
            try:
                eval_result = socreval_evaluate(question, gt, cond_output, client)
                result_entry[eval_key] = eval_result
                print(f"  eval {cond}: {time.time() - t0:.1f}s", flush=True)
            except Exception as e:
                result_entry[eval_key] = {"error": str(e)}
                print(f"  eval {cond}: FAILED {e}", flush=True)

        cache[key] = result_entry
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        print(f"  → saved")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Results saved to {cache_path}")
    return cache


# ========================== Metrics =========================================


def compute_metrics(results: Dict[str, dict]) -> dict:
    metrics: Dict[str, Any] = {cond: {} for cond in CONDITIONS}

    for cond in CONDITIONS:
        socreval_scores: Dict[str, List[float]] = {d: [] for d in DIMENSIONS}
        pred_scores: List[float] = []
        gt_scores: List[float] = []
        errors: List[float] = []
        tag_overlaps: List[float] = []

        for key, entry in results.items():
            cond_output = entry.get(cond, {})
            if "error" in cond_output or cond_output.get("risk_score", -1) < 0:
                continue

            pred_score = float(cond_output["risk_score"])
            gt_score = float(entry["gt"]["risk_score"])
            pred_scores.append(pred_score)
            gt_scores.append(gt_score)
            errors.append(abs(pred_score - gt_score))

            # SocREval dimension scores
            eval_data = entry.get(f"{cond}_eval")
            if eval_data and "error" not in eval_data:
                for dim in DIMENSIONS:
                    val = eval_data.get(dim)
                    if val is not None:
                        socreval_scores[dim].append(float(val))

            # Tag overlap (Jaccard)
            pred_tags = set(cond_output.get("risk_tags", []))
            gt_reasons = entry["gt"].get("reasons", "").lower()
            gt_implied_tags = set()
            tag_keywords = {
                "ambiguous_time": [
                    "time",
                    "date",
                    "when",
                    "deadline",
                    "year",
                    "month",
                    "soon",
                ],
                "undefined_term": [
                    "undefined",
                    "vague term",
                    "unclear definition",
                    "not defined",
                    "not specified",
                ],
                "unverified_source": [
                    "source",
                    "authoritative",
                    "verification",
                    "official source",
                ],
                "vague_condition": [
                    "condition",
                    "resolution",
                    "criteria",
                    "how resolved",
                ],
                "ambiguous_quantity": [
                    "quantity",
                    "amount",
                    "threshold",
                    "how much",
                    "how many",
                    "degree",
                ],
                "unidentified_subject": [
                    "subject",
                    "entity",
                    "who",
                    "which company",
                    "unidentified",
                ],
                "high_disputability": [
                    "dispute",
                    "subjective",
                    "interpretation",
                    "ambiguous",
                    "controversial",
                ],
            }
            for tag, keywords in tag_keywords.items():
                if any(kw in gt_reasons for kw in keywords):
                    gt_implied_tags.add(tag)
            union = pred_tags | gt_implied_tags
            if union:
                tag_overlaps.append(len(pred_tags & gt_implied_tags) / len(union))
            else:
                tag_overlaps.append(1.0)

        n_valid = len(pred_scores)
        metrics[cond]["n_valid"] = n_valid
        metrics[cond]["mae"] = float(np.mean(errors)) if errors else None
        metrics[cond]["score_errors"] = errors

        if n_valid >= 2:
            try:
                spearman = stats.spearmanr(pred_scores, gt_scores)
                metrics[cond]["spearman_rho"] = float(spearman.correlation)
                metrics[cond]["spearman_p"] = float(spearman.pvalue)
            except Exception:
                metrics[cond]["spearman_rho"] = None
                metrics[cond]["spearman_p"] = None
        else:
            metrics[cond]["spearman_rho"] = None
            metrics[cond]["spearman_p"] = None

        metrics[cond]["tag_overlap"] = (
            float(np.mean(tag_overlaps)) if tag_overlaps else None
        )

        for dim in DIMENSIONS:
            scores = socreval_scores[dim]
            metrics[cond][f"avg_{dim}"] = float(np.mean(scores)) if scores else None

        # Somers' D: overall_quality vs |error|
        overall_q = socreval_scores["overall_quality"]
        if len(overall_q) >= 2 and len(errors[: len(overall_q)]) >= 2:
            paired_errors = errors[: len(overall_q)]
            try:
                d_val = _somers_d(overall_q, [-e for e in paired_errors])
                metrics[cond]["somers_d"] = d_val
            except Exception:
                metrics[cond]["somers_d"] = None
        else:
            metrics[cond]["somers_d"] = None

    return metrics


def _somers_d(x: List[float], y: List[float]) -> Optional[float]:
    """Compute Somers' D (asymmetric) using concordant/discordant pairs."""
    n = len(x)
    if n < 2:
        return None
    concordant = 0
    discordant = 0
    tied = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0:
                tied += 1
            elif dx * dy > 0:
                concordant += 1
            else:
                discordant += 1
    denom = concordant + discordant + tied
    if denom == 0:
        return None
    return (concordant - discordant) / denom


# ========================== Report ==========================================


def print_report(metrics: dict, results: Dict[str, dict]) -> str:
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("SocREval Evaluation Report")
    lines.append("=" * 70)
    lines.append("")

    lines.append(
        f"{'Condition':<25} {'MAE':>8} {'Spearman ρ':>12} {'Somers D':>10} {'Tag Jaccard':>12}"
    )
    lines.append("-" * 70)
    for cond in CONDITIONS:
        m = metrics[cond]
        mae_str = f"{m['mae']:.2f}" if m["mae"] is not None else "N/A"
        sp_str = f"{m['spearman_rho']:.3f}" if m["spearman_rho"] is not None else "N/A"
        sd_str = f"{m['somers_d']:.3f}" if m["somers_d"] is not None else "N/A"
        to_str = f"{m['tag_overlap']:.3f}" if m["tag_overlap"] is not None else "N/A"
        label = CONDITION_LABELS[cond]
        lines.append(f"{label:<25} {mae_str:>8} {sp_str:>12} {sd_str:>10} {to_str:>12}")
    lines.append("")

    lines.append("SocREval Dimension Scores (1-5):")
    lines.append(
        f"{'Condition':<25} {'Score':>8} {'Rationale':>10} {'Tags':>8} {'Overall':>10}"
    )
    lines.append("-" * 70)
    for cond in CONDITIONS:
        m = metrics[cond]
        label = CONDITION_LABELS[cond]
        sa = (
            f"{m['avg_score_accuracy']:.2f}"
            if m.get("avg_score_accuracy") is not None
            else "N/A"
        )
        rq = (
            f"{m['avg_rationale_quality']:.2f}"
            if m.get("avg_rationale_quality") is not None
            else "N/A"
        )
        tc = (
            f"{m['avg_tag_correctness']:.2f}"
            if m.get("avg_tag_correctness") is not None
            else "N/A"
        )
        oq = (
            f"{m['avg_overall_quality']:.2f}"
            if m.get("avg_overall_quality") is not None
            else "N/A"
        )
        lines.append(f"{label:<25} {sa:>8} {rq:>10} {tc:>8} {oq:>10}")
    lines.append("")

    n_total = len(results)
    lines.append(f"Total examples evaluated: {n_total}")
    lines.append("")

    report = "\n".join(lines)
    print(report)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved to {REPORT_PATH}")

    return report


# ========================== Plots ===========================================


def _style_ax(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)


def generate_plots(metrics: dict, results: Dict[str, dict]) -> None:
    cond_colors = {
        "A_baseline": "#e74c3c",
        "B_system_basic": "#3498db",
        "C_system_full": "#2ecc71",
    }

    # --- Plot 1: Grouped bar chart (3 conditions × 4 dimensions) ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(DIMENSIONS))
    width = 0.22
    for i, cond in enumerate(CONDITIONS):
        vals = []
        for dim in DIMENSIONS:
            v = metrics[cond].get(f"avg_{dim}")
            vals.append(v if v is not None else 0)
        bars = ax.bar(
            x + i * width,
            vals,
            width,
            label=CONDITION_LABELS[cond],
            color=cond_colors[cond],
            edgecolor="white",
        )
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(x + width)
    ax.set_xticklabels(
        [
            "Score\nAccuracy",
            "Rationale\nQuality",
            "Tag\nCorrectness",
            "Overall\nQuality",
        ]
    )
    _style_ax(
        ax,
        "SocREval Dimension Scores by Condition",
        "Evaluation Dimension",
        "Average Score (1-5)",
    )
    ax.set_ylim(0, 5.5)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path1 = EVAL_DIR / "eval_dimension_comparison.png"
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    print(f"Saved {path1}")

    # --- Plot 2: Scatter plot (predicted vs GT risk_score) ---
    fig, ax = plt.subplots(figsize=(8, 8))
    for cond in CONDITIONS:
        preds = []
        gts = []
        for key, entry in results.items():
            co = entry.get(cond, {})
            if "error" in co or co.get("risk_score", -1) < 0:
                continue
            preds.append(float(co["risk_score"]))
            gts.append(float(entry["gt"]["risk_score"]))
        ax.scatter(
            gts,
            preds,
            alpha=0.5,
            label=CONDITION_LABELS[cond],
            color=cond_colors[cond],
            s=30,
            edgecolors="white",
            linewidth=0.3,
        )
    ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="Perfect")
    _style_ax(
        ax,
        "Predicted vs Ground Truth Risk Score",
        "Ground Truth Risk Score",
        "Predicted Risk Score",
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path2 = EVAL_DIR / "eval_score_scatter.png"
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    print(f"Saved {path2}")

    # --- Plot 3: Boxplot of score errors ---
    fig, ax = plt.subplots(figsize=(8, 6))
    error_data = []
    labels_used = []
    for cond in CONDITIONS:
        errs = metrics[cond].get("score_errors", [])
        if errs:
            error_data.append(errs)
            labels_used.append(CONDITION_LABELS[cond])
    if error_data:
        bp = ax.boxplot(error_data, tick_labels=labels_used, patch_artist=True)
        for patch, cond in zip(bp["boxes"], CONDITIONS):
            patch.set_facecolor(cond_colors[cond])
            patch.set_alpha(0.6)
    _style_ax(
        ax,
        "Distribution of Score Error (|predicted - GT|)",
        "Condition",
        "Absolute Error",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path3 = EVAL_DIR / "eval_score_error_boxplot.png"
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    print(f"Saved {path3}")


# ========================== CLI =============================================


def main() -> None:
    parser = argparse.ArgumentParser(description="SocREval evaluation pipeline")
    parser.add_argument(
        "--resume", action="store_true", help="Resume from cached results"
    )
    parser.add_argument(
        "--report-only", action="store_true", help="Regenerate report/plots from cache"
    )
    parser.add_argument(
        "--n", type=int, default=50, help="Number of examples to evaluate (default: 50)"
    )
    parser.add_argument(
        "--stratified",
        type=int,
        default=0,
        help="Per-tier sample count for stratified sampling (e.g. 10 = 10 low + 10 mid + 10 high)",
    )
    args = parser.parse_args()

    if args.report_only:
        if args.stratified > 0:
            cache_path = EVAL_DIR / "eval_results_stratified.json"
        else:
            cache_path = RESULTS_PATH
        if not cache_path.exists():
            print("No cached results found. Run the evaluation first.")
            sys.exit(1)
        with open(cache_path, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = run_evaluation(
            n_samples=args.n,
            resume=args.resume,
            stratified=args.stratified,
        )

    metrics = compute_metrics(results)
    print_report(metrics, results)
    generate_plots(metrics, results)


if __name__ == "__main__":
    main()
