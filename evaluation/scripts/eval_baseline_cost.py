"""
DeepSeek cost comparison for generation Conditions A, B, and C.

Condition A: minimal baseline prompt only.
Condition B: system prompt + scoring rubric + default few-shot examples.
Condition C: Condition B plus RAG-retrieved examples and Tavily web-search context.

Usage:
    .venv/bin/python evaluation/scripts/eval_baseline_cost.py --n 20
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
PROJECT_ROOT = EVAL_DIR.parent
RESULTS_DIR = EVAL_DIR / "results"
IMAGES_DIR = EVAL_DIR / "images"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import DEEPSEEK_MODEL
from eval_cost import (
    _add_usage,
    _call_with_usage,
    _empty_usage,
    _get_client,
    _get_pricing,
    _retrieve_rag_no_leak,
    load_stratified_n,
)
from eval_prompts import BASELINE_PROMPT_TEMPLATE
from main import merge_analysis_context
from prompts import SYSTEM_PROMPT, build_analysis_prompt
from search import WebSearchClient


CONDITIONS = {
    "A_baseline": "A: Minimal baseline",
    "B_system_basic": "B: System + few-shot",
    "C_system_full": "C: RAG + web search",
}


def _cost_request_for_condition(client, condition: str, question: str):
    if condition == "A_baseline":
        prompt = BASELINE_PROMPT_TEMPLATE.format(question=question)
        return _call_with_usage(
            client,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
        )

    if condition == "B_system_basic":
        prompt = build_analysis_prompt(question, include_few_shot=True)
        return _call_with_usage(
            client,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

    if condition == "C_system_full":
        web_ctx = WebSearchClient().build_context(question)
        ctx = merge_analysis_context(web_search_context=web_ctx)
        rag = _retrieve_rag_no_leak(question)
        prompt = build_analysis_prompt(
            question,
            context=ctx,
            few_shot_examples=rag,
            include_few_shot=True,
        )
        return _call_with_usage(
            client,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )

    raise ValueError(f"Unknown condition: {condition}")


def _average_usage(total, count: int):
    if count <= 0:
        return _empty_usage()
    return {key: value / count for key, value in total.items()}


def _plot_baseline_cost(out, chart_path: Path):
    labels = list(CONDITIONS)
    display_labels = ["A", "B", "C"]
    colors = ["#8e9aaf", "#4f81bd", "#c0504d"]

    avg = out["averages_per_success"]
    cost_per_1k = [avg[key]["cost_usd"] * 1000 for key in labels]

    token_parts = [
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "completion_tokens",
    ]
    token_labels = ["Prompt cache hit", "Prompt cache miss", "Completion"]
    token_colors = ["#6aa84f", "#3c78d8", "#e69138"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(display_labels, cost_per_1k, color=colors)
    ax.set_title("Generation Cost per 1,000 Examples")
    ax.set_ylabel("USD")
    ax.set_xlabel("Condition")
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, cost_per_1k):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1]
    bottoms = [0, 0, 0]
    for part, label, color in zip(token_parts, token_labels, token_colors):
        vals = [avg[key][part] for key in labels]
        ax.bar(display_labels, vals, bottom=bottoms, label=label, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title("Generation Tokens per Example")
    ax.set_ylabel("Tokens")
    ax.set_xlabel("Condition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    for i, total in enumerate(bottoms):
        ax.text(i, total, f"{total:.0f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"A/B/C Baseline Cost on {out['n_requested']} Examples ({out['model']})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare DeepSeek generation cost across A/B/C baselines."
    )
    parser.add_argument("--n", type=int, default=20, help="Number of examples to run")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "eval_baseline_cost_comparison.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=IMAGES_DIR / "eval_baseline_cost_comparison.png",
        help="Chart output path",
    )
    args = parser.parse_args()

    dataset, tier_counts = load_stratified_n(n=args.n, seed=args.seed)
    client = _get_client()
    pricing = _get_pricing(DEEPSEEK_MODEL)
    totals = {condition: _empty_usage() for condition in CONDITIONS}
    successes = {condition: 0 for condition in CONDITIONS}
    errors = {condition: [] for condition in CONDITIONS}
    rows = []

    print(f"Model: {DEEPSEEK_MODEL}")
    print(f"Examples: {len(dataset)} stratified samples {tier_counts}")
    print(
        "Pricing per 1M tokens: "
        f"input cache hit=${pricing['input_cache_hit']}, "
        f"input cache miss=${pricing['input_cache_miss']}, "
        f"output=${pricing['output']}"
    )

    for i, example in enumerate(dataset, 1):
        question = example["prompt"]
        print(f"[{i}/{len(dataset)}] {question[:60]}...", flush=True)
        row = {
            "question": question[:120],
            "tier": example.get("_tier"),
            "gt": example.get("risk_score"),
            "conditions": {},
        }

        for condition, label in CONDITIONS.items():
            try:
                _, usage = _cost_request_for_condition(client, condition, question)
                _add_usage(totals[condition], usage)
                successes[condition] += 1
                row["conditions"][condition] = usage
                print(
                    f"  {label}: {usage['total_tokens']} tokens  ${usage['cost_usd']:.6f}",
                    flush=True,
                )
            except Exception as exc:
                message = str(exc)[:200]
                errors[condition].append({"question": question[:120], "error": message})
                row["conditions"][condition] = {"error": message}
                print(f"  {label}: FAILED {message}", flush=True)
        rows.append(row)
        time.sleep(1)

    averages = {
        condition: _average_usage(totals[condition], successes[condition])
        for condition in CONDITIONS
    }

    print("\n" + "=" * 78)
    print("GENERATION COST COMPARISON: A vs B vs C")
    print("=" * 78)
    print(f"\nModel: {DEEPSEEK_MODEL}")
    print(f"Samples requested: {len(dataset)}")
    print(f"\n{'Condition':<24} {'Success':>8} {'Tokens/sample':>15} {'USD/sample':>14} {'USD/1k':>12}")
    print("-" * 78)
    for condition, label in CONDITIONS.items():
        avg = averages[condition]
        print(
            f"{label:<24} {successes[condition]:>8} "
            f"{avg['total_tokens']:>15,.0f} "
            f"{avg['cost_usd']:>14.6f} "
            f"{avg['cost_usd'] * 1000:>12.3f}"
        )

    out = {
        "n_requested": len(dataset),
        "model": DEEPSEEK_MODEL,
        "sample_tier_counts": tier_counts,
        "sampling_seed": args.seed,
        "pricing_per_1m_tokens_usd": pricing,
        "condition_labels": CONDITIONS,
        "successes": successes,
        "errors": errors,
        "totals": totals,
        "averages_per_success": averages,
        "rows": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.chart.parent.mkdir(parents=True, exist_ok=True)
    _plot_baseline_cost(out, args.chart)
    out["chart_path"] = str(args.chart)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")
    print(f"Saved chart to {args.chart}")


if __name__ == "__main__":
    main()
