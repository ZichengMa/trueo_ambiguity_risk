"""
Cost comparison: Plain evaluation vs SocREval evaluation.

Both approaches evaluate the same Condition C outputs. The only difference is
the evaluation prompt — Plain asks for 4 dimension scores directly, while
SocREval adds 3 Socratic reasoning steps (Dialectic, Maieutics, Definition)
before scoring.

Usage:
    .venv/bin/python evaluation/scripts/eval_cost.py --n 20
"""

import json
import argparse
import random
import re
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

from config import (
    DEEPSEEK_MODEL,
    PREDICTION_MARKET_EXAMPLES_PATH,
)
from eval_prompts import SOCREVAL_EVALUATOR_PROMPT, PLAIN_EVALUATOR_PROMPT
from llm_client import create_deepseek_client, deepseek_chat_options
from prompts import build_analysis_prompt, SYSTEM_PROMPT
from main import merge_analysis_context
from rag import (
    _tokenize,
    _build_idf,
    _cosine_sim,
    _to_few_shot_format,
    POSITIVE_MAX_SCORE,
    NEGATIVE_MIN_SCORE,
)
from search import WebSearchClient

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEEPSEEK_PRICING_PER_1M = {
    "deepseek-v4-flash": {
        "input_cache_hit": 0.0028,
        "input_cache_miss": 0.14,
        "output": 0.28,
    },
    "deepseek-v4-pro": {
        # Discounted DeepSeek-V4-Pro prices from the public pricing page.
        "input_cache_hit": 0.003625,
        "input_cache_miss": 0.435,
        "output": 0.87,
    },
    # Deprecated aliases map to deepseek-v4-flash compatibility modes.
    "deepseek-chat": {
        "input_cache_hit": 0.0028,
        "input_cache_miss": 0.14,
        "output": 0.28,
    },
    "deepseek-reasoner": {
        "input_cache_hit": 0.0028,
        "input_cache_miss": 0.14,
        "output": 0.28,
    },
}


def _get_pricing(model: str):
    if model not in DEEPSEEK_PRICING_PER_1M:
        supported = ", ".join(sorted(DEEPSEEK_PRICING_PER_1M))
        raise ValueError(
            f"No DeepSeek pricing configured for {model!r}. Supported: {supported}"
        )
    return DEEPSEEK_PRICING_PER_1M[model]


def _usage_value(usage, key: str, default: int = 0) -> int:
    value = getattr(usage, key, None)
    if value is None and hasattr(usage, "model_dump"):
        value = usage.model_dump().get(key)
    if value is None:
        value = default
    return int(value)


def _empty_usage():
    return {
        "prompt_tokens": 0,
        "prompt_cache_hit_tokens": 0,
        "prompt_cache_miss_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
    }


def _add_usage(total, usage):
    for key in total:
        total[key] += usage[key]


def _priced_usage(usage, model: str):
    prompt_tokens = _usage_value(usage, "prompt_tokens")
    cache_hit = _usage_value(usage, "prompt_cache_hit_tokens")
    cache_miss = _usage_value(
        usage,
        "prompt_cache_miss_tokens",
        default=max(0, prompt_tokens - cache_hit),
    )
    completion_tokens = _usage_value(usage, "completion_tokens")
    total_tokens = _usage_value(
        usage,
        "total_tokens",
        default=prompt_tokens + completion_tokens,
    )
    pricing = _get_pricing(model)
    cost_usd = (
        cache_hit * pricing["input_cache_hit"]
        + cache_miss * pricing["input_cache_miss"]
        + completion_tokens * pricing["output"]
    ) / 1_000_000
    return {
        "prompt_tokens": prompt_tokens,
        "prompt_cache_hit_tokens": cache_hit,
        "prompt_cache_miss_tokens": cache_miss,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost_usd,
    }


def _get_client():
    return create_deepseek_client()


def _call_with_usage(client, messages, temperature=0.3):
    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
        **deepseek_chat_options(json_output=True),
    )
    content = resp.choices[0].message.content
    return content, _priced_usage(resp.usage, DEEPSEEK_MODEL)


def _retrieve_rag_no_leak(question):
    with open(PREDICTION_MARKET_EXAMPLES_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
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
        return [
            e
            for e, _ in sorted(
                pool, key=lambda x: _cosine_sim(query_tokens, x[1], idf), reverse=True
            )
        ]

    top_pos, top_neg = rank(positives)[:5], rank(negatives)[:5]
    rag = []
    for p, n in zip(top_pos, top_neg):
        rag.append(_to_few_shot_format(p))
        rag.append(_to_few_shot_format(n))
    for e in top_pos[len(top_neg) :]:
        rag.append(_to_few_shot_format(e))
    for e in top_neg[len(top_pos) :]:
        rag.append(_to_few_shot_format(e))
    return rag


def load_stratified(per_tier=4, seed=42):
    with open(PREDICTION_MARKET_EXAMPLES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    selected = []
    for lo, hi in [(0, 22), (23, 84), (85, 100)]:
        bucket = [d for d in data if lo <= d["risk_score"] <= hi]
        selected.extend(rng.sample(bucket, min(per_tier, len(bucket))))
    return selected


def load_stratified_n(n=20, seed=42):
    with open(PREDICTION_MARKET_EXAMPLES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    rng = random.Random(seed)
    tiers = [
        ("low", 0, 22),
        ("mid", 23, 84),
        ("high", 85, 100),
    ]
    base = n // len(tiers)
    remainder = n % len(tiers)
    selected = []
    tier_counts = {}
    for i, (label, lo, hi) in enumerate(tiers):
        target = base + (1 if i < remainder else 0)
        bucket = [d for d in data if lo <= d["risk_score"] <= hi]
        sample = rng.sample(bucket, min(target, len(bucket)))
        selected.extend({**item, "_tier": label} for item in sample)
        tier_counts[label] = len(sample)
    return selected, tier_counts


def _fmt_eval_kwargs(question, gt, cond_out):
    return dict(
        question=question,
        gt_risk_score=gt["risk_score"],
        gt_reasons=gt["reasons"],
        pred_risk_score=cond_out["risk_score"],
        pred_risk_tags=json.dumps(cond_out["risk_tags"], ensure_ascii=False),
        pred_rationale=cond_out["rationale"],
    )


def _plot_cost_summary(out, chart_path: Path):
    avg = out["averages_per_sample"]
    cost_per_1k = {
        "Generation": avg["gen"]["cost_usd"] * 1000,
        "Plain eval": avg["plain"]["cost_usd"] * 1000,
        "SocREval eval": avg["socreval"]["cost_usd"] * 1000,
        "Full plain": (
            avg["gen"]["cost_usd"] + avg["plain"]["cost_usd"]
        )
        * 1000,
        "Full SocR": (
            avg["gen"]["cost_usd"] + avg["socreval"]["cost_usd"]
        )
        * 1000,
    }

    plain = avg["plain"]
    socr = avg["socreval"]
    token_parts = ["prompt_cache_hit_tokens", "prompt_cache_miss_tokens", "completion_tokens"]
    token_labels = ["Prompt cache hit", "Prompt cache miss", "Completion"]
    colors = ["#6aa84f", "#3c78d8", "#e69138"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    labels = list(cost_per_1k)
    values = [cost_per_1k[label] for label in labels]
    bar_colors = ["#8e9aaf", "#4f81bd", "#c0504d", "#7aa6c2", "#d9827f"]
    bars = ax.bar(labels, values, color=bar_colors)
    ax.set_title("DeepSeek Cost per 1,000 Examples")
    ax.set_ylabel("USD")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"${value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[1]
    x_labels = ["Plain eval", "SocREval eval"]
    bottoms = [0, 0]
    for part, label, color in zip(token_parts, token_labels, colors):
        vals = [plain[part], socr[part]]
        ax.bar(x_labels, vals, bottom=bottoms, label=label, color=color)
        bottoms = [b + v for b, v in zip(bottoms, vals)]
    ax.set_title("Evaluator Tokens per Example")
    ax.set_ylabel("Tokens")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    for i, total in enumerate(bottoms):
        ax.text(i, total, f"{total:.0f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle(
        f"Cost Comparison on {out['n']} Examples ({out['model']})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(chart_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Compare DeepSeek token and dollar cost for plain vs SocREval evaluation."
    )
    parser.add_argument("--n", type=int, default=20, help="Number of examples to run")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "eval_cost_comparison.json",
        help="JSON output path",
    )
    parser.add_argument(
        "--chart",
        type=Path,
        default=IMAGES_DIR / "eval_cost_comparison.png",
        help="Chart output path",
    )
    args = parser.parse_args()

    dataset, tier_counts = load_stratified_n(n=args.n, seed=args.seed)
    client = _get_client()

    gen_total = _empty_usage()
    plain_total = _empty_usage()
    socreval_total = _empty_usage()
    rows = []

    pricing = _get_pricing(DEEPSEEK_MODEL)
    print(f"Model: {DEEPSEEK_MODEL}")
    print(f"Examples: {len(dataset)} stratified samples {tier_counts}")
    print(
        "Pricing per 1M tokens: "
        f"input cache hit=${pricing['input_cache_hit']}, "
        f"input cache miss=${pricing['input_cache_miss']}, "
        f"output=${pricing['output']}"
    )

    for i, ex in enumerate(dataset):
        q = ex["prompt"]
        gt = {"risk_score": ex["risk_score"], "reasons": ex["reasons"]}
        print(f"[{i + 1}/{len(dataset)}] {q[:60]}...", flush=True)

        # Generation (Condition C) — same for both
        try:
            web_ctx = WebSearchClient().build_context(q)
            ctx = merge_analysis_context(web_search_context=web_ctx)
            rag = _retrieve_rag_no_leak(q)
            prompt = build_analysis_prompt(
                q, context=ctx, few_shot_examples=rag, include_few_shot=True
            )
            content, gu = _call_with_usage(
                client,
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
        except Exception as e:
            print(f"  gen FAILED: {str(e)[:80]}, skipping", flush=True)
            continue
        _add_usage(gen_total, gu)

        m = re.search(r"\{[\s\S]*\}", content)
        parsed = json.loads(m.group(0)) if m else {}
        cond_out = {
            "risk_score": max(0, min(100, int(parsed.get("risk_score", -1)))),
            "risk_tags": parsed.get("risk_tags", []),
            "rationale": parsed.get("rationale", ""),
        }

        kw = _fmt_eval_kwargs(q, gt, cond_out)

        # Plain evaluation
        plain_prompt = PLAIN_EVALUATOR_PROMPT.format(**kw)
        _, pu = _call_with_usage(
            client, [{"role": "user", "content": plain_prompt}], temperature=0.2
        )
        _add_usage(plain_total, pu)

        # SocREval evaluation
        socreval_prompt = SOCREVAL_EVALUATOR_PROMPT.format(**kw)
        _, su = _call_with_usage(
            client, [{"role": "user", "content": socreval_prompt}], temperature=0.2
        )
        _add_usage(socreval_total, su)

        rows.append(
            {
                "question": q[:60],
                "tier": ex.get("_tier"),
                "gt": gt["risk_score"],
                "gen": gu,
                "plain_eval": pu,
                "socreval_eval": su,
            }
        )
        print(
            f"  gen={gu['total_tokens']}  plain_eval={pu['total_tokens']}  socreval_eval={su['total_tokens']}  "
            f"overhead=+{su['total_tokens'] - pu['total_tokens']} ({(su['total_tokens'] / pu['total_tokens'] - 1) * 100:.0f}%)  "
            f"plain=${pu['cost_usd']:.6f}  socreval=${su['cost_usd']:.6f}",
            flush=True,
        )
        time.sleep(1)

    n = len(rows)
    if n == 0:
        print("No valid samples.")
        return

    avg_gen = {k: v / n for k, v in gen_total.items()}
    avg_plain = {k: v / n for k, v in plain_total.items()}
    avg_socreval = {k: v / n for k, v in socreval_total.items()}
    overhead_pct = (
        (socreval_total["total_tokens"] - plain_total["total_tokens"])
        / plain_total["total_tokens"]
        * 100
    )
    delta_prompt = avg_socreval["prompt_tokens"] - avg_plain["prompt_tokens"]
    delta_completion = (
        avg_socreval["completion_tokens"] - avg_plain["completion_tokens"]
    )
    cost_overhead_pct = (
        (socreval_total["cost_usd"] - plain_total["cost_usd"])
        / plain_total["cost_usd"]
        * 100
    )
    delta_eval_cost = avg_socreval["cost_usd"] - avg_plain["cost_usd"]

    print("\n" + "=" * 75)
    print("COST COMPARISON: Plain Evaluation vs SocREval Evaluation")
    print("=" * 75)
    print(f"\nModel: {DEEPSEEK_MODEL}")
    print(
        "DeepSeek prices per 1M tokens: "
        f"cache hit input=${pricing['input_cache_hit']}, "
        f"cache miss input=${pricing['input_cache_miss']}, "
        f"output=${pricing['output']}"
    )
    print(f"\nSamples: {n}")
    print(f"\n{'':30} {'Plain Eval':>14} {'SocREval':>14} {'Difference':>14}")
    print("-" * 75)
    for m in [
        "prompt_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "completion_tokens",
        "total_tokens",
    ]:
        d = socreval_total[m] - plain_total[m]
        print(
            f"  Total {m:<19} {plain_total[m]:>14,} {socreval_total[m]:>14,} {d:>+14,}"
        )
    print(
        f"  Total {'cost_usd':<19} {plain_total['cost_usd']:>14.6f} "
        f"{socreval_total['cost_usd']:>14.6f} "
        f"{socreval_total['cost_usd'] - plain_total['cost_usd']:>+14.6f}"
    )
    print(f"\n  Per-sample averages:")
    for m in [
        "prompt_tokens",
        "prompt_cache_hit_tokens",
        "prompt_cache_miss_tokens",
        "completion_tokens",
        "total_tokens",
    ]:
        d = avg_socreval[m] - avg_plain[m]
        print(
            f"    {m:<23} {avg_plain[m]:>11,.0f} {avg_socreval[m]:>11,.0f} {d:>+11,.0f}"
        )
    print(
        f"    {'cost_usd':<23} {avg_plain['cost_usd']:>11.6f} "
        f"{avg_socreval['cost_usd']:>11.6f} {delta_eval_cost:>+11.6f}"
    )
    print(f"\n  SocREval overhead: +{overhead_pct:.1f}% total tokens")
    print(f"  SocREval cost overhead: +{cost_overhead_pct:.1f}% dollars")
    print(
        f"    Prompt overhead:     +{delta_prompt:,.0f} tokens/sample (Socratic method instructions)"
    )
    print(
        f"    Completion overhead: +{delta_completion:,.0f} tokens/sample (own_assessment + qualitative_analysis)"
    )
    print()
    print(
        f"  Generation cost:       {avg_gen['total_tokens']:,.0f} tokens/sample "
        f"(${avg_gen['cost_usd']:.6f}/sample)"
    )
    print(
        f"  Plain eval cost:       {avg_plain['total_tokens']:,.0f} tokens/sample "
        f"(${avg_plain['cost_usd']:.6f}/sample)"
    )
    print(
        f"  SocREval eval cost:    {avg_socreval['total_tokens']:,.0f} tokens/sample "
        f"(${avg_socreval['cost_usd']:.6f}/sample)"
    )
    print(
        f"  Full pipeline (plain): {avg_gen['total_tokens'] + avg_plain['total_tokens']:,.0f} tokens/sample "
        f"(${avg_gen['cost_usd'] + avg_plain['cost_usd']:.6f}/sample)"
    )
    print(
        f"  Full pipeline (SocR):  {avg_gen['total_tokens'] + avg_socreval['total_tokens']:,.0f} tokens/sample "
        f"(${avg_gen['cost_usd'] + avg_socreval['cost_usd']:.6f}/sample)"
    )

    out = {
        "n": n,
        "model": DEEPSEEK_MODEL,
        "sample_tier_counts": tier_counts,
        "sampling_seed": args.seed,
        "pricing_per_1m_tokens_usd": pricing,
        "gen_total": gen_total,
        "plain_total": plain_total,
        "socreval_total": socreval_total,
        "avg_gen": avg_gen,
        "avg_plain": avg_plain,
        "avg_socreval": avg_socreval,
        "averages_per_sample": {
            "gen": avg_gen,
            "plain": avg_plain,
            "socreval": avg_socreval,
        },
        "full_pipeline_avg": {
            "plain_tokens": avg_gen["total_tokens"] + avg_plain["total_tokens"],
            "socreval_tokens": avg_gen["total_tokens"]
            + avg_socreval["total_tokens"],
            "plain_cost_usd": avg_gen["cost_usd"] + avg_plain["cost_usd"],
            "socreval_cost_usd": avg_gen["cost_usd"] + avg_socreval["cost_usd"],
        },
        "overhead_pct": overhead_pct,
        "cost_overhead_pct": cost_overhead_pct,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.chart.parent.mkdir(parents=True, exist_ok=True)
    _plot_cost_summary(out, args.chart)
    out["chart_path"] = str(args.chart)
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {args.output}")
    print(f"Saved chart to {args.chart}")


if __name__ == "__main__":
    main()
