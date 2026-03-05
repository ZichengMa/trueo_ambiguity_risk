"""
Question rewrite helper for generating more objectively resolvable prompts.
"""

import json
import re
from typing import Optional

from zhipuai import ZhipuAI

from config import ZHIPU_API_KEY, ZHIPU_MODEL
from models import RewriteSuggestionItem, RewriteSuggestions


REWRITE_SYSTEM_PROMPT = """You are a prediction market resolution expert.
Your task is to rewrite ambiguous market questions into versions that are objectively resolvable.

Rules:
1. Keep the original intent.
2. Make time windows explicit with exact dates/times/timezone.
3. Define key terms clearly.
4. Specify objective resolution sources (official docs, websites, filings, exchange API, etc.).
5. Return concise, production-ready market wording.
6. Return valid JSON only."""


def suggest_resolvable_rewrites(
    question: str,
    risk_tags: Optional[list[str]] = None,
    rationale: Optional[str] = None,
    search_summary: Optional[str] = None,
    max_suggestions: int = 2,
) -> RewriteSuggestions:
    """
    Generate rewrite suggestions for a market question.

    This function uses an LLM first, then falls back to deterministic rewrites
    if LLM output is unavailable or invalid.
    """
    max_suggestions = max(1, min(3, int(max_suggestions)))

    try:
        suggestions = _suggest_with_llm(
            question=question,
            risk_tags=risk_tags or [],
            rationale=rationale,
            search_summary=search_summary,
            max_suggestions=max_suggestions,
        )
        if suggestions.suggestions:
            return suggestions
    except Exception:
        pass

    return _fallback_rewrites(question=question, max_suggestions=max_suggestions)


def _suggest_with_llm(
    question: str,
    risk_tags: list[str],
    rationale: Optional[str],
    search_summary: Optional[str],
    max_suggestions: int,
) -> RewriteSuggestions:
    if not ZHIPU_API_KEY:
        raise ValueError("ZHIPU_API_KEY is missing")

    client = ZhipuAI(api_key=ZHIPU_API_KEY)

    risk_tags_str = ", ".join(risk_tags) if risk_tags else "none"
    rationale_str = rationale or "none"
    search_summary_str = search_summary or "none"

    user_prompt = f"""Rewrite this prediction market question into {max_suggestions} clearer versions.

Original Question:
{question}

Current Risk Tags:
{risk_tags_str}

Current Rationale:
{rationale_str}

Web Search Summary:
{search_summary_str}

Return JSON in this schema:
{{
  "suggestions": [
    {{
      "rewritten_question": "<string>",
      "why_clearer": "<string>"
    }}
  ],
  "general_guidance": "<string>"
}}"""

    response = client.chat.completions.create(
        model=ZHIPU_MODEL,
        messages=[
            {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    content = response.choices[0].message.content
    return _parse_rewrite_response(content=content, max_suggestions=max_suggestions)


def _parse_rewrite_response(content: str, max_suggestions: int) -> RewriteSuggestions:
    json_match = re.search(r"\{[\s\S]*\}", content)
    if not json_match:
        raise ValueError("Could not find JSON in rewrite response")

    parsed = json.loads(json_match.group(0))
    raw_suggestions = parsed.get("suggestions", [])

    suggestions = []
    for item in raw_suggestions:
        if not isinstance(item, dict):
            continue
        rewritten_question = str(item.get("rewritten_question", "")).strip()
        why_clearer = str(item.get("why_clearer", "")).strip()
        if not rewritten_question:
            continue
        if not why_clearer:
            why_clearer = "Adds explicit terms and objective resolution criteria."
        suggestions.append(
            RewriteSuggestionItem(
                rewritten_question=rewritten_question,
                why_clearer=why_clearer,
            )
        )
        if len(suggestions) >= max_suggestions:
            break

    if not suggestions:
        raise ValueError("No rewrite suggestions parsed")

    general_guidance = parsed.get("general_guidance")
    if general_guidance is not None:
        general_guidance = str(general_guidance).strip() or None

    return RewriteSuggestions(
        suggestions=suggestions,
        general_guidance=general_guidance,
    )


def _fallback_rewrites(question: str, max_suggestions: int) -> RewriteSuggestions:
    base = question.strip().rstrip("?")

    rewrite_1 = (
        f"{base} by March 31, 2026 23:59 UTC, based on an official announcement on the organization's website or official press/blog channels?"
    )
    rewrite_2 = (
        f"{base}, where 'release' means generally available to the public (not limited beta), resolved using official release notes or product documentation published by March 31, 2026 23:59 UTC?"
    )
    rewrite_3 = (
        f"{base} before March 31, 2026 23:59 UTC, with resolution determined by a single primary source: the organization's official website announcement page?"
    )

    candidates = [rewrite_1, rewrite_2, rewrite_3][:max_suggestions]
    suggestions = [
        RewriteSuggestionItem(
            rewritten_question=candidate,
            why_clearer="Specifies timeframe, definition boundaries, and source-based resolution criteria.",
        )
        for candidate in candidates
    ]

    return RewriteSuggestions(
        suggestions=suggestions,
        general_guidance="Prefer explicit deadlines, strict term definitions, and one authoritative resolution source.",
    )
