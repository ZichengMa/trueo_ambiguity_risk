"""
Local tests for rewrite suggestion helper.
"""

import os
import sys
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rewriter import _fallback_rewrites, _parse_rewrite_response, suggest_resolvable_rewrites


def test_parse_rewrite_response():
    content = """
    {
      "suggestions": [
        {
          "rewritten_question": "Will OpenAI release GPT-6 by March 31, 2026 23:59 UTC, based on OpenAI's official newsroom announcements?",
          "why_clearer": "Defines a concrete model name, deadline, timezone, and source."
        }
      ],
      "general_guidance": "Use explicit dates and authoritative sources."
    }
    """
    parsed = _parse_rewrite_response(content=content, max_suggestions=2)
    assert len(parsed.suggestions) == 1
    assert "March 31, 2026" in parsed.suggestions[0].rewritten_question
    assert parsed.general_guidance == "Use explicit dates and authoritative sources."


def test_fallback_rewrites_count():
    fallback = _fallback_rewrites(
        question="Will OpenAI release a new model in March this year?",
        max_suggestions=2,
    )
    assert len(fallback.suggestions) == 2
    assert fallback.general_guidance is not None


def test_suggest_resolvable_rewrites_falls_back():
    with patch("rewriter._suggest_with_llm", side_effect=RuntimeError("mock failure")):
        fallback = suggest_resolvable_rewrites(
            question="Will OpenAI release a new model in March this year?",
            risk_tags=["ambiguous_time"],
            rationale="This year is ambiguous.",
            max_suggestions=1,
        )
    assert len(fallback.suggestions) == 1
