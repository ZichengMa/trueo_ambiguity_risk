"""
Test module for the Market Prompt Ambiguity Risk Scoring System.

This module contains test cases to validate the functionality of the
risk scoring system.
"""

import sys
import os
import json
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import analyze_market_prompt
from models import RiskScoreResult, SearchContext, SearchEvidenceItem
from config import FEW_SHOT_EXAMPLES_PATH
from search import WebSearchClient, format_search_context


def test_basic_analysis():
    """
    Test basic analysis with a simple market question.
    
    This test verifies that:
    1. The function returns a valid RiskScoreResult
    2. The risk_score is within 0-100 range
    3. The risk_tags is a list
    4. The rationale is a non-empty string
    """
    print("\n" + "="*60)
    print("TEST: Basic Analysis")
    print("="*60)
    
    question = "Will OpenAI release a new model in March this year?"
    
    print(f"\nInput Question: {question}")
    print("\nCalling API... (this may take a few seconds)")
    
    result = analyze_market_prompt(question)
    
    # Validate result type
    assert isinstance(result, RiskScoreResult), "Result should be RiskScoreResult"
    print(f"\n✓ Result type: RiskScoreResult")
    
    # Validate risk_score range
    assert 0 <= result.risk_score <= 100, "Risk score must be 0-100"
    print(f"✓ Risk score: {result.risk_score}/100")
    
    # Validate risk_tags
    assert isinstance(result.risk_tags, list), "Risk tags must be a list"
    print(f"✓ Risk tags: {result.risk_tags}")
    
    # Validate rationale
    assert isinstance(result.rationale, str), "Rationale must be a string"
    assert len(result.rationale) > 0, "Rationale must not be empty"
    print(f"✓ Rationale: {result.rationale[:100]}...")
    
    print("\n" + "="*60)
    print("✅ TEST PASSED: Basic Analysis")
    print("="*60)
    
    return result


def test_output_format():
    """
    Test that the output format matches the expected JSON structure.
    """
    print("\n" + "="*60)
    print("TEST: Output Format Validation")
    print("="*60)
    
    question = "Will Bitcoin exceed $100,000 USD on December 31, 2025?"
    
    print(f"\nInput Question: {question}")
    print("\nCalling API...")
    
    result = analyze_market_prompt(question)
    
    # Convert to dict
    result_dict = result.model_dump()
    
    print(f"\nOutput JSON:")
    print(result.model_dump_json(indent=2))
    
    # Validate JSON structure
    assert "risk_score" in result_dict, "Missing 'risk_score' field"
    assert "risk_tags" in result_dict, "Missing 'risk_tags' field"
    assert "rationale" in result_dict, "Missing 'rationale' field"
    
    print("\n✓ Output contains all required fields:")
    print("  - risk_score: integer")
    print("  - risk_tags: list")
    print("  - rationale: string")
    
    print("\n" + "="*60)
    print("✅ TEST PASSED: Output Format Validation")
    print("="*60)
    
    return result


def test_high_risk_question():
    """
    Test with a question that should have high risk score.
    """
    print("\n" + "="*60)
    print("TEST: High Risk Question Detection")
    print("="*60)
    
    # This question is intentionally vague
    question = "Will something important happen soon?"
    
    print(f"\nInput Question: {question}")
    print("\nCalling API...")
    
    result = analyze_market_prompt(question, use_few_shot=False)
    
    print(f"\nRisk Score: {result.risk_score}/100")
    print(f"Risk Tags: {result.risk_tags}")
    print(f"Rationale: {result.rationale}")
    
    # We expect this question to have a high risk score
    print(f"\n✓ High ambiguity detected: {result.risk_score >= 50}")
    
    print("\n" + "="*60)
    print("✅ TEST PASSED: High Risk Question Detection")
    print("="*60)
    
    return result


def test_few_shot_examples_file():
    """
    Test that examples.json exists and follows the expected schema.
    """
    print("\n" + "="*60)
    print("TEST: Few-Shot Examples File Validation")
    print("="*60)

    assert os.path.exists(FEW_SHOT_EXAMPLES_PATH), "examples.json file is missing"

    with open(FEW_SHOT_EXAMPLES_PATH, "r", encoding="utf-8") as f:
        examples = json.load(f)

    assert isinstance(examples, list), "examples.json must contain a list"
    assert len(examples) > 0, "examples.json must contain at least one example"

    for index, example in enumerate(examples, 1):
        assert isinstance(example, dict), f"Example {index} must be an object"
        assert "question" in example, f"Example {index} is missing 'question'"
        assert "result" in example, f"Example {index} is missing 'result'"
        assert isinstance(example["question"], str), f"Example {index} question must be a string"
        assert example["question"].strip(), f"Example {index} question must not be empty"
        assert isinstance(example["result"], dict), f"Example {index} result must be an object"

        result = example["result"]
        assert "risk_score" in result, f"Example {index} result is missing 'risk_score'"
        assert "risk_tags" in result, f"Example {index} result is missing 'risk_tags'"
        assert "rationale" in result, f"Example {index} result is missing 'rationale'"
        assert isinstance(result["risk_score"], int), f"Example {index} risk_score must be an integer"
        assert 0 <= result["risk_score"] <= 100, f"Example {index} risk_score must be between 0 and 100"
        assert isinstance(result["risk_tags"], list), f"Example {index} risk_tags must be a list"
        assert all(isinstance(tag, str) for tag in result["risk_tags"]), f"Example {index} risk_tags must contain only strings"
        assert isinstance(result["rationale"], str), f"Example {index} rationale must be a string"
        assert result["rationale"].strip(), f"Example {index} rationale must not be empty"

    print(f"\n✓ Validated examples file: {FEW_SHOT_EXAMPLES_PATH}")
    print(f"✓ Example count: {len(examples)}")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Few-Shot Examples File Validation")
    print("="*60)


def test_format_search_context():
    """
    Test that structured search evidence is formatted for prompt injection.
    """
    print("\n" + "="*60)
    print("TEST: Web Search Context Formatting")
    print("="*60)

    search_context = SearchContext(
        query="OpenAI release new model official source definition resolution criteria",
        provider="tavily",
        summary="Multiple official release channels may create resolution ambiguity.",
        evidence=[
            SearchEvidenceItem(
                title="OpenAI News",
                url="https://openai.com/news",
                snippet="Official announcements are published on the OpenAI website.",
                source="openai.com",
                score=0.95,
                published_date="2025-03-01",
            )
        ],
    )

    formatted = format_search_context(search_context)

    assert "Web Search Evidence:" in formatted
    assert "Provider: tavily" in formatted
    assert "Top Sources:" in formatted
    assert "OpenAI News" in formatted
    assert "https://openai.com/news" in formatted

    print("\n✓ Search context formatting includes provider, evidence, and URL")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Web Search Context Formatting")
    print("="*60)


def test_web_search_requires_api_key():
    """
    Test that explicit web search fails fast without an API key.
    """
    print("\n" + "="*60)
    print("TEST: Web Search API Key Requirement")
    print("="*60)

    with patch("search.TAVILY_API_KEY", ""):
        try:
            WebSearchClient(api_key="")
            raise AssertionError("Expected WebSearchClient to require an API key")
        except ValueError as exc:
            assert "TAVILY_API_KEY" in str(exc)

    print("\n✓ Missing Tavily API key is reported clearly")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Web Search API Key Requirement")
    print("="*60)


def test_analyze_market_prompt_with_web_search():
    """
    Test that analyze_market_prompt merges caller context with web-search evidence.
    """
    print("\n" + "="*60)
    print("TEST: Analyze Market Prompt With Web Search")
    print("="*60)

    captured = {}

    class FakeSearchClient:
        def build_context(self, question: str) -> str:
            captured["search_question"] = question
            return "Web Search Evidence:\nProvider: tavily\nTop Sources:\n1. Official Blog"

    class FakeScorer:
        def score(self, question: str, context=None, include_few_shot: bool = True):
            captured["score_question"] = question
            captured["score_context"] = context
            captured["include_few_shot"] = include_few_shot
            return RiskScoreResult(
                risk_score=55,
                risk_tags=["unverified_source"],
                rationale="Test result from mocked scorer.",
            )

    with patch("main.WebSearchClient", FakeSearchClient), patch("main.RiskScorer", FakeScorer):
        result = analyze_market_prompt(
            "Will OpenAI release a new model in March this year?",
            context="Caller supplied context.",
            use_few_shot=False,
            use_web_search=True,
        )

    assert captured["search_question"] == "Will OpenAI release a new model in March this year?"
    assert "User-Provided Context:" in captured["score_context"]
    assert "Caller supplied context." in captured["score_context"]
    assert "Web Search Evidence:" in captured["score_context"]
    assert captured["include_few_shot"] is False
    assert isinstance(result, RiskScoreResult)

    print("\n✓ Web search evidence is merged into scorer context")

    print("\n" + "="*60)
    print("✅ TEST PASSED: Analyze Market Prompt With Web Search")
    print("="*60)


def run_all_tests():
    """
    Run all tests and report results.
    """
    print("\n" + "="*60)
    print("RUNNING ALL TESTS")
    print("="*60)
    
    tests = [
        ("Few-Shot Examples File", test_few_shot_examples_file),
        ("Web Search Context Formatting", test_format_search_context),
        ("Web Search API Key Requirement", test_web_search_requires_api_key),
        ("Web Search Integration", test_analyze_market_prompt_with_web_search),
        ("Basic Analysis", test_basic_analysis),
        ("Output Format", test_output_format),
        ("High Risk Detection", test_high_risk_question),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"Error: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run risk scorer tests")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only the basic test"
    )
    
    args = parser.parse_args()
    
    if args.quick:
        result = test_basic_analysis()
        print("\n" + "="*60)
        print("FINAL OUTPUT")
        print("="*60)
        print(result.model_dump_json(indent=2))
    else:
        success = run_all_tests()
        sys.exit(0 if success else 1)
