"""
Main entry point for the Market Prompt Ambiguity Risk Scoring System.

This module provides the primary interface for analyzing market prompts
and detecting ambiguity risks.
"""

from typing import Optional

from scorer import RiskScorer
from models import RiskScoreResult, MarketProposal
from search import WebSearchClient


def merge_analysis_context(
    context: Optional[str] = None,
    web_search_context: Optional[str] = None
) -> Optional[str]:
    """
    Merge user-provided context with web-search evidence.

    Args:
        context: User-provided context
        web_search_context: Formatted web-search evidence

    Returns:
        Combined context string or None if neither exists
    """
    sections = []

    if context:
        sections.append("User-Provided Context:")
        sections.append(context)

    if web_search_context:
        sections.append(web_search_context)

    return "\n\n".join(sections) if sections else None


def analyze_market_prompt(
    question: str,
    context: Optional[str] = None,
    use_few_shot: bool = True,
    use_web_search: bool = False
) -> RiskScoreResult:
    """
    Analyze a market prompt for ambiguity risks.
    
    This is the main entry point for the risk scoring system. It takes a
    market question and returns a comprehensive risk assessment.
    
    Args:
        question: The market question to analyze
        context: Optional additional context supplied by the caller
        use_few_shot: Whether to use few-shot examples for better prompting
        use_web_search: Whether to augment the analysis with web-search evidence
        
    Returns:
        RiskScoreResult containing:
            - risk_score: Integer 0-100 (higher = more ambiguous)
            - risk_tags: List of identified risk categories
            - rationale: Detailed explanation of the assessment
            
    Example:
        >>> result = analyze_market_prompt(
        ...     "Will OpenAI release a new model in March this year?"
        ... )
        >>> print(result.risk_score)
        65
        >>> print(result.risk_tags)
        ['ambiguous_time', 'undefined_term']
    """
    merged_context = context

    if use_web_search:
        search_client = WebSearchClient()
        web_search_context = search_client.build_context(question)
        merged_context = merge_analysis_context(
            context=context,
            web_search_context=web_search_context
        )
    
    # Create scorer and analyze
    scorer = RiskScorer()
    return scorer.score(
        question=question,
        context=merged_context,
        include_few_shot=use_few_shot
    )


def analyze_proposal(proposal: MarketProposal) -> RiskScoreResult:
    """
    Analyze a MarketProposal object for ambiguity risks.
    
    Args:
        proposal: MarketProposal containing the question and optional context
        
    Returns:
        RiskScoreResult containing the risk assessment
    """
    scorer = RiskScorer()
    return scorer.score_proposal(proposal)


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description="Analyze market prompts for ambiguity risks"
    )
    parser.add_argument(
        "question",
        type=str,
        help="The market question to analyze"
    )
    parser.add_argument(
        "--context",
        type=str,
        default=None,
        help="Optional context for the analysis"
    )
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="Disable few-shot examples in prompting"
    )
    parser.add_argument(
        "--use-web-search",
        action="store_true",
        help="Augment the analysis with Tavily web-search evidence"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    result = analyze_market_prompt(
        question=args.question,
        context=args.context,
        use_few_shot=not args.no_few_shot,
        use_web_search=args.use_web_search
    )
    
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print(f"\n{'='*60}")
        print("MARKET PROMPT RISK ANALYSIS")
        print(f"{'='*60}")
        print(f"\nQuestion: {args.question}")
        print(f"\nRisk Score: {result.risk_score}/100")
        print(f"\nRisk Tags: {', '.join(result.risk_tags) if result.risk_tags else 'None'}")
        print(f"\nRationale:\n{result.rationale}")
        print(f"\n{'='*60}\n")
