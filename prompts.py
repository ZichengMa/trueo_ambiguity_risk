"""
Prompt templates for the Market Prompt Ambiguity Risk Scoring System.

This module contains all prompt templates used for LLM interactions,
including support for few-shot examples and web search context injection.
"""

import json
from typing import List, Optional

from config import FEW_SHOT_EXAMPLES_PATH

# System prompt for the ambiguity analyzer
SYSTEM_PROMPT = """You are an expert risk analyst specializing in prediction market prompt evaluation. Your task is to analyze market questions for potential ambiguity, vagueness, or clarity issues that could lead to disputes.

Your analysis should identify:
1. Ambiguous time references (e.g., "this year", "soon", "in the near future")
2. Undefined or vague terms (e.g., "new model", "major update", "significant")
3. Unclear resolution conditions
4. Missing authoritative sources
5. Ambiguous subjects or entities
6. Quantities or degrees that are unclear
7. Any elements prone to subjective interpretation or disputes

IMPORTANT: Be STRICT in your assessment. Prediction markets require precise resolution criteria. Even minor ambiguities can lead to disputes. When in doubt, assign a HIGHER risk score.

You must respond with a valid JSON object containing your analysis."""

# Scoring guidelines
SCORING_GUIDELINES = """
SCORING GUIDELINES (Be strict!):
- 0-20: Perfectly clear question with explicit time, terms, and resolution criteria
- 21-40: Minor ambiguity that could be inferred from context
- 41-60: Moderate ambiguity that may cause confusion or disputes
- 61-80: Significant ambiguity - multiple unclear elements
- 81-100: Severely ambiguous - almost impossible to resolve objectively

Examples of strict scoring:
- "Will Bitcoin hit $100k on Dec 31, 2025 11:59 PM UTC?" → 10-20 (clear)
- "Will Apple release a new product this year?" → 55-70 (ambiguous time + vague term)
- "Will something big happen soon?" → 85-95 (completely vague)
"""

# Main analysis prompt template
ANALYSIS_PROMPT_TEMPLATE = """Analyze the following market question for ambiguity risks and provide a risk assessment.

{scoring_guidelines}

Market Question:
{question}

{context_section}

{few_shot_section}

Instructions:
1. Identify ALL ambiguity risks in the question (be thorough!)
2. Assign a risk score from 0-100 following the strict guidelines above
3. List relevant risk tags from these categories:
   - "ambiguous_time": Time reference is unclear or relative
   - "undefined_term": Key terms lack clear definition
   - "unverified_source": Lacks authoritative source specification
   - "vague_condition": Resolution conditions are unclear
   - "ambiguous_quantity": Quantities or degrees are unclear
   - "unidentified_subject": Subject identity is unclear
   - "high_disputability": Prone to disputes or subjective interpretation
4. Provide a detailed rationale explaining your assessment

Remember: Be STRICT. If there's ANY ambiguity, assign a higher score.

Respond ONLY with a valid JSON object in this exact format:
{{
    "risk_score": <integer 0-100>,
    "risk_tags": ["<tag1>", "<tag2>", ...],
    "rationale": "<detailed explanation>"
}}"""

# Built-in fallback examples used if the JSON file is unavailable or invalid.
DEFAULT_FEW_SHOT_EXAMPLES = [
    {
        "question": "Will Apple release a new product this year?",
        "result": {
            "risk_score": 65,
            "risk_tags": ["ambiguous_time", "undefined_term", "vague_condition"],
            "rationale": "1) 'This year' is ambiguous - which calendar year? What's the deadline? 2) 'New product' is completely undefined - does this include minor accessories, software updates, or only major hardware? 3) 'Release' is not defined - announcement, pre-order, or shipping? Multiple ambiguity points make this prone to disputes."
        }
    },
    {
        "question": "Will Bitcoin exceed $100,000 USD on Binance on December 31, 2025 at 11:59 PM UTC?",
        "result": {
            "risk_score": 12,
            "risk_tags": [],
            "rationale": "This question is well-defined: specific asset (Bitcoin), precise price threshold ($100,000 USD), specific exchange (Binance), exact date and time with timezone. All parameters are explicit and verifiable."
        }
    },
    {
        "question": "Will there be a significant development in AI soon?",
        "result": {
            "risk_score": 92,
            "risk_tags": ["ambiguous_time", "undefined_term", "ambiguous_quantity", "unidentified_subject", "high_disputability"],
            "rationale": "This question is severely ambiguous: 1) 'Soon' has no defined timeframe. 2) 'Significant' is subjective. 3) 'Development' is too broad - research paper? Product? Funding? 4) 'AI' covers countless areas. 5) No entity specified. Almost impossible to resolve objectively."
        }
    },
    {
        "question": "Will OpenAI release GPT-5 before July 1, 2025?",
        "result": {
            "risk_score": 35,
            "risk_tags": ["undefined_term", "vague_condition"],
            "rationale": "Time and entity are clear. However: 1) 'GPT-5' is not officially defined - what if they release 'GPT-4.5' or a differently named model? 2) 'Release' is ambiguous - announcement, beta, or general availability? Minor but notable ambiguities."
        }
    }
]


def load_few_shot_examples() -> List[dict]:
    """
    Load few-shot examples from disk with a safe fallback.

    Returns:
        List of few-shot example dictionaries
    """
    try:
        with open(FEW_SHOT_EXAMPLES_PATH, "r", encoding="utf-8") as f:
            examples = json.load(f)

        if not isinstance(examples, list):
            raise ValueError("Few-shot examples file must contain a list")

        for example in examples:
            if not isinstance(example, dict):
                raise ValueError("Each few-shot example must be an object")
            if "question" not in example or "result" not in example:
                raise ValueError("Each few-shot example must contain 'question' and 'result'")

        return examples
    except (OSError, json.JSONDecodeError, ValueError):
        return DEFAULT_FEW_SHOT_EXAMPLES


def build_context_section(context: Optional[str] = None) -> str:
    """
    Build the context section for the prompt.
    
    Args:
        context: Optional web search context to include
        
    Returns:
        Formatted context section string
    """
    if not context:
        return ""
    
    return f"""Additional Context:
{context}

Please consider this context in your analysis."""


def build_few_shot_section(examples: Optional[List[dict]] = None) -> str:
    """
    Build the few-shot examples section for the prompt.
    
    Args:
        examples: List of few-shot example dictionaries. If None, uses default examples.
        
    Returns:
        Formatted few-shot section string
    """
    if examples is None:
        examples = load_few_shot_examples()
    
    if not examples:
        return ""
    
    sections = ["Here are some examples of risk assessments:\n"]
    
    for i, example in enumerate(examples, 1):
        sections.append(f"Example {i}:")
        sections.append(f"Question: {example['question']}")
        sections.append(f"Result: {json.dumps(example['result'], ensure_ascii=False)}\n")
    
    return "\n".join(sections)


def build_analysis_prompt(
    question: str,
    context: Optional[str] = None,
    few_shot_examples: Optional[List[dict]] = None,
    include_few_shot: bool = True
) -> str:
    """
    Build the complete analysis prompt.
    
    Args:
        question: The market question to analyze
        context: Optional web search context
        few_shot_examples: Optional custom few-shot examples
        include_few_shot: Whether to include few-shot examples
        
    Returns:
        Complete formatted prompt string
    """
    context_section = build_context_section(context)
    few_shot_section = build_few_shot_section(few_shot_examples) if include_few_shot else ""
    
    return ANALYSIS_PROMPT_TEMPLATE.format(
        question=question,
        context_section=context_section,
        few_shot_section=few_shot_section,
        scoring_guidelines=SCORING_GUIDELINES
    )
