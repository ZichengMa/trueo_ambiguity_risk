"""
LLM Agent module for the Market Prompt Ambiguity Risk Scoring System.

This module contains the SemanticAnalysisAgent class that interfaces with
the GLM-4.7 API to perform semantic analysis and ambiguity detection.
"""

import json
import re
from typing import Optional, List
from zhipuai import ZhipuAI

from config import ZHIPU_API_KEY, ZHIPU_MODEL
from models import RiskScoreResult
from prompts import build_analysis_prompt, SYSTEM_PROMPT
from rag import retrieve_few_shot_examples


class SemanticAnalysisAgent:
    """
    Agent for analyzing market prompts using GLM-4.7.
    
    This agent interfaces with the Zhipu AI API to perform semantic analysis
    and detect ambiguity risks in market questions.
    
    Attributes:
        client: ZhipuAI client instance
        model: Model identifier to use
        few_shot_examples: Optional list of few-shot examples for prompting
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        few_shot_examples: Optional[List[dict]] = None
    ):
        """
        Initialize the SemanticAnalysisAgent.
        
        Args:
            api_key: Zhipu AI API key (defaults to config value)
            model: Model identifier (defaults to config value)
            few_shot_examples: Optional few-shot examples for better prompting
        """
        self.api_key = api_key or ZHIPU_API_KEY
        self.model = model or ZHIPU_MODEL
        self.client = ZhipuAI(api_key=self.api_key)
        self.few_shot_examples = few_shot_examples
    
    def analyze(
        self, 
        question: str,
        context: Optional[str] = None,
        include_few_shot: bool = True
    ) -> RiskScoreResult:
        """
        Analyze a market question for ambiguity risks.
        
        Args:
            question: The market question to analyze
            context: Optional additional context (e.g., from web search)
            include_few_shot: Whether to include few-shot examples in prompt
            
        Returns:
            RiskScoreResult containing risk score, tags, and rationale
            
        Raises:
            ValueError: If the API response cannot be parsed
        """
        # Resolve few-shot examples: prefer explicit ones, then RAG retrieval
        few_shot_examples = self.few_shot_examples
        if include_few_shot and few_shot_examples is None:
            few_shot_examples = retrieve_few_shot_examples(question)

        # Build the prompt
        prompt = build_analysis_prompt(
            question=question,
            context=context,
            few_shot_examples=few_shot_examples,
            include_few_shot=include_few_shot
        )
        
        # Call the API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent outputs
        )
        
        # Extract the response content
        content = response.choices[0].message.content
        
        # Parse the JSON response
        result = self._parse_response(content)
        
        return result
    
    def _parse_response(self, content: str) -> RiskScoreResult:
        """
        Parse the LLM response into a RiskScoreResult.
        
        Args:
            content: Raw response content from the LLM
            
        Returns:
            Parsed RiskScoreResult object
            
        Raises:
            ValueError: If parsing fails
        """
        # Try to extract JSON from the response
        # Sometimes the LLM might include extra text
        json_match = re.search(r'\{[\s\S]*\}', content)
        
        if not json_match:
            raise ValueError(f"Could not find JSON in response: {content}")
        
        json_str = json_match.group(0)
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON: {e}\nContent: {json_str}")
        
        # Validate required fields
        if "risk_score" not in data:
            raise ValueError("Missing 'risk_score' in response")
        if "risk_tags" not in data:
            raise ValueError("Missing 'risk_tags' in response")
        if "rationale" not in data:
            raise ValueError("Missing 'rationale' in response")
        
        # Ensure risk_score is within bounds
        risk_score = max(0, min(100, int(data["risk_score"])))
        
        return RiskScoreResult(
            risk_score=risk_score,
            risk_tags=data["risk_tags"],
            rationale=data["rationale"],
            confidence=data.get("confidence")
        )
    
    def batch_analyze(
        self, 
        questions: List[str],
        **kwargs
    ) -> List[RiskScoreResult]:
        """
        Analyze multiple questions in batch.
        
        Args:
            questions: List of market questions to analyze
            **kwargs: Additional arguments passed to analyze()
            
        Returns:
            List of RiskScoreResult objects
        """
        results = []
        for question in questions:
            result = self.analyze(question, **kwargs)
            results.append(result)
        return results
