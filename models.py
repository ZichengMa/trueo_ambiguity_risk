"""
Data models for the Market Prompt Ambiguity Risk Scoring System.

This module defines the Pydantic models used for structured data throughout the system.
"""

from pydantic import BaseModel, Field
from typing import List, Optional


class RiskScoreResult(BaseModel):
    """
    Result of risk scoring for a market prompt.
    
    Attributes:
        risk_score: Risk score from 0-100 (higher = more ambiguous/risky)
        risk_tags: List of identified risk categories
        rationale: Detailed explanation of why these risks were identified
        confidence: Optional confidence level of the assessment (0-1)
    """
    risk_score: int = Field(
        ..., 
        ge=0, 
        le=100, 
        description="Risk score from 0 (no risk) to 100 (extremely high risk)"
    )
    risk_tags: List[str] = Field(
        ..., 
        description="List of identified ambiguity/risk categories"
    )
    rationale: str = Field(
        ..., 
        description="Detailed explanation of the identified risks"
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence level of the assessment"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "risk_score": 65,
                "risk_tags": ["ambiguous_time", "undefined_term"],
                "rationale": "1) 'this year' does not specify which year; 2) 'new model' is not clearly defined",
                "confidence": 0.85
            }
        }


class AnalysisResult(BaseModel):
    """
    Result of semantic analysis from the LLM Agent.
    
    Attributes:
        ambiguities: List of identified ambiguity points
        semantic_patterns: Detected semantic patterns
        suggested_clarifications: Suggestions for making the prompt clearer
    """
    ambiguities: List[str] = Field(
        default_factory=list,
        description="List of identified ambiguity points"
    )
    semantic_patterns: List[str] = Field(
        default_factory=list,
        description="Detected semantic patterns in the text"
    )
    suggested_clarifications: List[str] = Field(
        default_factory=list,
        description="Suggestions for clarifying the prompt"
    )


class SearchEvidenceItem(BaseModel):
    """
    Evidence item returned from a web search provider.

    Attributes:
        title: Result title
        url: Result URL
        snippet: Short snippet or extracted content
        source: Domain or source label
        score: Optional provider relevance score
        published_date: Optional publication date string
    """
    title: str = Field(..., description="Search result title")
    url: str = Field(..., description="Search result URL")
    snippet: str = Field(..., description="Search result snippet or extracted content")
    source: Optional[str] = Field(None, description="Source or domain name")
    score: Optional[float] = Field(None, description="Provider relevance score")
    published_date: Optional[str] = Field(None, description="Publication date if available")


class SearchContext(BaseModel):
    """
    Structured search context injected into the prompt.

    Attributes:
        query: Search query sent to the provider
        provider: Search provider name
        summary: Optional search summary
        evidence: Search evidence items
    """
    query: str = Field(..., description="Query sent to the search provider")
    provider: str = Field(..., description="Search provider name")
    summary: Optional[str] = Field(None, description="Optional high-level summary")
    evidence: List[SearchEvidenceItem] = Field(
        default_factory=list,
        description="Structured evidence returned from search"
    )


class MarketProposal(BaseModel):
    """
    Input model for a market proposal.
    
    Attributes:
        question: The market question to be analyzed
        context: Optional additional context (for future web search integration)
    """
    question: str = Field(
        ..., 
        description="The market question to analyze for ambiguity risks"
    )
    context: Optional[str] = Field(
        None,
        description="Additional context from web search or other sources"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "question": "Will OpenAI release a new model in March this year?",
                "context": None
            }
        }
