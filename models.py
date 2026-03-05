"""
Data models for the Market Prompt Ambiguity Risk Scoring System.

This module defines the Pydantic models used for structured data throughout the system.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


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
    search_debug: Optional["SearchDebugInfo"] = Field(
        None,
        description="Optional web-search debug information for evidence-chain inspection"
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


class SearchDebugInfo(BaseModel):
    """
    Optional debug information for web-search evidence chains.

    Attributes:
        provider: Search provider name
        initial_query: Primary search query
        follow_up_queries: Additional site-biased queries
        raw_answer: Provider summary from the primary search response
        raw_results: Normalized search results before reranking
        simplified_context: Reranked search context used for prompting
        formatted_context: Prompt-ready search context string
    """
    provider: str = Field(..., description="Search provider name")
    initial_query: str = Field(..., description="Primary query sent to the search provider")
    follow_up_queries: List[str] = Field(
        default_factory=list,
        description="Additional follow-up queries sent to the search provider"
    )
    raw_answer: Optional[str] = Field(
        None,
        description="Provider-generated answer or summary from the primary search request"
    )
    raw_results: List[SearchEvidenceItem] = Field(
        default_factory=list,
        description="Normalized search results before reranking"
    )
    display_evidence: List["SearchDisplayEvidenceItem"] = Field(
        default_factory=list,
        description="Front-end-friendly evidence cards derived from reranked search results"
    )
    simplified_context: SearchContext = Field(
        ...,
        description="Reranked search context used for prompt injection"
    )
    formatted_context: str = Field(
        ...,
        description="Prompt-ready text derived from the simplified context"
    )


class SearchDisplayEvidenceItem(BaseModel):
    """
    Front-end-friendly evidence item derived from reranked search results.

    Attributes:
        rank: Display order after reranking
        title: Display title
        url: Evidence URL
        source: Source/domain label
        snippet: Short preview text
        published_date: Optional publication date
        relevance_score: Optional provider relevance score
        source_category: High-level source type for UI grouping
        is_official: Whether the source is likely official/authoritative
        display_reason: Short explanation for why this result is useful
    """
    rank: int = Field(..., ge=1, description="1-based display rank")
    title: str = Field(..., description="Evidence title")
    url: str = Field(..., description="Evidence URL")
    source: Optional[str] = Field(None, description="Source or domain name")
    snippet: str = Field(..., description="Short display preview")
    published_date: Optional[str] = Field(None, description="Publication date if available")
    relevance_score: Optional[float] = Field(None, description="Provider relevance score")
    source_category: str = Field(..., description="High-level source category for UI display")
    is_official: bool = Field(..., description="Whether the source is likely official")
    display_reason: str = Field(..., description="Short explanation for why this evidence is useful")


class RewriteSuggestionItem(BaseModel):
    """
    Single rewrite suggestion for a more resolvable market question.

    Attributes:
        rewritten_question: Suggested market question rewrite
        why_clearer: Why this rewrite is easier to resolve objectively
    """
    rewritten_question: str = Field(..., description="Suggested rewritten market question")
    why_clearer: str = Field(..., description="Brief explanation of why the rewrite is clearer")


class RewriteSuggestions(BaseModel):
    """
    Collection of rewrite suggestions and optional guidance.

    Attributes:
        suggestions: Suggested rewritten questions
        general_guidance: Optional high-level guidance
    """
    suggestions: List[RewriteSuggestionItem] = Field(
        default_factory=list,
        description="List of suggested rewritten questions"
    )
    general_guidance: Optional[str] = Field(
        None,
        description="Optional high-level guidance for writing resolvable questions"
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


RiskScoreResult.model_rebuild()
