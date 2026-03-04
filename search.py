"""
Web search integration for ambiguity risk scoring.

This module retrieves structured search evidence and formats it into
prompt-ready context for downstream ambiguity analysis.
"""

import json
from typing import Any, Dict, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from config import (
    DEFAULT_WEB_SEARCH_DEPTH,
    DEFAULT_WEB_SEARCH_MAX_RESULTS,
    DEFAULT_WEB_SEARCH_TOPIC,
    TAVILY_API_KEY,
    TAVILY_SEARCH_URL,
)
from models import SearchContext, SearchEvidenceItem


def build_search_query(question: str) -> str:
    """
    Build a search query focused on authoritative resolution evidence.

    Args:
        question: Market question being analyzed

    Returns:
        Search query string
    """
    return f"{question} official source definition resolution criteria"


def format_search_context(search_context: SearchContext) -> str:
    """
    Format structured search evidence into prompt-ready text.

    Args:
        search_context: Structured search context

    Returns:
        Formatted context string
    """
    sections = [
        "Web Search Evidence:",
        f"Provider: {search_context.provider}",
        f"Search Query: {search_context.query}",
    ]

    if search_context.summary:
        sections.extend(["Search Summary:", search_context.summary])

    if not search_context.evidence:
        sections.append("Evidence: No relevant search results were returned.")
    else:
        sections.append("Top Sources:")
        for index, item in enumerate(search_context.evidence, 1):
            sections.append(f"{index}. {item.title}")

            metadata = []
            if item.source:
                metadata.append(f"Source: {item.source}")
            if item.published_date:
                metadata.append(f"Published: {item.published_date}")
            if item.score is not None:
                metadata.append(f"Relevance: {item.score:.2f}")

            if metadata:
                sections.append("   " + " | ".join(metadata))

            sections.append(f"   URL: {item.url}")
            sections.append(f"   Snippet: {item.snippet}")

    sections.append(
        "Use this evidence to judge whether the market question has a unique subject, clear terminology, authoritative sources, and objective resolution criteria."
    )

    return "\n".join(sections)


class WebSearchClient:
    """
    Tavily-backed web search client for ambiguity analysis.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        search_url: Optional[str] = None,
        max_results: int = DEFAULT_WEB_SEARCH_MAX_RESULTS,
        search_depth: str = DEFAULT_WEB_SEARCH_DEPTH,
        topic: str = DEFAULT_WEB_SEARCH_TOPIC,
    ):
        self.api_key = api_key or TAVILY_API_KEY
        self.search_url = search_url or TAVILY_SEARCH_URL
        self.max_results = max_results
        self.search_depth = search_depth
        self.topic = topic

        if not self.api_key:
            raise ValueError("TAVILY_API_KEY is required when use_web_search=True")

    def search(self, question: str) -> SearchContext:
        """
        Search the web for evidence relevant to resolving a market question.

        Args:
            question: Market question to search around

        Returns:
            Structured search context
        """
        query = build_search_query(question)
        payload = {
            "api_key": self.api_key,
            "query": query,
            "topic": self.topic,
            "search_depth": self.search_depth,
            "max_results": self.max_results,
            "include_answer": True,
            "include_raw_content": False,
        }

        request = Request(
            self.search_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=15) as response:
                response_data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Tavily search failed with HTTP {exc.code}: {error_body}") from exc
        except URLError as exc:
            raise RuntimeError(f"Tavily search request failed: {exc.reason}") from exc

        return self._parse_response(query=query, response_data=response_data)

    def build_context(self, question: str) -> str:
        """
        Search and format prompt-ready context for a market question.

        Args:
            question: Market question to search around

        Returns:
            Formatted web-search context string
        """
        return format_search_context(self.search(question))

    def _parse_response(self, query: str, response_data: Dict[str, Any]) -> SearchContext:
        """
        Parse Tavily search response into structured models.

        Args:
            query: Search query used
            response_data: Raw Tavily response body

        Returns:
            Structured search context
        """
        evidence = []

        for result in response_data.get("results", []):
            title = self._clean_text(result.get("title")) or result.get("url") or "Untitled result"
            url = result.get("url", "")
            snippet = self._truncate_text(
                self._clean_text(result.get("content") or result.get("snippet")) or "No snippet provided.",
                limit=400,
            )
            evidence.append(
                SearchEvidenceItem(
                    title=title,
                    url=url,
                    snippet=snippet,
                    source=self._extract_source(url),
                    score=self._safe_float(result.get("score")),
                    published_date=result.get("published_date"),
                )
            )

        summary = self._clean_text(response_data.get("answer")) or self._build_summary(evidence)

        return SearchContext(
            query=query,
            provider="tavily",
            summary=summary,
            evidence=evidence,
        )

    @staticmethod
    def _clean_text(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return " ".join(value.split())

    @staticmethod
    def _truncate_text(text: str, limit: int = 400) -> str:
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_source(url: str) -> Optional[str]:
        if not url:
            return None
        return urlparse(url).netloc or None

    @staticmethod
    def _build_summary(evidence: list[SearchEvidenceItem]) -> Optional[str]:
        if not evidence:
            return None

        top_titles = [item.title for item in evidence[:3] if item.title]
        top_sources = [item.source for item in evidence[:3] if item.source]

        summary_parts = []
        if top_titles:
            summary_parts.append("Top evidence focuses on: " + "; ".join(top_titles))
        if top_sources:
            summary_parts.append("Primary sources include: " + ", ".join(top_sources))

        return " ".join(summary_parts) if summary_parts else None
