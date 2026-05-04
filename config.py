"""
Configuration module for the Market Prompt Ambiguity Risk Scoring System.

This module contains configuration settings including API keys and model settings.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file if exists
load_dotenv()

# API Configuration
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_THINKING = os.getenv("DEEPSEEK_THINKING", "disabled")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
TAVILY_SEARCH_URL = os.getenv("TAVILY_SEARCH_URL", "https://api.tavily.com/search")

# Risk Scoring Configuration
MIN_RISK_SCORE = 0
MAX_RISK_SCORE = 100
DEFAULT_RISK_THRESHOLD = 50  # Scores above this are considered high risk
DEFAULT_WEB_SEARCH_MAX_RESULTS = int(os.getenv("TAVILY_SEARCH_MAX_RESULTS", "5"))
DEFAULT_WEB_SEARCH_DEPTH = os.getenv("TAVILY_SEARCH_DEPTH", "basic")
DEFAULT_WEB_SEARCH_TOPIC = os.getenv("TAVILY_SEARCH_TOPIC", "general")

# Predefined Risk Tags
RISK_TAGS = [
    "ambiguous_time",  # Time reference is unclear (e.g., "this year", "soon")
    "undefined_term",  # Key terms lack clear definition (e.g., "new model")
    "unverified_source",  # Lacks authoritative source
    "vague_condition",  # Resolution conditions are unclear
    "ambiguous_quantity",  # Quantities or degrees are unclear (e.g., "significant")
    "unidentified_subject",  # Subject identity is unclear (e.g., "some company")
    "high_disputability",  # Prone to disputes or subjective interpretation
]

# Few-shot Examples Path
FEW_SHOT_EXAMPLES_PATH = os.path.join(
    os.path.dirname(__file__), "few_shot_examples", "examples.json"
)

# RAG dataset path
PREDICTION_MARKET_EXAMPLES_PATH = os.path.join(
    os.path.dirname(__file__), "prediction_market_200_examples.json"
)
