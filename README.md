# Market Prompt Ambiguity Risk Scoring System

A system for analyzing prediction market prompts and detecting ambiguity risks using GLM-4.7.

## Overview

This system analyzes market questions for potential ambiguity, vagueness, or clarity issues that could lead to disputes. It uses GLM-4.7 (via Zhipu AI) to perform semantic analysis and generate risk assessments.

## Features

- 🎯 **Risk Scoring**: Scores from 0-100 (higher = more ambiguous)
- 🏷️ **Risk Tags**: Identifies specific ambiguity categories
- 📝 **Detailed Rationale**: Provides explanations for the assessment
- 🔧 **File-Based Few-Shot Examples**: Default examples are loaded from `few_shot_examples/examples.json`
- ♻️ **Hot Reload Friendly**: Updating `examples.json` affects the next analysis call without code changes
- 🛡️ **Safe Fallback**: Falls back to built-in examples if the JSON file is missing or invalid
- 🌐 **Optional Web Search Evidence**: Can enrich scoring with Tavily search evidence when resolution criteria depend on real-world sources
- ✍️ **Automatic Rewrite Suggestions**: Generates clearer, more resolvable versions of ambiguous market questions
- 🔌 **Extensible**: Designed for web search integration and custom prompt iteration

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd trueo_ambiguity_risk

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your Zhipu AI API key
```

## Configuration

1. Get your API key from [Zhipu AI](https://open.bigmodel.cn/)
2. Copy `.env.example` to `.env`
3. Add your API key to `.env`:
   ```
   ZHIPU_API_KEY=your_api_key_here
   ```
4. Optional: add a Tavily API key to enable `use_web_search=True`:
   ```
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

### Few-shot Examples

By default, few-shot examples are loaded from `few_shot_examples/examples.json`.

- Edit that file to update the examples used in prompting
- Changes are picked up on the next call to `analyze_market_prompt(...)` or `python main.py ...`
- If the file is missing, malformed, or has an invalid structure, the system falls back to built-in examples in `prompts.py`

Example file format:

```json
[
  {
    "question": "Will Apple release a new product this year?",
    "result": {
      "risk_score": 70,
      "risk_tags": ["ambiguous_time", "undefined_term", "vague_condition"],
      "rationale": "Explain why this question is ambiguous."
    }
  }
]
```

### Web Search Configuration

Web search is optional and is only used when `use_web_search=True` or `--use-web-search` is enabled.

- Provider: Tavily
- Required env var: `TAVILY_API_KEY`
- Optional env vars:
  - `TAVILY_SEARCH_MAX_RESULTS` defaults to `5`
  - `TAVILY_SEARCH_DEPTH` defaults to `basic`
  - `TAVILY_SEARCH_TOPIC` defaults to `general`

The search layer is designed to help with ambiguity scoring, not event prediction. It looks for evidence about:

- authoritative sources
- naming ambiguity
- competing interpretations
- resolution criteria
- dispute risk

## Usage

### Web App

The repo includes a Streamlit frontend for analysts who want a browser-based workflow.

Start it with:

```bash
streamlit run app.py
```

The UI supports:

- question input and optional analyst context
- few-shot toggle
- web-search toggle
- search-debug toggle
- automatic rewrite suggestions
- evidence cards for front-end-style display
- raw/simplified search inspection for internal review

### Python API

```python
from main import analyze_market_prompt

# Analyze a market question
result = analyze_market_prompt("Will OpenAI release a new model in March this year?")

print(f"Risk Score: {result.risk_score}/100")
print(f"Risk Tags: {result.risk_tags}")
print(f"Rationale: {result.rationale}")
```

To disable few-shot examples:

```python
result = analyze_market_prompt(
    "Will OpenAI release a new model in March this year?",
    use_few_shot=False
)
```

To augment the analysis with web-search evidence:

```python
result = analyze_market_prompt(
    "Will OpenAI release a new model in March this year?",
    use_web_search=True
)
```

To return the web-search evidence chain for product/debug display:

```python
result = analyze_market_prompt(
    "Will OpenAI release a new model in March this year?",
    use_web_search=True,
    include_search_debug=True
)
```

### How Web Search Is Used

When web search is enabled, the system does four things:

1. Builds a search query from the market question, biased toward authoritative and resolution-related evidence.
2. Calls Tavily and collects the top results plus Tavily's short answer when available.
3. Re-ranks the results to prefer likely official or authoritative sources over media and community discussion when possible.
4. Simplifies the re-ranked results into a compact `SearchContext` with:
   - `query`
   - `provider`
   - `summary`
   - `evidence[]` containing `title`, `url`, `snippet`, `source`, `score`, and `published_date`
5. Formats that simplified context into prompt text and injects it into the ambiguity analysis.

This means the LLM does not receive the full raw API payload. It receives a reduced, prompt-friendly summary of the search evidence.

If `include_search_debug=True`, the final `RiskScoreResult` also includes a `search_debug` field with:

- `provider`
- `initial_query`
- `follow_up_queries`
- `raw_answer`
- `raw_results` from all search queries before reranking
- `display_evidence` for front-end-friendly evidence cards
- `simplified_context`
- `formatted_context`

This is intended for debugging and product-side evidence-chain display.

`display_evidence[]` is designed for direct UI rendering and includes:

- `rank`
- `title`
- `url`
- `source`
- `snippet`
- `published_date`
- `relevance_score`
- `source_category`
- `is_official`
- `display_reason`

High-level flow:

```text
Market question
  -> Tavily search
  -> raw search results
  -> simplified SearchContext
  -> formatted prompt context
  -> ambiguity scoring
```

The implementation lives in `search.py`.

### Command Line

```bash
python main.py "Will OpenAI release a new model in March this year?"
```

To bypass few-shot examples:

```bash
python main.py "Will OpenAI release a new model in March this year?" --no-few-shot
```

To augment the analysis with web-search evidence:

```bash
python main.py "Will OpenAI release a new model in March this year?" --use-web-search
```

To include the web-search evidence chain in JSON output:

```bash
python main.py "Will OpenAI release a new model in March this year?" --use-web-search --include-search-debug --json
```

## Output Format

```json
{
  "risk_score": 65,
  "risk_tags": ["ambiguous_time", "undefined_term"],
  "rationale": "1) 'this year' does not specify which year; 2) 'new model' is not clearly defined"
}
```

When web search is enabled, the returned JSON still has the same schema. The search evidence is used internally to improve the score and rationale; it is not returned as a separate field in the final API result.

If `include_search_debug=True`, the JSON response additionally includes a `search_debug` object. By default this field remains `null` unless you explicitly request web-search debug output.

## Risk Categories

| Tag | Description |
|-----|-------------|
| `ambiguous_time` | Time reference is unclear |
| `undefined_term` | Key terms lack clear definition |
| `unverified_source` | Lacks authoritative source |
| `vague_condition` | Resolution conditions are unclear |
| `ambiguous_quantity` | Quantities or degrees are unclear |
| `unidentified_subject` | Subject identity is unclear |
| `high_disputability` | Prone to disputes or subjective interpretation |

## Project Structure

```
trueo_ambiguity_risk/
├── app.py               # Streamlit frontend
├── PLAN.md              # Design documentation
├── TODO.md              # Paused and upcoming work items
├── README.md            # This file
├── requirements.txt     # Python dependencies
├── config.py            # Configuration settings
├── models.py            # Data models (Pydantic)
├── prompts.py           # Prompt templates and context injection logic
├── agent.py             # LLM Agent (GLM-4.7)
├── scorer.py            # Risk Scorer
├── main.py              # Main entry point
├── search.py            # Tavily search client and evidence formatter
├── rewriter.py          # Automatic rewrite suggestion generator
├── few_shot_examples/   # Default few-shot examples loaded at runtime
└── tests/               # Test cases
    ├── test_scorer.py
    └── test_rewriter.py
```

## Testing

Run tests:

```bash
python tests/test_scorer.py
```

Quick test (single API call):

```bash
python tests/test_scorer.py --quick
```

Note: the examples file and web-search integration tests are local-only, but the main scoring tests still call the live Zhipu API.

Web-search-specific local tests currently cover:

- search context formatting
- missing Tavily API key handling
- main pipeline integration with mocked search evidence

## Future Enhancements

- [ ] Batch processing API
- [ ] External Retrieval Agent (MCP/Tools)

## Partner

Trueo - Prediction Market Platform

## License

MIT License
