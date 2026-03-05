"""
Streamlit frontend for the ambiguity risk scoring system.
"""

from html import escape

import streamlit as st

from main import analyze_market_prompt
from rewriter import suggest_resolvable_rewrites


SAMPLE_QUESTIONS = [
    "Will OpenAI release a new model in March this year?",
    "Will Bitcoin exceed $100,000 USD on Binance on December 31, 2025 at 11:59 PM UTC?",
    "Will there be a significant development in AI soon?",
]


def inject_styles() -> None:
    """Inject custom CSS for a more intentional visual design."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg: #f6f0e8;
            --panel: rgba(255, 251, 245, 0.88);
            --ink: #1f1c19;
            --muted: #6e6359;
            --accent: #0d7c66;
            --accent-2: #f0a202;
            --danger: #b0413e;
            --border: rgba(31, 28, 25, 0.1);
            --shadow: 0 18px 60px rgba(68, 47, 32, 0.12);
        }

        html, body, [data-testid="stAppViewContainer"] {
            background:
                radial-gradient(circle at top left, rgba(13, 124, 102, 0.16), transparent 24%),
                radial-gradient(circle at top right, rgba(240, 162, 2, 0.18), transparent 26%),
                linear-gradient(180deg, #fbf5ee 0%, #f3ece4 100%);
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
        }

        [data-testid="stSidebar"] {
            background: rgba(255, 248, 240, 0.78);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            font-family: 'Space Grotesk', sans-serif;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }

        h1, h2, h3 {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -0.03em;
        }

        .hero {
            background: linear-gradient(135deg, rgba(255,255,255,0.84), rgba(255, 248, 240, 0.74));
            border: 1px solid var(--border);
            border-radius: 28px;
            padding: 1.6rem 1.4rem 1.2rem 1.4rem;
            box-shadow: var(--shadow);
            backdrop-filter: blur(12px);
            position: relative;
            overflow: hidden;
            margin-bottom: 1.2rem;
        }

        .hero:after {
            content: "";
            position: absolute;
            inset: auto -8% -55% auto;
            width: 240px;
            height: 240px;
            background: radial-gradient(circle, rgba(13,124,102,0.2), transparent 70%);
            transform: rotate(12deg);
        }

        .hero-kicker {
            display: inline-block;
            margin-bottom: 0.6rem;
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
        }

        .hero-copy {
            color: var(--muted);
            max-width: 860px;
            font-size: 1rem;
            line-height: 1.6;
        }

        .metric-card, .panel-card, .evidence-card, .rewrite-card, .tag-chip, .prompt-pill {
            border: 1px solid var(--border);
            box-shadow: var(--shadow);
        }

        .metric-card {
            background: var(--panel);
            border-radius: 22px;
            padding: 1rem 1rem 0.95rem 1rem;
            min-height: 124px;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
            margin-bottom: 0.35rem;
        }

        .metric-value {
            font-size: 2rem;
            line-height: 1;
            font-weight: 700;
        }

        .metric-help {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.5rem;
        }

        .panel-card {
            background: var(--panel);
            border-radius: 24px;
            padding: 1.1rem 1.1rem 1rem 1.1rem;
            margin-top: 1rem;
        }

        .panel-title {
            font-size: 1.05rem;
            font-weight: 700;
            margin-bottom: 0.7rem;
        }

        .panel-text {
            color: var(--ink);
            line-height: 1.75;
        }

        .tag-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .tag-chip {
            display: inline-flex;
            align-items: center;
            padding: 0.44rem 0.8rem;
            border-radius: 999px;
            background: rgba(13, 124, 102, 0.08);
            color: var(--accent);
            font-size: 0.86rem;
            font-weight: 700;
        }

        .prompt-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.42rem 0.78rem;
            margin: 0.2rem 0.32rem 0.2rem 0;
            border-radius: 999px;
            background: rgba(240, 162, 2, 0.08);
            color: #7d5400;
            font-size: 0.82rem;
            font-family: 'IBM Plex Mono', monospace;
        }

        .evidence-card {
            background: rgba(255,255,255,0.78);
            border-radius: 22px;
            padding: 1rem;
            margin-bottom: 0.85rem;
        }

        .evidence-rank {
            color: var(--muted);
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .evidence-title {
            font-size: 1rem;
            font-weight: 700;
            margin: 0.2rem 0 0.35rem 0;
        }

        .evidence-meta {
            color: var(--muted);
            font-size: 0.88rem;
            margin-bottom: 0.55rem;
        }

        .evidence-snippet {
            color: var(--ink);
            line-height: 1.6;
            font-size: 0.96rem;
        }

        .rewrite-card {
            background: rgba(255,255,255,0.82);
            border-radius: 20px;
            padding: 0.95rem;
            margin-bottom: 0.75rem;
        }

        .rewrite-head {
            color: var(--muted);
            font-size: 0.8rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.3rem;
        }

        .rewrite-question {
            color: var(--ink);
            font-weight: 700;
            line-height: 1.55;
            margin-bottom: 0.45rem;
        }

        .rewrite-why {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .mono-block {
            font-family: 'IBM Plex Mono', monospace;
            font-size: 0.86rem;
            white-space: pre-wrap;
            word-break: break-word;
            color: #2d2924;
            background: rgba(31, 28, 25, 0.04);
            border-radius: 18px;
            padding: 0.85rem;
            border: 1px solid var(--border);
        }

        .stButton > button, .stDownloadButton > button, .stFormSubmitButton > button {
            border-radius: 999px;
            border: 1px solid rgba(13, 124, 102, 0.35);
            background: linear-gradient(135deg, #0d7c66, #18a17f);
            color: white;
            font-weight: 700;
            padding: 0.65rem 1.1rem;
        }

        .stTextArea textarea, .stTextInput input {
            border-radius: 18px;
            border: 1px solid var(--border);
            background: rgba(255,255,255,0.78);
        }

        @media (max-width: 820px) {
            .block-container {
                padding-top: 1rem;
            }
            .hero {
                border-radius: 22px;
                padding: 1.1rem;
            }
            .metric-card, .panel-card, .evidence-card {
                border-radius: 18px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, help_text: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{escape(label)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-help">{escape(help_text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def panel_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="panel-card">
            <div class="panel-title">{escape(title)}</div>
            <div class="panel-text">{escape(body).replace('\n', '<br>')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tag_row(tags: list[str]) -> None:
    if not tags:
        st.markdown('<div class="panel-text">No risk tags returned.</div>', unsafe_allow_html=True)
        return

    chips = "".join(f'<span class="tag-chip">{escape(tag)}</span>' for tag in tags)
    st.markdown(f'<div class="tag-row">{chips}</div>', unsafe_allow_html=True)


def render_prompt_pills(items: list[str]) -> None:
    if not items:
        st.markdown('<div class="panel-text">No follow-up queries were issued.</div>', unsafe_allow_html=True)
        return

    pills = "".join(f'<span class="prompt-pill">{escape(item)}</span>' for item in items)
    st.markdown(pills, unsafe_allow_html=True)


def render_evidence_cards(result) -> None:
    if not result:
        st.info("Enable search debug to view evidence cards.")
        return

    for item in result:
        category = item.source_category.replace("-", " ").title()
        official_flag = "Official" if item.is_official else "Context"
        meta_parts = [
            f"{item.source or 'unknown source'}",
            category,
            official_flag,
        ]
        if item.relevance_score is not None:
            meta_parts.append(f"score {item.relevance_score:.2f}")
        if item.published_date:
            meta_parts.append(item.published_date)

        st.markdown(
            f"""
            <div class="evidence-card">
                <div class="evidence-rank">Evidence {item.rank}</div>
                <div class="evidence-title"><a href="{escape(item.url)}" target="_blank">{escape(item.title)}</a></div>
                <div class="evidence-meta">{escape(' · '.join(meta_parts))}</div>
                <div class="evidence-snippet">{escape(item.snippet)}</div>
                <div class="evidence-meta" style="margin-top:0.65rem;">{escape(item.display_reason)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_search_debug(search_debug) -> None:
    if search_debug is None:
        st.info("Enable both web search and search debug to inspect the evidence chain.")
        return

    tab_cards, tab_queries, tab_raw = st.tabs(["Display Evidence", "Queries", "Raw / Prompt"])

    with tab_cards:
        render_evidence_cards(search_debug.display_evidence)

    with tab_queries:
        panel_card("Initial Query", search_debug.initial_query)
        st.markdown('<div class="panel-card"><div class="panel-title">Follow-up Queries</div></div>', unsafe_allow_html=True)
        render_prompt_pills(search_debug.follow_up_queries)

    with tab_raw:
        col1, col2 = st.columns(2)
        with col1:
            panel_card("Raw Provider Answer", search_debug.raw_answer or "No provider answer returned.")
            st.markdown(
                f'<div class="panel-card"><div class="panel-title">Raw Result Count</div><div class="panel-text">{len(search_debug.raw_results)}</div></div>',
                unsafe_allow_html=True,
            )
        with col2:
            panel_card("Simplified Summary", search_debug.simplified_context.summary or "No simplified summary available.")
            st.markdown(
                f'<div class="panel-card"><div class="panel-title">Prompt Evidence Count</div><div class="panel-text">{len(search_debug.simplified_context.evidence)}</div></div>',
                unsafe_allow_html=True,
            )

        st.markdown('<div class="panel-card"><div class="panel-title">Prompt-Ready Context</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mono-block">{escape(search_debug.formatted_context)}</div>', unsafe_allow_html=True)

        with st.expander("Raw Results JSON", expanded=False):
            st.json([item.model_dump() for item in search_debug.raw_results])

        with st.expander("Simplified Context JSON", expanded=False):
            st.json(search_debug.simplified_context.model_dump())


def render_rewrite_suggestions(suggestions) -> None:
    if suggestions is None or not suggestions.suggestions:
        st.info("No rewrite suggestions were generated.")
        return

    for idx, suggestion in enumerate(suggestions.suggestions, 1):
        st.markdown(
            f"""
            <div class="rewrite-card">
                <div class="rewrite-head">Suggestion {idx}</div>
                <div class="rewrite-question">{escape(suggestion.rewritten_question)}</div>
                <div class="rewrite-why">{escape(suggestion.why_clearer)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.code(suggestion.rewritten_question, language="text")

    if suggestions.general_guidance:
        panel_card("Rewrite Guidance", suggestions.general_guidance)


def main() -> None:
    st.set_page_config(
        page_title="Ambiguity Risk Studio",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()

    if "question" not in st.session_state:
        st.session_state.question = SAMPLE_QUESTIONS[0]

    with st.sidebar:
        st.markdown("## Controls")
        selected_sample = st.selectbox("Sample question", ["Custom"] + SAMPLE_QUESTIONS)
        if selected_sample != "Custom":
            st.session_state.question = selected_sample

        use_few_shot = st.toggle("Use few-shot examples", value=True)
        use_web_search = st.toggle("Use web search", value=True)
        include_search_debug = st.toggle("Show search debug", value=True, disabled=not use_web_search)
        generate_rewrites = st.toggle("Generate rewrite suggestions", value=True)
        rewrite_count = st.slider("Rewrite suggestions", min_value=1, max_value=3, value=2)

        st.markdown("---")
        st.markdown("### Output Focus")
        st.caption("Use search debug when you want the full evidence chain for UI display or internal review.")

    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">Prediction Market Review</div>
            <h1>Ambiguity Risk Studio</h1>
            <div class="hero-copy">
                Stress-test a market question before it goes live. This UI surfaces the score,
                the dispute rationale, and the evidence chain behind web-search-assisted judgments.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("analysis-form", clear_on_submit=False):
        question = st.text_area(
            "Market question",
            value=st.session_state.question,
            height=150,
            placeholder="Will OpenAI release a new model in March this year?",
        )
        context = st.text_area(
            "Optional analyst context",
            height=110,
            placeholder="Paste extra context, background notes, or internal resolution guidance.",
        )
        submitted = st.form_submit_button("Run Analysis")

    if submitted:
        st.session_state.question = question
        if not question.strip():
            st.error("Enter a market question before running analysis.")
            return

        with st.spinner("Running ambiguity analysis..."):
            try:
                result = analyze_market_prompt(
                    question=question.strip(),
                    context=context.strip() or None,
                    use_few_shot=use_few_shot,
                    use_web_search=use_web_search,
                    include_search_debug=include_search_debug,
                )
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")
                return

        rewrite_suggestions = None
        if generate_rewrites:
            with st.spinner("Generating clearer, more resolvable rewrites..."):
                try:
                    search_summary = (
                        result.search_debug.simplified_context.summary
                        if result.search_debug is not None
                        else None
                    )
                    rewrite_suggestions = suggest_resolvable_rewrites(
                        question=question.strip(),
                        risk_tags=result.risk_tags,
                        rationale=result.rationale,
                        search_summary=search_summary,
                        max_suggestions=rewrite_count,
                    )
                except Exception as exc:
                    st.warning(f"Rewrite suggestion generation failed: {exc}")

        risk_score = result.risk_score
        level = (
            "Low clarity risk"
            if risk_score < 35
            else "Moderate dispute risk"
            if risk_score < 65
            else "High ambiguity risk"
        )
        risk_help = "Higher scores mean more room for resolution disputes."

        col1, col2, col3 = st.columns(3)
        with col1:
            metric_card("Risk Score", f"{risk_score}/100", risk_help)
        with col2:
            metric_card("Risk Level", level, f"{len(result.risk_tags)} tags detected")
        with col3:
            debug_count = (
                len(result.search_debug.display_evidence)
                if result.search_debug is not None
                else 0
            )
            metric_card("Evidence Cards", str(debug_count), "Available when search debug is enabled")

        st.progress(min(max(risk_score, 0), 100) / 100)

        left, right = st.columns([1.05, 0.95], gap="large")
        with left:
            panel_card("Rationale", result.rationale)
            st.markdown('<div class="panel-card"><div class="panel-title">Risk Tags</div></div>', unsafe_allow_html=True)
            render_tag_row(result.risk_tags)
            st.markdown('<div class="panel-card"><div class="panel-title">Question Rewrite Suggestions</div></div>', unsafe_allow_html=True)
            if generate_rewrites:
                render_rewrite_suggestions(rewrite_suggestions)
            else:
                st.info("Enable 'Generate rewrite suggestions' from the sidebar to view improved market wording.")

        with right:
            search_summary = (
                result.search_debug.simplified_context.summary
                if result.search_debug is not None
                else "Search debug disabled. Enable web search and debug to inspect evidence."
            )
            panel_card("Search Summary", search_summary)
            st.markdown(
                '<div class="panel-card"><div class="panel-title">Evidence Chain</div></div>',
                unsafe_allow_html=True,
            )
            render_search_debug(result.search_debug)

    else:
        st.info("Pick a sample or write your own market question, then run the analysis.")


if __name__ == "__main__":
    main()
