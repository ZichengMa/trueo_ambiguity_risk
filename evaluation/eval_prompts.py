"""
Prompt templates for the SocREval evaluation pipeline.

Contains the minimal baseline prompt (Condition A) and the combined
SocREval "All" strategy evaluator prompt used to judge all three conditions.
"""

BASELINE_PROMPT_TEMPLATE = """Analyze the following prediction market question for ambiguity risk.

Market Question:
{question}

Assign a risk score from 0-100, list relevant risk tags, and explain your reasoning.

Available risk tags:
- "ambiguous_time": Time reference is unclear or relative
- "undefined_term": Key terms lack clear definition
- "unverified_source": Lacks authoritative source specification
- "vague_condition": Resolution conditions are unclear
- "ambiguous_quantity": Quantities or degrees are unclear
- "unidentified_subject": Subject identity is unclear
- "high_disputability": Prone to disputes or subjective interpretation

Respond ONLY with a valid JSON object in this exact format:
{{
    "risk_score": <integer 0-100>,
    "risk_tags": ["<tag1>", "<tag2>", ...],
    "rationale": "<detailed explanation>"
}}"""

SOCREVAL_EVALUATOR_PROMPT = """You are an expert evaluator for a prediction-market ambiguity risk scoring system.

Your task is to independently assess a market question, then compare a model's output
against ground truth using the Socratic method (Dialectic + Maieutics + Definition).

## Step 1 — Dialectic: Independent Assessment
First, form your OWN assessment of the market question's ambiguity risk. Identify the
key issues, estimate a risk score, and note which risk tags should apply.

## Step 2 — Maieutics: Qualitative Comparison
Compare the model's output against the ground truth explanation. Analyse whether the
model identified the same core issues, missed important ones, or raised spurious concerns.

## Step 3 — Definition: Dimension Scoring
Score the model's output across four explicit dimensions on a 1-5 scale:
- score_accuracy: How close is the predicted risk_score to the ground truth?
  5 = within ±5, 4 = within ±10, 3 = within ±20, 2 = within ±30, 1 = off by >30
- rationale_quality: Does the rationale correctly identify the same ambiguity issues
  as the ground truth? 5 = all key issues identified, 1 = completely off.
- tag_correctness: Are the risk_tags consistent with the ambiguity issues identified?
  5 = tags perfectly match issues, 1 = tags are irrelevant.
- overall_quality: Comprehensive quality of the full response.
  5 = excellent, 4 = good, 3 = adequate, 2 = poor, 1 = very poor.

---

### Input

**Market Question:**
{question}

**Ground Truth:**
- Risk Score: {gt_risk_score}
- Explanation: {gt_reasons}

**Model Output:**
- Risk Score: {pred_risk_score}
- Risk Tags: {pred_risk_tags}
- Rationale: {pred_rationale}

---

Respond ONLY with a valid JSON object in this exact format:
{{
    "own_assessment": {{
        "risk_score": <int>,
        "key_issues": ["<issue1>", "<issue2>", ...]
    }},
    "qualitative_analysis": "<detailed comparison>",
    "score_accuracy": <int 1-5>,
    "rationale_quality": <int 1-5>,
    "tag_correctness": <int 1-5>,
    "overall_quality": <int 1-5>
}}"""
