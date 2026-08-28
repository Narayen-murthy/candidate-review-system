import json
import os
import re
from typing import Any, Dict, List

import streamlit as st


# ============================================================
# STREAMLIT PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Candidate Review System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main {
            padding-top: 1rem;
        }

        .app-title {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }

        .app-subtitle {
            color: #777;
            font-size: 1.05rem;
            margin-bottom: 2rem;
        }

        .result-card {
            padding: 1.5rem;
            border-radius: 14px;
            border: 1px solid rgba(128,128,128,0.25);
            margin-bottom: 1rem;
        }

        .decision {
            font-size: 2rem;
            font-weight: 700;
            text-transform: uppercase;
        }

        .metric-box {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.25);
            text-align: center;
        }

        .agent-card {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid rgba(128,128,128,0.2);
            margin-bottom: 0.75rem;
        }

        .small-text {
            color: #777;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def decision_label(decision: str) -> str:
    """Format a decision for display."""
    if not decision:
        return "UNKNOWN"

    return str(decision).replace("_", " ").upper()


def extract_skills(resume_text: str) -> List[str]:
    """
    Simple fallback skill extraction.

    This is intentionally lightweight so the Streamlit UI
    continues working even when an LLM/API key is unavailable.
    """

    known_skills = [
        "Python",
        "Java",
        "JavaScript",
        "TypeScript",
        "React",
        "Node.js",
        "Docker",
        "Kubernetes",
        "AWS",
        "Azure",
        "GCP",
        "LangChain",
        "Pinecone",
        "SQL",
        "PostgreSQL",
        "MongoDB",
        "Git",
        "FastAPI",
        "Flask",
        "Django",
        "Machine Learning",
        "Deep Learning",
        "LLM",
        "RAG",
    ]

    text_lower = resume_text.lower()

    found = []

    for skill in known_skills:
        if skill.lower() in text_lower:
            found.append(skill)

    return found


def fallback_evaluate(
    resume_text: str,
    transcript_text: str,
) -> Dict[str, Any]:
    """
    Local fallback evaluator.

    This guarantees that the Streamlit application can actually
    render a result even if the OpenAI/API evaluation layer is
    unavailable.
    """

    resume_text = resume_text.strip()
    transcript_text = transcript_text.strip()

    combined = f"{resume_text}\n{transcript_text}".lower()

    skills = extract_skills(resume_text)

    positive_terms = [
        "built",
        "designed",
        "developed",
        "architect",
        "production",
        "led",
        "implemented",
        "improved",
        "reduced",
        "increased",
        "deployed",
    ]

    concern_terms = [
        "unclear",
        "lack",
        "missing",
        "unable",
        "not sure",
        "cannot",
        "could not verify",
    ]

    positive_count = sum(
        combined.count(term)
        for term in positive_terms
    )

    concern_count = sum(
        combined.count(term)
        for term in concern_terms
    )

    # Base score
    score = 55.0

    score += min(positive_count * 3.0, 25.0)
    score -= min(concern_count * 4.0, 20.0)

    # Skills contribute slightly
    score += min(len(skills) * 2.0, 10.0)

    score = max(0.0, min(100.0, score))

    if score >= 75:
        recommendation = "hire"
    elif score >= 55:
        recommendation = "maybe"
    else:
        recommendation = "reject"

    confidence = min(
        0.95,
        max(
            0.50,
            0.50 + abs(score - 50) / 100,
        ),
    )

    strengths = []

    if skills:
        strengths.extend(
            [f"Skill: {skill}" for skill in skills]
        )

    if positive_count:
        strengths.append(
            "Resume/transcript contains positive implementation and ownership signals."
        )

    if not strengths:
        strengths.append(
            "Insufficient evidence to identify strong strengths."
        )

    concerns = []

    if concern_count:
        concerns.append(
            "Some claims or details require additional verification."
        )

    if not transcript_text:
        concerns.append(
            "No interview transcript was provided."
        )

    if not concerns:
        concerns.append(
            "No major concerns detected by the fallback evaluator."
        )

    agents = [
        {
            "agent_id": "agent_tech",
            "role": "Technical",
            "decision": recommendation,
            "score": max(0, score - 5),
            "confidence": confidence,
            "rationale": (
                "Technical review based on demonstrated skills, "
                "implementation experience, and available evidence."
            ),
            "evidences": [],
        },
        {
            "agent_id": "agent_hr",
            "role": "HR",
            "decision": recommendation,
            "score": min(100, score + 3),
            "confidence": confidence,
            "rationale": (
                "HR review based on communication, leadership, "
                "and overall candidate signals."
            ),
            "evidences": [],
        },
        {
            "agent_id": "agent_hm",
            "role": "HiringManager",
            "decision": recommendation,
            "score": score,
            "confidence": confidence,
            "rationale": (
                "Hiring-manager review based on role fit and "
                "available evidence."
            ),
            "evidences": [],
        },
        {
            "agent_id": "agent_skeptic",
            "role": "Skeptic",
            "decision": recommendation,
            "score": max(0, score - 2),
            "confidence": confidence,
            "rationale": (
                "Skeptical review checking whether claims are "
                "supported by the supplied material."
            ),
            "evidences": [],
        },
    ]

    return {
        "recommendation": recommendation,
        "confidence": round(confidence, 3),
        "score": round(score, 1),
        "decisive_evidence": [],
        "strengths": strengths,
        "concerns": concerns,
        "agent_summaries": agents,
        "unresolved_disagreements": [],
        "initial_opinions": agents,
        "debate_history": [],
        "reasoning": [
            "Evaluation performed using the local fallback evaluator.",
            f"Detected {len(skills)} relevant skills.",
            f"Positive evidence signals: {positive_count}.",
            f"Concern signals: {concern_count}.",
        ],
        "metadata": {
            "provider": "local_fallback",
            "rounds": 0,
            "agent_ids": [
                "agent_tech",
                "agent_hr",
                "agent_hm",
                "agent_skeptic",
            ],
        },
    }


# ============================================================
# OPTIONAL OPENAI EVALUATOR
# ============================================================

def openai_evaluate(
    resume_text: str,
    transcript_text: str,
) -> Dict[str, Any]:
    """
    Optional OpenAI evaluator.

    If OPENAI_API_KEY is not configured, the application
    automatically falls back to local evaluation.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return fallback_evaluate(
            resume_text,
            transcript_text,
        )

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = f"""
You are a professional candidate evaluation system.

Evaluate the candidate using the resume and interview transcript below.

Return ONLY valid JSON.

Required JSON structure:

{{
    "recommendation": "hire|maybe|reject",
    "confidence": 0.0,
    "score": 0.0,
    "decisive_evidence": [],
    "strengths": [],
    "concerns": [],
    "agent_summaries": [],
    "unresolved_disagreements": [],
    "reasoning": []
}}

Resume:
{resume_text}

Interview transcript:
{transcript_text}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a structured candidate review "
                        "assistant. Return valid JSON only."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        content = response.choices[0].message.content

        if not content:
            return fallback_evaluate(
                resume_text,
                transcript_text,
            )

        # Remove accidental Markdown fences if the model returns them.
        content = content.strip()

        if content.startswith("```"):
            content = re.sub(
                r"^```(?:json)?\s*",
                "",
                content,
                flags=re.IGNORECASE,
            )

            content = re.sub(
                r"\s*```$",
                "",
                content,
            )

        result = json.loads(content)

        result.setdefault(
            "recommendation",
            "maybe",
        )

        result.setdefault(
            "confidence",
            0.5,
        )

        result.setdefault(
            "score",
            50.0,
        )

        result.setdefault(
            "strengths",
            [],
        )

        result.setdefault(
            "concerns",
            [],
        )

        result.setdefault(
            "agent_summaries",
            [],
        )

        result.setdefault(
            "reasoning",
            [],
        )

        result.setdefault(
            "metadata",
            {
                "provider": "openai",
                "rounds": 1,
            },
        )

        return result

    except Exception as exc:
        st.warning(
            "OpenAI evaluation was unavailable. "
            "Using the local evaluator instead."
        )

        result = fallback_evaluate(
            resume_text,
            transcript_text,
        )

        result["metadata"]["openai_error"] = str(exc)

        return result


# ============================================================
# FILE READING
# ============================================================

def read_uploaded_file(uploaded_file) -> str:
    """Read a Streamlit uploaded text file."""

    if uploaded_file is None:
        return ""

    try:
        raw = uploaded_file.read()

        return raw.decode(
            "utf-8",
            errors="ignore",
        )

    except Exception:
        return ""


# ============================================================
# DISPLAY FUNCTIONS
# ============================================================

def display_agent_summary(agent: Dict[str, Any]) -> None:
    """Display one agent result."""

    role = agent.get(
        "role",
        agent.get("agent_id", "Agent"),
    )

    decision = decision_label(
        agent.get("decision", "unknown")
    )

    score = safe_float(
        agent.get("score", 0)
    )

    confidence = safe_float(
        agent.get("confidence", 0)
    )

    rationale = agent.get(
        "rationale",
        "No rationale provided.",
    )

    with st.container(border=True):

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader(role)

        with col2:
            st.metric(
                "Decision",
                decision,
            )

        with col3:
            st.metric(
                "Score",
                f"{score:.1f}/100",
            )

        st.progress(
            min(max(score / 100, 0.0), 1.0)
        )

        st.caption(
            f"Confidence: {confidence:.0%}"
        )

        st.write(rationale)

        evidences = agent.get(
            "evidences",
            [],
        )

        if evidences:

            st.markdown("**Evidence**")

            for evidence in evidences:

                quote = evidence.get(
                    "quote",
                    "",
                )

                source = evidence.get(
                    "source",
                    "",
                )

                if quote:
                    st.markdown(
                        f'> "{quote}"'
                    )

                if source:
                    st.caption(source)


def display_result(result: Dict[str, Any]) -> None:
    """Render the complete candidate evaluation."""

    recommendation = str(
        result.get(
            "recommendation",
            "maybe",
        )
    ).lower()

    score = safe_float(
        result.get("score", 0)
    )

    confidence = safe_float(
        result.get("confidence", 0)
    )

    # --------------------------------------------------------
    # Main result
    # --------------------------------------------------------

    st.markdown("---")

    st.subheader("Final Recommendation")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Recommendation",
            decision_label(recommendation),
        )

    with col2:
        st.metric(
            "Candidate Score",
            f"{score:.1f}/100",
        )

    with col3:
        st.metric(
            "Confidence",
            f"{confidence:.0%}",
        )

    # --------------------------------------------------------
    # Strengths / concerns
    # --------------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader("Strengths")

        strengths = result.get(
            "strengths",
            [],
        )

        if strengths:
            for strength in strengths:
                st.success(strength)
        else:
            st.info(
                "No strengths were identified."
            )

    with right:

        st.subheader("Concerns")

        concerns = result.get(
            "concerns",
            [],
        )

        if concerns:
            for concern in concerns:
                st.warning(str(concern))
        else:
            st.info(
                "No concerns were identified."
            )

    # --------------------------------------------------------
    # Evidence
    # --------------------------------------------------------

    decisive = result.get(
        "decisive_evidence",
        [],
    )

    if decisive:

        st.subheader(
            "Decisive Evidence"
        )

        for item in decisive:

            if isinstance(item, dict):

                quote = item.get(
                    "quote",
                    "",
                )

                source = item.get(
                    "source",
                    "",
                )

                if quote:
                    st.markdown(
                        f'> "{quote}"'
                    )

                if source:
                    st.caption(source)

            else:
                st.write(item)

    # --------------------------------------------------------
    # Agent reviews
    # --------------------------------------------------------

    agents = result.get(
        "agent_summaries",
        [],
    )

    if agents:

        st.subheader(
            "Multi-Agent Review"
        )

        for agent in agents:
            display_agent_summary(agent)

    # --------------------------------------------------------
    # Reasoning
    # --------------------------------------------------------

    reasoning = result.get(
        "reasoning",
        [],
    )

    if reasoning:

        with st.expander(
            "Evaluation Reasoning"
        ):

            for item in reasoning:
                st.write(
                    f"• {item}"
                )

    # --------------------------------------------------------
    # Raw JSON
    # --------------------------------------------------------

    with st.expander(
        "View Complete JSON"
    ):

        st.json(result)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Candidate Review")

    st.write(
        "Upload candidate information and "
        "run the multi-agent evaluation."
    )

    st.divider()

    st.subheader("Evaluation")

    use_openai = st.checkbox(
        "Use OpenAI when API key is available",
        value=True,
    )

    st.caption(
        "Without an API key, the app uses "
        "the built-in local evaluator."
    )

    st.divider()

    st.subheader("Input")

    resume_file = st.file_uploader(
        "Upload Resume",
        type=[
            "txt",
            "md",
        ],
    )

    transcript_file = st.file_uploader(
        "Upload Interview Transcript",
        type=[
            "txt",
            "md",
        ],
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="app-title">👤 Candidate Review System</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-subtitle">
        AI-assisted candidate evaluation using resume evidence,
        interview transcripts, and multi-agent review.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT AREA
# ============================================================

col1, col2 = st.columns(2)

with col1:

    st.subheader("Resume")

    resume_text = st.text_area(
        "Paste the candidate's resume",
        height=350,
        placeholder=(
            "Paste resume text here..."
        ),
    )

    if resume_file is not None:

        uploaded_resume = read_uploaded_file(
            resume_file
        )

        if uploaded_resume:

            resume_text = uploaded_resume

            st.success(
                f"Loaded: {resume_file.name}"
            )


with col2:

    st.subheader("Interview Transcript")

    transcript_text = st.text_area(
        "Paste the interview transcript",
        height=350,
        placeholder=(
            "Paste interview transcript here..."
        ),
    )

    if transcript_file is not None:

        uploaded_transcript = read_uploaded_file(
            transcript_file
        )

        if uploaded_transcript:

            transcript_text = uploaded_transcript

            st.success(
                f"Loaded: {transcript_file.name}"
            )


# ============================================================
# EVALUATION BUTTON
# ============================================================

st.markdown("")

evaluate_button = st.button(
    "🚀 Evaluate Candidate",
    type="primary",
    use_container_width=True,
)


if evaluate_button:

    if not resume_text.strip():

        st.error(
            "Please provide a resume before evaluating."
        )

    elif not transcript_text.strip():

        st.error(
            "Please provide an interview transcript before evaluating."
        )

    else:

        with st.spinner(
            "Analyzing candidate evidence..."
        ):

            if use_openai:

                result = openai_evaluate(
                    resume_text,
                    transcript_text,
                )

            else:

                result = fallback_evaluate(
                    resume_text,
                    transcript_text,
                )

            st.session_state.result = result


# ============================================================
# RESULTS
# ============================================================

if st.session_state.result is not None:

    display_result(
        st.session_state.result
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Candidate Review System • Streamlit"
)
