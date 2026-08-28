```python
#!/usr/bin/env python3
"""
Single-file Candidate Review System

Features:
- Flask web application
- Candidate resume + interview transcript input
- File upload support
- Dummy LLM by default
- Optional OpenAI LLM
- 4 independent agents:
    1. Technical
    2. HR
    3. Hiring Manager
    4. Skeptic
- 3-round multi-agent debate
- Evidence-weighted final recommendation
- JSON API
- JSON report download
- Health endpoint

Run:
    python candidate_review.py

Then open:
    http://localhost:8080

Optional:
    python candidate_review.py --real-llm
    python candidate_review.py --output report.json
"""

import os
import re
import json
import time
import argparse
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from time import sleep

from flask import (
    Flask,
    request,
    jsonify,
    send_file,
    render_template_string,
)
from werkzeug.utils import secure_filename

try:
    import openai
except Exception:
    openai = None


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class Evidence:
    quote: str
    source: str
    start: Optional[int] = None
    end: Optional[int] = None


@dataclass
class Claim:
    text: str
    evidences: List[Evidence] = field(default_factory=list)


@dataclass
class CandidateProfile:
    name: Optional[str] = None
    email: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    experiences: List[str] = field(default_factory=list)
    education: List[str] = field(default_factory=list)
    claims: List[Claim] = field(default_factory=list)
    raw_resume: Optional[str] = None
    raw_transcript: Optional[str] = None


@dataclass
class AgentOpinion:
    agent_id: str
    role: str
    decision: str
    score: float
    confidence: float
    rationale: str
    evidences: List[Evidence] = field(default_factory=list)


@dataclass
class DebateReply:
    agent_id: str
    replies_to: str
    text: str
    updated_opinion: Optional[AgentOpinion] = None
    round_index: Optional[int] = None
    timestamp: Optional[float] = None


@dataclass
class FinalReport:
    recommendation: str
    confidence: float
    decisive_evidence: List[Evidence]
    strengths: List[str]
    concerns: List[str]
    agent_summaries: List[AgentOpinion]
    unresolved_disagreements: List[str]
    initial_opinions: List[AgentOpinion] = field(default_factory=list)
    debate_history: List[DebateReply] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# SAMPLE DATA
# ============================================================

SAMPLE_RESUME = """Rohan Malhotra
Senior AI/Backend Engineer

Summary
AI engineer with 3.5 years of experience building multi-agent LLM systems and Python backends. Led design of a production agent platform now handling thousands of daily freight exceptions. Known for moving fast and shipping under pressure.

Experience
Senior AI Engineer — Voltrix Logistics Tech (Jan 2025 – Present, 7 months)
• Designed and built the exception-handling engine end-to-end for Voltrix’s multi-agent freight ops platform (planner/executor/reviewer pattern), cutting manual exception review time by 40%.
• Owned prompt design and model routing across GPT-4 and open-weight SLMs, reducing inference cost by ~30%.
• Sole architect of the retry/escalation logic now running in production, handling 5,000+ freight exceptions/month.
• Presented the system design at a company-wide tech talk.

AI Engineer — Quickship Data Systems (Feb 2024 – Dec 2024, 11 months)
• Built a RAG pipeline over carrier rate documents using LangChain + Pinecone, cutting manual rate lookup time significantly.
• Improved BOL/invoice extraction accuracy through better OCR pre-processing.

Backend Developer — Nimbus Cloud Solutions (Aug 2022 – Jan 2024, 1.5 years)
• Built Python microservices for a SaaS analytics product used by 50+ enterprise clients.
• Led a 4-person team migrating a legacy monolith to microservices.

Skills
Python, FastAPI, LangGraph, CrewAI, MongoDB, React (basic), RAG, Vector Search (Pinecone, FAISS), Prompt Engineering, Docker, Kubernetes

Education
B.Tech Computer Science, 2022

Certifications
• LangChain for LLM Application Development (2024)
"""


SAMPLE_TRANSCRIPT = """Interviewer: Walk me through the exception-handling engine you built at Voltrix.
Candidate: It’s planner-executor-reviewer. Failures come in, get classified, retried or escalated, then double-checked. I designed the whole retry/escalation logic.

Interviewer: What made you choose that structure over a simpler rule-based system?
Candidate: Rules don’t scale. Too many failure types — timeouts, bad EDI, missing BOL fields. Agents handle that better.

Interviewer: How do you measure whether the reviewer agent is actually catching real problems?
Candidate: We track override rate. It’s low. I’d have to check the exact number though, haven’t looked recently.

Interviewer: What’s your approach to model routing?
Candidate: Cost-based. Simple stuff to the SLM, harder reasoning to GPT-4. No formal study, just tuned it as things broke.

Interviewer: Tell me about a time you disagreed with a teammate on a technical decision.
Candidate: Teammate wanted to hardcode more categories up front. I pushed for the agent approach. We went with mine.

Interviewer: Who actually wrote the retry/escalation logic that’s in production now?
Candidate: I designed it. Priya did a lot of the implementation, I reviewed her PRs. I was the architect.

Interviewer: (Skeptic follow-up) Your resume says “sole architect.” But it sounds like Priya built a lot of it. Can you clarify?
Candidate: Fine — “sole architect” is probably too strong. I led the design, she built most of the production version.

Interviewer: Why should we invest in ramping you up here versus someone with more freight-domain experience?
Candidate: I move fast. I’ve built something structurally close to this already. I don’t think I’d need much ramp time.

Interviewer: This role needs long-term ownership of production reliability. How do you feel about being on-call for agent failures?
Candidate: Fine, I’ve done on-call before. Though Voltrix’s user base is still small, so I haven’t seen serious incident volume yet.

Interviewer: You’ve had three roles in 3.5 years, each under a year except the first. What’s driving that?
Candidate: Better pay and title, mostly. Voltrix is more aligned with what I want long-term.
"""


# ============================================================
# PROFILE BUILDER
# ============================================================

def parse_basic_contact(resume_text: str):
    name = None
    email = None

    lines = [
        line.rstrip()
        for line in resume_text.splitlines()
        if line.strip()
    ]

    if lines:
        first = lines[0]

        if (
            2 <= len(first.split()) <= 4
            and re.match(r"^[A-Z][a-z]", first)
        ):
            name = first.strip()

    email_match = re.search(
        r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)",
        resume_text,
    )

    if email_match:
        email = email_match.group(1)

    return name, email


def extract_skills(resume_text: str) -> List[str]:
    skills = []

    match = re.search(
        r"Skills?:\s*(.+)",
        resume_text,
        re.IGNORECASE,
    )

    if match:
        raw = match.group(1)

        parts = re.split(
            r"[,|;]\s*",
            raw,
        )

        skills = [
            p.strip()
            for p in parts
            if p.strip()
        ]

    else:
        tokens = [
            "Python",
            "Java",
            "C++",
            "Go",
            "SQL",
            "React",
            "Node",
            "Docker",
            "Kubernetes",
            "AWS",
            "GCP",
        ]

        for token in tokens:
            if re.search(
                r"\b" + re.escape(token) + r"\b",
                resume_text,
                re.IGNORECASE,
            ):
                skills.append(token)

    seen = set()
    result = []

    for skill in skills:
        if skill.lower() not in seen:
            seen.add(skill.lower())
            result.append(skill)

    return result


def extract_experiences(resume_text: str) -> List[str]:
    experiences = []

    for line in resume_text.splitlines():

        if re.search(
            r"\b(Engineer|Developer|Manager|Intern|Consultant|SWE)\b",
            line,
            re.IGNORECASE,
        ):

            if len(line.strip()) > 5:
                experiences.append(line.strip())

    return experiences


def extract_claims(
    resume_text: str,
    transcript_text: str,
) -> List[Claim]:

    claims = []

    # Resume claims
    for line_number, line in enumerate(
        resume_text.splitlines(),
        start=1,
    ):

        for match in re.finditer(
            r"([^.]*\b(?:years|led|improved|built|designed|implemented|managed)\b[^.]*\.)",
            line,
            re.IGNORECASE,
        ):

            sentence = match.group(1).strip()

            claims.append(
                Claim(
                    text=sentence,
                    evidences=[
                        Evidence(
                            quote=sentence,
                            source=f"resume:L{line_number}",
                        )
                    ],
                )
            )

    # Transcript claims
    for line_number, line in enumerate(
        transcript_text.splitlines(),
        start=1,
    ):

        if re.search(
            r"\b(I led|I built|I designed|I implemented|I managed|I have \d+ years|we built|my team)\b",
            line,
            re.IGNORECASE,
        ):

            text = line.strip()

            if text:

                speaker_match = re.match(
                    r"^(\w+):\s*(.*)$",
                    text,
                )

                if speaker_match:
                    speaker = speaker_match.group(1)
                    quote = speaker_match.group(2)
                else:
                    speaker = None
                    quote = text

                claims.append(
                    Claim(
                        text=quote,
                        evidences=[
                            Evidence(
                                quote=quote,
                                source=(
                                    f"transcript:"
                                    f"{speaker or 'line'}"
                                    f"@L{line_number}"
                                ),
                            )
                        ],
                    )
                )

    return claims


def build_candidate_profile(
    resume_text: str,
    transcript_text: str,
) -> CandidateProfile:

    name, email = parse_basic_contact(resume_text)

    skills = extract_skills(resume_text)

    experiences = extract_experiences(resume_text)

    claims = extract_claims(
        resume_text,
        transcript_text,
    )

    education = []

    for line in resume_text.splitlines():

        match = re.search(
            r"(B\.S\.|Bachelors|M\.S\.|Masters|Ph\.D|Bachelor of|Master of)(.*)",
            line,
            re.IGNORECASE,
        )

        if match:
            education.append(line.strip())

    if not skills:
        skills = ["(unknown)"]

    return CandidateProfile(
        name=name,
        email=email,
        skills=skills,
        experiences=experiences,
        education=education,
        claims=claims,
        raw_resume=resume_text,
        raw_transcript=transcript_text,
    )


# ============================================================
# LLM CLIENT
# ============================================================

class LLMClient:

    def __init__(self, provider: str = "auto"):

        self.api_key = os.getenv("OPENAI_API_KEY")

        self.model = os.getenv(
            "LLM_MODEL",
            "gpt-3.5-turbo",
        )

        self.client = None

        if provider == "dummy":
            self.provider = "dummy"
            return

        if provider == "openai":

            if not self.api_key:
                self.provider = "dummy"
                return

        if self.api_key and openai is not None:

            try:

                if hasattr(openai, "OpenAI"):

                    self.client = openai.OpenAI(
                        api_key=self.api_key
                    )

                else:
                    openai.api_key = self.api_key

                self.provider = "openai"

            except Exception:

                self.provider = "dummy"

        else:

            self.provider = "dummy"


    def _call_openai(self, prompt: str) -> str:

        last_exception = None

        for attempt in range(3):

            try:

                if self.provider != "openai":
                    raise RuntimeError(
                        "OpenAI provider not configured"
                    )

                if hasattr(openai, "ChatCompletion"):

                    response = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[
                            {
                                "role": "user",
                                "content": prompt,
                            }
                        ],
                        temperature=0.0,
                        max_tokens=1000,
                    )

                    return response[
                        "choices"
                    ][0][
                        "message"
                    ][
                        "content"
                    ]

                if hasattr(openai, "OpenAI"):

                    client = (
                        self.client
                        or openai.OpenAI(
                            api_key=self.api_key
                        )
                    )

                    response = (
                        client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "user",
                                    "content": prompt,
                                }
                            ],
                            temperature=0.0,
                            max_tokens=1000,
                        )
                    )

                    try:

                        return response[
                            "choices"
                        ][0][
                            "message"
                        ][
                            "content"
                        ]

                    except Exception:

                        return (
                            response
                            .choices[0]
                            .message
                            .content
                        )

                raise RuntimeError(
                    "No compatible OpenAI API found"
                )

            except Exception as exception:

                last_exception = exception

                sleep(1 + attempt)

        raise last_exception


    def _extract_json(self, text: str) -> Any:

        try:
            return json.loads(text)

        except Exception:
            pass

        match = re.search(
            r"\{.*\}",
            text,
            re.S,
        )

        if match:

            candidate = match.group(0)

            try:
                return json.loads(candidate)

            except Exception:

                candidate2 = candidate.replace(
                    "'",
                    '"',
                )

                try:
                    return json.loads(
                        candidate2
                    )

                except Exception:
                    pass

        raise ValueError(
            "Could not parse JSON from model output"
        )


    def _validate_opinion(
        self,
        parsed: Any,
    ) -> Dict[str, Any]:

        if not isinstance(parsed, dict):

            return {
                "decision": "maybe",
                "score": 60,
                "confidence": 0.5,
                "rationale": "Invalid agent output",
                "evidences": [],
            }

        required = {
            "decision",
            "score",
            "confidence",
            "rationale",
            "evidences",
        }

        if not required.issubset(
            set(parsed.keys())
        ):

            parsed2 = {
                key: parsed.get(key)
                for key in (
                    required
                    & set(parsed.keys())
                )
            }

            parsed2.setdefault(
                "decision",
                "maybe",
            )

            parsed2.setdefault(
                "score",
                60,
            )

            parsed2.setdefault(
                "confidence",
                0.5,
            )

            parsed2.setdefault(
                "rationale",
                "Incomplete output, using conservative defaults",
            )

            parsed2.setdefault(
                "evidences",
                [],
            )

            return parsed2

        return parsed


    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:

        if self.provider == "openai":

            text = self._call_openai(prompt)

            try:

                parsed = self._extract_json(
                    text
                )

                return self._validate_opinion(
                    parsed
                )

            except Exception as exception:

                return {
                    "decision": "maybe",
                    "score": 60,
                    "confidence": 0.5,
                    "rationale": (
                        "Could not parse model JSON: "
                        f"{exception}"
                    ),
                    "evidences": [],
                }

        # ----------------------------------------------------
        # DUMMY LLM
        # ----------------------------------------------------

        low = prompt.lower()

        is_debate = (
            "respond by addressing" in low
            or "other agents' opinions" in low
            or "reply to" in low
        )

        if is_debate:

            match = re.search(
                r"you are\s+([A-Za-z ]+?)\s+agent",
                prompt,
                re.IGNORECASE,
            )

            role = (
                match.group(1).strip().lower()
                if match
                else "unknown"
            )

            if "technical" in role:

                return {
                    "reply_to": "agent_skeptic",
                    "text": (
                        "Technical: I acknowledge "
                        "verification concerns, but the "
                        "transcript/resume shows relevant "
                        "technology and measurable claims. "
                        "Recommend targeted technical follow-up."
                    ),
                    "updated_opinion": {
                        "decision": "maybe",
                        "score": 65,
                        "confidence": 0.65,
                        "rationale": (
                            "Technical reduces score slightly "
                            "because of missing design details, "
                            "but retains partial confidence "
                            "due to relevant skills."
                        ),
                        "evidences": [
                            {
                                "quote": (
                                    "Designed and built the "
                                    "exception-handling engine "
                                    "end-to-end for Voltrix’s "
                                    "multi-agent freight ops platform."
                                ),
                                "source": "resume:L2",
                            },
                            {
                                "quote": (
                                    "Built a RAG pipeline over "
                                    "carrier rate documents using "
                                    "LangChain + Pinecone"
                                ),
                                "source": "resume:L3",
                            },
                        ],
                    },
                }

            if "hr" in role:

                return {
                    "reply_to": "agent_skeptic",
                    "text": (
                        "HR: Leadership and communication "
                        "are clear. These are positive signals "
                        "even if artifacts are requested."
                    ),
                    "updated_opinion": {
                        "decision": "hire",
                        "score": 82,
                        "confidence": 0.75,
                        "rationale": (
                            "Strong communication and "
                            "leadership signals."
                        ),
                        "evidences": [
                            {
                                "quote": (
                                    "Presented the system design "
                                    "at a company-wide tech talk."
                                ),
                                "source": "resume:L2",
                            },
                            {
                                "quote": (
                                    "I led the design, she built "
                                    "most of the production version."
                                ),
                                "source": (
                                    "transcript:Candidate@L7"
                                ),
                            },
                        ],
                    },
                }

            if "hiring" in role:

                return {
                    "reply_to": "agent_tech",
                    "text": (
                        "HiringManager: The candidate looks "
                        "like a plausible fit. Recommend proceeding "
                        "to a technical interview."
                    ),
                    "updated_opinion": {
                        "decision": "maybe",
                        "score": 80,
                        "confidence": 0.72,
                        "rationale": (
                            "Role-fit is plausible; "
                            "recommend technical validation."
                        ),
                        "evidences": [
                            {
                                "quote": (
                                    "Sole architect of the "
                                    "retry/escalation logic now "
                                    "running in production."
                                ),
                                "source": "resume:L2",
                            },
                            {
                                "quote": (
                                    "I move fast. I’ve built "
                                    "something structurally close "
                                    "to this already."
                                ),
                                "source": (
                                    "transcript:Candidate@L8"
                                ),
                            },
                        ],
                    },
                }

            if "skeptic" in role:

                return {
                    "reply_to": "agent_hm",
                    "text": (
                        "Skeptic: Throughput and improvement "
                        "claims lack corroborating evidence. "
                        "Request artifacts, logs, or specifics "
                        "before hiring."
                    ),
                }

            return {
                "reply_to": "agent_hm",
                "text": (
                    "I question unverifiable claims; "
                    "please provide more detail."
                ),
                "updated_opinion": {
                    "decision": "maybe",
                    "score": 60,
                    "confidence": 0.6,
                    "rationale": "Cautious",
                    "evidences": [],
                },
            }

        # ----------------------------------------------------
        # INITIAL DUMMY OPINIONS
        # ----------------------------------------------------

        if "technical" in low:

            return {
                "decision": "maybe",
                "score": 78,
                "confidence": 0.75,
                "rationale": (
                    "Strong hands-on agent and production "
                    "experience but lacking artifacts for "
                    "some claimed metrics."
                ),
                "evidences": [
                    {
                        "quote": (
                            "Designed and built the "
                            "exception-handling engine "
                            "end-to-end for Voltrix’s "
                            "multi-agent freight ops platform."
                        ),
                        "source": "resume:L2",
                    },
                    {
                        "quote": (
                            "Sole architect of the "
                            "retry/escalation logic now "
                            "running in production."
                        ),
                        "source": "resume:L2",
                    },
                ],
            }

        if "hr" in low or "culture" in low:

            return {
                "decision": "hire",
                "score": 80,
                "confidence": 0.7,
                "rationale": (
                    "Leadership and communication look good."
                ),
                "evidences": [
                    {
                        "quote": (
                            "Presented the system design "
                            "at a company-wide tech talk."
                        ),
                        "source": "resume:L2",
                    }
                ],
            }

        if (
            "hiringmanager" in low
            or "hiring manager" in low
        ):

            return {
                "decision": "maybe",
                "score": 75,
                "confidence": 0.68,
                "rationale": (
                    "Seems like a fit but needs to "
                    "validate scale and depth."
                ),
                "evidences": [
                    {
                        "quote": (
                            "handling 5,000+ freight "
                            "exceptions/month."
                        ),
                        "source": "resume:L2",
                    }
                ],
            }

        if "skeptic" in low:

            return {
                "decision": "reject",
                "score": 35,
                "confidence": 0.7,
                "rationale": (
                    "Overstated claims, short tenures, "
                    "and missing metrics are concerning."
                ),
                "evidences": [
                    {
                        "quote": (
                            "Fine — 'sole architect' "
                            "is probably too strong."
                        ),
                        "source": (
                            "transcript:Candidate@L7"
                        ),
                    }
                ],
            }

        return {
            "decision": "maybe",
            "score": 60,
            "confidence": 0.5,
            "rationale": "Generic simulated opinion.",
            "evidences": [
                {
                    "quote": (
                        "I have 3.5 years of experience "
                        "building multi-agent LLM systems "
                        "and Python backends."
                    ),
                    "source": "resume:L1",
                }
            ],
        }


# ============================================================
# AGENTS
# ============================================================

ROLE_GUIDANCE = {

    "Technical":
        "- Assess technical skills, depth, relevant projects, measurable impact.",

    "HR":
        "- Assess communication, teamwork, honesty, and cultural fit.",

    "HiringManager":
        "- Assess role fit, potential, and ability to deliver.",

    "Skeptic":
        "- Find contradictions, exaggerations, missing details, or red flags.",
}


PROMPT_TEMPLATE = """
You are the {role} agent.

Evaluate the candidate based ONLY on the provided CandidateProfile.

Rules:

1. Do NOT use external knowledge.
2. Do NOT hallucinate.
3. Cite evidence from the provided profile.
4. Output decision, score, confidence, rationale and evidence.

CandidateProfile:

{profile_json}

Questions for your role:

{role_guidance}
"""


class Agent:

    def __init__(
        self,
        agent_id: str,
        role: str,
        llm_client: LLMClient,
    ):

        self.agent_id = agent_id
        self.role = role
        self.llm = llm_client


    def build_prompt(
        self,
        profile_json: str,
    ):

        guidance = ROLE_GUIDANCE.get(
            self.role,
            "",
        )

        safe_profile = (
            profile_json
            .replace("{", "{{")
            .replace("}", "}}")
        )

        return PROMPT_TEMPLATE.format(
            role=self.role,
            profile_json=safe_profile,
            role_guidance=guidance,
        )


    def run_initial_opinion(
        self,
        profile_json: str,
    ) -> AgentOpinion:

        prompt = self.build_prompt(
            profile_json
        )

        raw = self.llm.generate(prompt)

        parsed = (
            raw
            if isinstance(raw, dict)
            else json.loads(raw)
        )

        parsed = self.llm._validate_opinion(
            parsed
        )

        evidences = []

        for evidence in parsed.get(
            "evidences",
            [],
        ):

            evidences.append(
                Evidence(
                    quote=evidence.get(
                        "quote",
                        "",
                    ),
                    source=evidence.get(
                        "source",
                        "",
                    ),
                )
            )

        if (
            self.llm.provider == "openai"
            and not evidences
        ):

            evidences.append(
                Evidence(
                    quote="(no evidence provided by model)",
                    source="model_output",
                )
            )

            parsed["confidence"] = min(
                0.4,
                float(
                    parsed.get(
                        "confidence",
                        0.5,
                    )
                ),
            )

        return AgentOpinion(
            agent_id=self.agent_id,
            role=self.role,
            decision=parsed.get(
                "decision",
                "maybe",
            ),
            score=float(
                parsed.get(
                    "score",
                    0,
                )
            ),
            confidence=float(
                parsed.get(
                    "confidence",
                    0,
                )
            ),
            rationale=parsed.get(
                "rationale",
                "",
            ),
            evidences=evidences,
        )


    def run_debate_response(
        self,
        profile_json: str,
        other_opinions: List[AgentOpinion],
        round_index: int = 0,
    ) -> DebateReply:

        others_summary = "\n".join(
            [
                (
                    f"{o.agent_id} ({o.role}): "
                    f"decision={o.decision}, "
                    f"score={o.score}, "
                    f"evidences="
                    f"{[e.quote for e in o.evidences]}"
                )
                for o in other_opinions
            ]
        )

        debate_prompt = (
            f"You are {self.role} agent.\n"
            f"Other agents' opinions:\n"
            f"{others_summary}\n\n"
            f"Respond by addressing at least "
            f"one other agent's point.\n"
            f"Profile:\n{profile_json}"
        )

        raw = self.llm.generate(
            debate_prompt
        )

        parsed = (
            raw
            if isinstance(raw, dict)
            else json.loads(raw)
        )

        reply_to = parsed.get(
            "reply_to",
            "",
        )

        text = parsed.get(
            "text",
            "",
        )

        updated_opinion = None

        if parsed.get("updated_opinion"):

            updated = self.llm._validate_opinion(
                parsed["updated_opinion"]
            )

            evidences = []

            for evidence in updated.get(
                "evidences",
                [],
            ):

                evidences.append(
                    Evidence(
                        quote=evidence.get(
                            "quote",
                            "",
                        ),
                        source=evidence.get(
                            "source",
                            "",
                        ),
                    )
                )

            updated_opinion = AgentOpinion(
                agent_id=self.agent_id,
                role=self.role,
                decision=updated.get(
                    "decision",
                    "maybe",
                ),
                score=float(
                    updated.get(
                        "score",
                        0,
                    )
                ),
                confidence=float(
                    updated.get(
                        "confidence",
                        0,
                    )
                ),
                rationale=updated.get(
                    "rationale",
                    "",
                ),
                evidences=evidences,
            )

        return DebateReply(
            agent_id=self.agent_id,
            replies_to=reply_to,
            text=text,
            updated_opinion=updated_opinion,
            round_index=round_index,
            timestamp=time.time(),
        )


# ============================================================
# PIPELINE
# ============================================================

def run_pipeline(
    resume_text: str,
    transcript_text: str,
    provider: str = "dummy",
) -> FinalReport:

    profile = build_candidate_profile(
        resume_text,
        transcript_text,
    )

    profile_json = json.dumps(
        asdict(profile),
        indent=2,
    )

    llm_client = LLMClient(
        provider=provider
    )

    agent_defs = [
        ("agent_tech", "Technical"),
        ("agent_hr", "HR"),
        ("agent_hm", "HiringManager"),
        ("agent_skeptic", "Skeptic"),
    ]

    # --------------------------------------------------------
    # Initial opinions
    # --------------------------------------------------------

    initial_opinions = []

    for agent_id, role in agent_defs:

        agent = Agent(
            agent_id,
            role,
            llm_client,
        )

        opinion = agent.run_initial_opinion(
            profile_json
        )

        initial_opinions.append(
            opinion
        )

    opinions = list(initial_opinions)

    # --------------------------------------------------------
    # Three-round debate
    # --------------------------------------------------------

    debate_replies = []

    rounds = 3

    for round_index in range(rounds):

        for agent_id, role in agent_defs:

            agent = Agent(
                agent_id,
                role,
                llm_client,
            )

            others = [
                opinion
                for opinion in opinions
                if opinion.agent_id != agent_id
            ]

            reply = agent.run_debate_response(
                profile_json,
                others,
                round_index,
            )

            debate_replies.append(
                reply
            )

            if reply.updated_opinion:

                for index, opinion in enumerate(
                    opinions
                ):

                    if opinion.agent_id == agent_id:

                        opinions[index] = (
                            reply.updated_opinion
                        )

                        break

    # --------------------------------------------------------
    # Evidence-weighted aggregation
    # --------------------------------------------------------

    role_weights = {
        "HiringManager": 1.4,
        "Technical": 1.2,
        "HR": 1.0,
        "Skeptic": 1.0,
    }

    weighted_sum = 0.0
    weight_total = 0.0

    reasoning = []

    evidence_scores = {}

    for opinion in opinions:

        evidence_count = max(
            0,
            len(opinion.evidences),
        )

        role_weight = role_weights.get(
            opinion.role,
            1.0,
        )

        weight = (
            opinion.confidence
            * role_weight
            * (
                1
                + evidence_count / 4.0
            )
        )

        weighted_sum += (
            opinion.score
            * weight
        )

        weight_total += weight

        for evidence in opinion.evidences:

            key = (
                f"{evidence.quote}"
                f" || "
                f"{evidence.source}"
            )

            evidence_scores[key] = (
                evidence_scores.get(
                    key,
                    0.0,
                )
                + (
                    opinion.score
                    * opinion.confidence
                    * role_weight
                )
            )

    final_score = (
        weighted_sum / weight_total
        if weight_total
        else 60
    )

    # --------------------------------------------------------
    # Recommendation
    # --------------------------------------------------------

    if final_score >= 75:
        recommendation = "hire"

    elif final_score >= 55:
        recommendation = "maybe"

    else:
        recommendation = "reject"

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence_numerator = sum(
        [
            opinion.confidence
            * role_weights.get(
                opinion.role,
                1.0,
            )
            for opinion in opinions
        ]
    )

    confidence_denominator = sum(
        [
            role_weights.get(
                opinion.role,
                1.0,
            )
            for opinion in opinions
        ]
    )

    final_confidence = (
        confidence_numerator
        / confidence_denominator
        if confidence_denominator
        else 0.5
    )

    # --------------------------------------------------------
    # Decisive evidence
    # --------------------------------------------------------

    top_evidence = sorted(
        evidence_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:3]

    decisive_evidence = []

    for key, score in top_evidence:

        quote, source = key.split(
            " || ",
            1,
        )

        decisive_evidence.append(
            Evidence(
                quote=quote,
                source=source,
            )
        )

        reasoning.append(
            f"Evidence '{quote}' from {source} "
            f"contributed score weight {score:.1f}"
        )

    # --------------------------------------------------------
    # Skeptic override
    # --------------------------------------------------------

    skeptic = next(
        (
            opinion
            for opinion in opinions
            if opinion.role == "Skeptic"
        ),
        None,
    )

    hiring_manager = next(
        (
            opinion
            for opinion in opinions
            if opinion.role == "HiringManager"
        ),
        None,
    )

    if (
        skeptic
        and skeptic.score < 30
        and skeptic.confidence > 0.75
    ):

        if (
            hiring_manager
            and hiring_manager.score >= 85
        ):

            recommendation = "maybe"

            reasoning.append(
                "Skeptic raised a high-confidence "
                "low score but HiringManager was strongly "
                "favorable; compromise to maybe."
            )

        else:

            recommendation = "reject"

            reasoning.append(
                "Skeptic raised a high-confidence "
                "low score and there was no strong "
                "HiringManager override."
            )

    # --------------------------------------------------------
    # Disagreements
    # --------------------------------------------------------

    decisions = {
        opinion.decision
        for opinion in opinions
    }

    unresolved = []

    if len(decisions) > 1:

        unresolved = [
            (
                f"{opinion.agent_id}"
                f"({opinion.role}) => "
                f"{opinion.decision} "
                f"(score={opinion.score}, "
                f"conf={opinion.confidence})"
            )
            for opinion in opinions
        ]

    # --------------------------------------------------------
    # Strengths
    # --------------------------------------------------------

    strengths = [
        f"Skill: {skill}"
        for skill in profile.skills
    ][:10]

    # --------------------------------------------------------
    # Concerns
    # --------------------------------------------------------

    concerns = []

    for evidence in decisive_evidence:

        text = evidence.quote.lower()

        if any(
            word in text
            for word in [
                "no ",
                "not ",
                "lack",
                "unverified",
                "risk",
            ]
        ):

            concerns.append(
                evidence.quote
            )

    if not concerns:

        concerns = [
            evidence.quote
            for evidence in decisive_evidence
        ][:5]

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "provider": llm_client.provider,
        "timestamp": time.time(),
        "rounds": rounds,
        "agent_ids": [
            agent_id
            for agent_id, role
            in agent_defs
        ],
        "final_score": round(
            float(final_score),
            2,
        ),
    }

    return FinalReport(
        recommendation=recommendation,
        confidence=round(
            float(final_confidence),
            3,
        ),
        decisive_evidence=decisive_evidence,
        strengths=strengths,
        concerns=concerns,
        agent_summaries=opinions,
        unresolved_disagreements=unresolved,
        initial_opinions=initial_opinions,
        debate_history=debate_replies,
        reasoning=reasoning,
        metadata=metadata,
    )


# ============================================================
# EVALUATE WRAPPER
# ============================================================

def evaluate(
    resume_text: str,
    transcript_text: str,
    provider: str = "dummy",
) -> Dict[str, Any]:

    report = run_pipeline(
        resume_text,
        transcript_text,
        provider=provider,
    )

    return asdict(report)


# ============================================================
# FLASK APP
# ============================================================

app = Flask(__name__)


# ============================================================
# HTML
# ============================================================

HTML_PAGE = r"""
<!DOCTYPE html>
<html lang="en">

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>Candidate Review System</title>

<style>

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    font-family:
        Inter,
        Arial,
        sans-serif;

    background:
        radial-gradient(
            circle at top,
            #172554,
            #020617 55%
        );

    color: #f8fafc;
    min-height: 100vh;
}

.container {
    max-width: 1100px;
    margin: auto;
    padding: 45px 20px 70px;
}

.hero {
    text-align: center;
    margin-bottom: 35px;
}

.badge {
    display: inline-block;

    padding: 8px 14px;

    border-radius: 999px;

    background: #052e16;

    border: 1px solid #166534;

    color: #86efac;

    font-size: 12px;

    font-weight: 800;

    letter-spacing: .5px;
}

h1 {
    font-size:
        clamp(38px, 7vw, 65px);

    letter-spacing: -3px;

    margin:
        16px 0 12px;
}

.hero p {
    max-width: 700px;

    margin: auto;

    color: #94a3b8;

    line-height: 1.7;
}

.card {
    background: rgba(
        15,
        23,
        42,
        .92
    );

    border:
        1px solid
        #1e293b;

    border-radius: 20px;

    padding: 27px;

    margin-bottom: 20px;

    box-shadow:
        0 20px 60px
        rgba(0,0,0,.25);
}

.title {
    font-size: 20px;

    font-weight: 800;

    margin-bottom: 7px;
}

.desc {
    color: #64748b;

    font-size: 14px;

    margin-bottom: 20px;
}

.grid {
    display: grid;

    grid-template-columns:
        1fr 1fr;

    gap: 18px;
}

.field {
    margin-bottom: 10px;
}

label {
    display: block;

    font-size: 13px;

    font-weight: 700;

    margin-bottom: 8px;
}

input,
textarea {
    width: 100%;

    background: #020617;

    color: #f8fafc;

    border:
        1px solid
        #334155;

    border-radius: 12px;

    padding: 13px;

    font:
        inherit;

    outline: none;
}

input:focus,
textarea:focus {
    border-color: #60a5fa;
}

textarea {
    min-height: 190px;

    resize: vertical;
}

.file-box {
    margin-top: 12px;

    padding: 12px;

    border:
        1px dashed
        #475569;

    border-radius: 12px;
}

.sample {
    display: flex;

    align-items: center;

    gap: 10px;

    margin-top: 10px;

    padding: 14px;

    background: #172554;

    border-radius: 12px;

    color: #bfdbfe;
}

.sample input {
    width: auto;
}

.agents {
    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 13px;
}

.agent {
    text-align: center;

    padding: 20px 10px;

    background: #020617;

    border:
        1px solid
        #1e293b;

    border-radius: 14px;
}

.icon {
    font-size: 29px;

    margin-bottom: 9px;
}

.agent b {
    display: block;

    margin-bottom: 5px;
}

.agent span {
    color: #64748b;

    font-size: 12px;

    line-height: 1.5;
}

.process {
    display: grid;

    grid-template-columns:
        repeat(3, 1fr);

    gap: 13px;
}

.step {
    background: #020617;

    border:
        1px solid
        #1e293b;

    border-radius: 13px;

    padding: 18px;
}

.step p {
    color: #64748b;

    font-size: 13px;

    line-height: 1.5;
}

.num {
    color: #60a5fa;

    font-weight: 900;

    margin-bottom: 8px;
}

.button {
    width: 100%;

    border: none;

    border-radius: 14px;

    padding: 17px;

    background:
        linear-gradient(
            135deg,
            #2563eb,
            #4f46e5
        );

    color: white;

    font-size: 15px;

    font-weight: 900;

    cursor: pointer;

    box-shadow:
        0 10px 30px
        rgba(37,99,235,.25);
}

.button:hover {
    filter: brightness(1.12);
}

.error {
    margin-top: 18px;

    padding: 15px;

    border-radius: 12px;

    background: #450a0a;

    border: 1px solid #7f1d1d;

    color: #fca5a5;
}

.footer {
    text-align: center;

    color: #475569;

    font-size: 12px;

    margin-top: 30px;
}

.result {
    margin-top: 25px;
}

.recommendation {
    text-align: center;

    padding: 25px;

    border-radius: 18px;

    margin-bottom: 18px;
}

.recommendation.hire {
    background: #052e16;

    border: 1px solid #166534;

    color: #86efac;
}

.recommendation.maybe {
    background: #422006;

    border: 1px solid #92400e;

    color: #fde68a;
}

.recommendation.reject {
    background: #450a0a;

    border: 1px solid #991b1b;

    color: #fca5a5;
}

.score {
    font-size: 42px;

    font-weight: 900;
}

.agent-result {
    padding: 16px;

    background: #020617;

    border:
        1px solid
        #1e293b;

    border-radius: 13px;

    margin-top: 12px;
}

.agent-result small {
    color: #64748b;
}

.evidence {
    padding: 12px;

    margin-top: 8px;

    border-left:
        3px solid
        #3b82f6;

    background: #0f172a;

    border-radius: 7px;

    color: #cbd5e1;

    font-size: 13px;
}

.download {
    display: inline-block;

    margin-top: 12px;

    padding: 11px 16px;

    border-radius: 10px;

    background: #1e293b;

    color: white;

    text-decoration: none;

    font-weight: 700;
}

@media(max-width: 800px) {

    .grid,
    .process {
        grid-template-columns: 1fr;
    }

    .agents {
        grid-template-columns:
            1fr 1fr;
    }
}

@media(max-width: 500px) {

    .agents {
        grid-template-columns: 1fr;
    }
}

</style>

</head>

<body>

<div class="container">

<header class="hero">

<div class="badge">
● DUMMY LLM • SYSTEM READY
</div>

<h1>
Candidate Review System
</h1>

<p>
Multi-agent candidate evaluation combining
technical, HR, hiring-manager and skeptical
analysis into an evidence-based recommendation.
</p>

</header>


<form
    method="POST"
    action="/evaluate"
    enctype="multipart/form-data"
>


<div class="card">

<div class="title">
Candidate Information
</div>

<div class="desc">
Enter the candidate and role details.
</div>

<div class="grid">

<div class="field">

<label>
Candidate Name
</label>

<input
    type="text"
    name="candidate_name"
    placeholder="e.g. Rahul Sharma"
>

</div>


<div class="field">

<label>
Job Description
</label>

<input
    type="text"
    name="job_description"
    placeholder="e.g. AI / ML Engineer"
>

</div>

</div>


<label class="sample">

<input
    type="checkbox"
    name="use_sample"
>

Use built-in sample candidate

</label>

</div>


<div class="card">

<div class="title">
Resume
</div>

<div class="desc">
Paste a resume or upload a text file.
</div>

<textarea
    name="resume_text"
    placeholder="Paste candidate resume here..."
></textarea>

<div class="file-box">

<label>
Upload Resume
</label>

<input
    type="file"
    name="resume_file"
    accept=".txt,.md,.csv,.json"
>

</div>

</div>


<div class="card">

<div class="title">
Interview Transcript
</div>

<div class="desc">
Paste the interview transcript or upload a text file.
</div>

<textarea
    name="transcript_text"
    placeholder="Paste interview transcript here..."
></textarea>

<div class="file-box">

<label>
Upload Transcript
</label>

<input
    type="file"
    name="transcript_file"
    accept=".txt,.md,.csv,.json"
>

</div>

</div>


<div class="card">

<div class="title">
Multi-Agent Review
</div>

<div class="desc">
Four specialized agents review the candidate
before a three-round debate.
</div>

<div class="agents">

<div class="agent">

<div class="icon">
💻
</div>

<b>
Technical
</b>

<span>
Skills, experience and problem solving
</span>

</div>


<div class="agent">

<div class="icon">
🤝
</div>

<b>
HR
</b>

<span>
Communication, behavior and culture fit
</span>

</div>


<div class="agent">

<div class="icon">
🎯
</div>

<b>
Hiring Manager
</b>

<span>
Role fit, impact and delivery
</span>

</div>


<div class="agent">

<div class="icon">
🔎
</div>

<b>
Skeptic
</b>

<span>
Risks, contradictions and missing evidence
</span>

</div>

</div>

</div>


<div class="card">

<div class="title">
Review Process
</div>

<div class="process">

<div class="step">

<div class="num">
01
</div>

<b>
Independent Review
</b>

<p>
Each agent evaluates the candidate independently.
</p>

</div>


<div class="step">

<div class="num">
02
</div>

<b>
Three-Round Debate
</b>

<p>
Agents challenge and refine the conclusions.
</p>

</div>


<div class="step">

<div class="num">
03
</div>

<b>
Final Decision
</b>

<p>
Evidence is aggregated into Hire, Maybe or Reject.
</p>

</div>

</div>

</div>


<button
    class="button"
    type="submit"
>
Run Candidate Review →
</button>


</form>


{% if error %}

<div class="error">
{{ error }}
</div>

{% endif %}


<div class="footer">

Candidate Review System • Dummy LLM • Multi-Agent Evaluation

</div>

</div>

</body>

</html>
"""


# ============================================================
# REPORT PAGE
# ============================================================

REPORT_PAGE = r"""
<!DOCTYPE html>

<html>

<head>

<meta charset="UTF-8">

<meta
    name="viewport"
    content="width=device-width, initial-scale=1.0"
>

<title>
Candidate Review Report
</title>

<style>

body {
    margin: 0;

    background: #020617;

    color: #f8fafc;

    font-family:
        Arial,
        sans-serif;
}

.container {
    max-width: 1050px;

    margin: auto;

    padding: 40px 20px;
}

.card {
    background: #0f172a;

    border:
        1px solid
        #1e293b;

    border-radius: 18px;

    padding: 25px;

    margin-bottom: 18px;
}

h1 {
    margin-bottom: 5px;
}

.muted {
    color: #94a3b8;
}

.recommendation {
    text-align: center;

    padding: 28px;

    border-radius: 18px;

    margin: 20px 0;

    font-weight: 900;

    text-transform: uppercase;
}

.hire {
    background: #052e16;

    color: #86efac;
}

.maybe {
    background: #422006;

    color: #fde68a;
}

.reject {
    background: #450a0a;

    color: #fca5a5;
}

.score {
    font-size: 45px;
}

.item {
    padding: 12px;

    margin-top: 10px;

    background: #020617;

    border-radius: 10px;

    border:
        1px solid
        #1e293b;
}

.evidence {
    padding: 12px;

    margin-top: 9px;

    background: #020617;

    border-left:
        3px solid
        #3b82f6;

    color: #cbd5e1;

    border-radius: 7px;
}

a {
    color: #93c5fd;
}

.button {
    display: inline-block;

    padding: 12px 18px;

    border-radius: 10px;

    background: #2563eb;

    color: white;

    text-decoration: none;

    font-weight: 800;

    margin-right: 8px;
}

</style>

</head>

<body>

<div class="container">

<h1>
Candidate Review Report
</h1>

<p class="muted">

Candidate:
<strong>
{{ candidate_name }}
</strong>

{% if job_description %}

<br>

Role:
<strong>
{{ job_description }}
</strong>

{% endif %}

</p>


<div class="recommendation {{ report.recommendation }}">

<div>
Final Recommendation
</div>

<div class="score">
{{ report.recommendation | upper }}
</div>

<div>
Confidence:
{{ (report.confidence * 100) | round(1) }}%
</div>

{% if report.metadata.final_score %}

<div>
Score:
{{ report.metadata.final_score }}
/ 100
</div>

{% endif %}

</div>


<div class="card">

<h2>
Strengths
</h2>

{% for strength in report.strengths %}

<div class="item">
{{ strength }}
</div>

{% endfor %}

</div>


<div class="card">

<h2>
Decisive Evidence
</h2>

{% for evidence in report.decisive_evidence %}

<div class="evidence">

<strong>
{{ evidence.source }}
</strong>

<br>

{{ evidence.quote }}

</div>

{% endfor %}

</div>


<div class="card">

<h2>
Agent Opinions
</h2>

{% for agent in report.agent_summaries %}

<div class="item">

<h3>
{{ agent.role }}
</h3>

<p>

Decision:
<strong>
{{ agent.decision | upper }}
</strong>

<br>

Score:
<strong>
{{ agent.score }}
</strong>

<br>

Confidence:
<strong>
{{ (agent.confidence * 100) | round(1) }}%
</strong>

</p>

<p>
{{ agent.rationale }}
</p>

{% for evidence in agent.evidences %}

<div class="evidence">

{{ evidence.quote }}

<br>

<small>
{{ evidence.source }}
</small>

</div>

{% endfor %}

</div>

{% endfor %}

</div>


<div class="card">

<h2>
Debate History
</h2>

{% for debate in report.debate_history %}

<div class="item">

<strong>
Round {{ debate.round_index + 1 }}
</strong>

<br>

{{ debate.agent_id }}

→

{{ debate.replies_to }}

<p>
{{ debate.text }}
</p>

</div>

{% endfor %}

</div>


{% if report.unresolved_disagreements %}

<div class="card">

<h2>
Unresolved Disagreements
</h2>

{% for disagreement in report.unresolved_disagreements %}

<div class="item">
{{ disagreement }}
</div>

{% endfor %}

</div>

{% endif %}


<div class="card">

<h2>
Reasoning
</h2>

{% for reason in report.reasoning %}

<div class="item">
{{ reason }}
</div>

{% endfor %}

</div>


<div class="card">

<a
    class="button"
    href="/"
>
← New Review
</a>

<a
    class="button"
    href="/outputs/{{ output_file }}"
>
Download JSON Report
</a>

</div>


</div>

</body>

</html>
"""


# ============================================================
# UPLOAD / OUTPUT DIRECTORIES
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "uploads",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True,
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# ============================================================
# FILE HANDLING
# ============================================================

def read_uploaded_file(file_obj):

    if (
        not file_obj
        or not file_obj.filename
    ):
        return ""

    filename = secure_filename(
        file_obj.filename
    )

    allowed = {
        ".txt",
        ".md",
        ".csv",
        ".json",
    }

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension not in allowed:

        raise ValueError(
            "Only TXT, MD, CSV, and JSON "
            "files are supported."
        )

    path = os.path.join(
        UPLOAD_DIR,
        filename,
    )

    file_obj.save(path)

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as file:

        return file.read()


def get_text(form_name):

    file_obj = request.files.get(
        f"{form_name}_file"
    )

    if (
        file_obj
        and file_obj.filename
    ):

        return read_uploaded_file(
            file_obj
        )

    return request.form.get(
        f"{form_name}_text",
        "",
    ).strip()


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template_string(
        HTML_PAGE
    )


@app.route(
    "/evaluate",
    methods=["POST"],
)
def run_evaluation():

    candidate_name = (
        request.form.get(
            "candidate_name",
            "Candidate",
        ).strip()
        or "Candidate"
    )

    job_description = (
        request.form.get(
            "job_description",
            "",
        ).strip()
    )

    use_sample = (
        request.form.get(
            "use_sample"
        )
        == "on"
    )

    try:

        resume_text = get_text(
            "resume"
        )

        transcript_text = get_text(
            "transcript"
        )

        if use_sample:

            if not resume_text:
                resume_text = SAMPLE_RESUME

            if not transcript_text:
                transcript_text = SAMPLE_TRANSCRIPT

        if not resume_text:

            return render_template_string(
                HTML_PAGE,
                error=(
                    "Please provide a resume "
                    "or select 'Use sample candidate'."
                ),
            ), 400

        if not transcript_text:

            return render_template_string(
                HTML_PAGE,
                error=(
                    "Please provide an interview "
                    "transcript or select 'Use sample candidate'."
                ),
            ), 400

        # Force dummy LLM
        report = evaluate(
            resume_text,
            transcript_text,
            provider="dummy",
        )

        timestamp = int(
            time.time()
        )

        safe_name = secure_filename(
            candidate_name
        )

        if not safe_name:
            safe_name = "candidate"

        filename = (
            f"report_{safe_name}_"
            f"{timestamp}.json"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            filename,
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                report,
                file,
                indent=2,
                ensure_ascii=False,
            )

        return render_template_string(
            REPORT_PAGE,
            report=report,
            candidate_name=candidate_name,
            job_description=job_description,
            output_file=filename,
        )

    except Exception as exception:

        return render_template_string(
            HTML_PAGE,
            error=(
                f"Evaluation failed: "
                f"{exception}"
            ),
        ), 500


# ============================================================
# JSON API
# ============================================================

@app.route(
    "/api/evaluate",
    methods=["POST"],
)
def api_evaluate():

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    resume = data.get(
        "resume",
        "",
    )

    transcript = data.get(
        "transcript",
        "",
    )

    if not resume or not transcript:

        return jsonify(
            {
                "error":
                    "resume and transcript are required"
            }
        ), 400

    try:

        report = evaluate(
            resume,
            transcript,
            provider="dummy",
        )

        return jsonify(report)

    except Exception as exception:

        return jsonify(
            {
                "error": str(exception)
            }
        ), 500


# ============================================================
# DOWNLOAD JSON REPORT
# ============================================================

@app.route(
    "/outputs/<path:filename>"
)
def download_output(filename):

    safe_filename = secure_filename(
        filename
    )

    path = os.path.join(
        OUTPUT_DIR,
        safe_filename,
    )

    if not os.path.isfile(path):

        return jsonify(
            {
                "error":
                    "Report not found"
            }
        ), 404

    return send_file(
        path,
        as_attachment=True,
        download_name=safe_filename,
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health")
def health():

    return jsonify(
        {
            "status": "ok",
            "service":
                "candidate-review-system",
            "llm": "dummy",
        }
    )


# ============================================================
# CLI
# ============================================================

def pretty_print_report(
    report_dict: Dict[str, Any]
):

    print(
        json.dumps(
            report_dict,
            indent=2,
        )
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--real-llm",
        action="store_true",
        help=(
            "Use real OpenAI LLM "
            "(requires OPENAI_API_KEY)"
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write evaluation JSON to this path",
    )

    parser.add_argument(
        "--web",
        action="store_true",
        help="Start the Flask web application",
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Web mode
    # --------------------------------------------------------

    if args.web:

        port = int(
            os.environ.get(
                "PORT",
                8080,
            )
        )

        app.run(
            host="0.0.0.0",
            port=port,
            debug=False,
        )

        return

    # --------------------------------------------------------
    # CLI evaluation
    # --------------------------------------------------------

    provider = (
        "openai"
        if args.real_llm
        else "dummy"
    )

    if (
        args.real_llm
        and not os.getenv(
            "OPENAI_API_KEY"
        )
    ):

        print(
            "WARNING: --real-llm specified "
            "but OPENAI_API_KEY is not set. "
            "Using dummy provider."
        )

        provider = "dummy"

    print(
        "Running example evaluation..."
    )

    report = evaluate(
        SAMPLE_RESUME,
        SAMPLE_TRANSCRIPT,
        provider=provider,
    )

    pretty_print_report(
        report
    )

    if args.output:

        try:

            with open(
                args.output,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    report,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

            print(
                f"\nWrote report to "
                f"{args.output}"
            )

        except Exception as exception:

            print(
                f"Failed to write output: "
                f"{exception}"
            )


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    # When started normally:
    #
    # python candidate_review.py
    #
    # it starts the website.
    #
    # CLI evaluation can be run with:
    #
    # python candidate_review.py --output report.json
    #

    port = int(
        os.environ.get(
            "PORT",
            8080,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False,
    )
```
