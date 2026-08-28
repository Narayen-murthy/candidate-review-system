#!/usr/bin/env python3
"""
Single-file Candidate Review prototype for hackathon.
- Build candidate profile from resume + transcript text
- Run 4 independent agents (Technical, HR, HiringManager, Skeptic)
- Run a single-round debate where each agent must reply to at least one other agent
- Produce a reasoned final report (not simple averaging)

Usage:
- As a script:
    python candidate_review.py
  This runs an example with sample texts and prints the JSON final report.

- As an import:
    from candidate_review import evaluate
    report = evaluate(resume_text, transcript_text)
"""

import re
import json
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from time import sleep

try:
    import openai
except Exception:
    openai = None

# -----------------------
# Data classes
# -----------------------
@dataclass
class Evidence:
    quote: str
    source: str  # e.g., "resume:line 12" or "transcript:speaker@00:15"
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
    decision: str  # "hire","maybe","reject"
    score: float  # 0-100
    confidence: float  # 0.0-1.0
    rationale: str
    evidences: List[Evidence] = field(default_factory=list)

@dataclass
class DebateReply:
    agent_id: str
    replies_to: str
    text: str
    updated_opinion: Optional[AgentOpinion] = None

@dataclass
class FinalReport:
    recommendation: str
    confidence: float
    decisive_evidence: List[Evidence]
    strengths: List[str]
    concerns: List[str]
    agent_summaries: List[AgentOpinion]
    unresolved_disagreements: List[str]

# -----------------------
# Profile builder
# -----------------------
def parse_basic_contact(resume_text: str):
    name = None
    email = None
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    if lines:
        first = lines[0]
        if 2 <= len(first.split()) <= 4 and re.match(r"^[A-Z][a-z]", first):
            name = first.strip()
    m2 = re.search(r"([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", resume_text)
    if m2:
        email = m2.group(1)
    return name, email

def extract_skills(resume_text: str) -> List[str]:
    skills = []
    m = re.search(r"Skills?:\s*(.+)", resume_text, re.IGNORECASE)
    if m:
        raw = m.group(1)
        parts = re.split(r"[,|;]\s*", raw)
        skills = [p.strip() for p in parts if p.strip()]
    else:
        tokens = ["Python", "Java", "C++", "Go", "SQL", "React", "Node", "Docker", "Kubernetes", "AWS", "GCP"]
        for t in tokens:
            if re.search(r"\b" + re.escape(t) + r"\b", resume_text, re.IGNORECASE):
                skills.append(t)
    seen = set()
    res = []
    for s in skills:
        if s.lower() not in seen:
            seen.add(s.lower())
            res.append(s)
    return res

def extract_experiences(resume_text: str) -> List[str]:
    exps = []
    for line in resume_text.splitlines():
        if re.search(r"\b(Engineer|Developer|Manager|Intern|Consultant|SWE)\b", line, re.IGNORECASE):
            if len(line.strip()) > 5:
                exps.append(line.strip())
    return exps

def extract_claims(resume_text: str, transcript_text: str) -> List[Claim]:
    claims: List[Claim] = []
    for m in re.finditer(r"([^.]*\b(?:years|led|improved|built|designed|implemented|managed)\b[^.]*\.)", resume_text, re.IGNORECASE):
        sentence = m.group(1).strip()
        claims.append(Claim(text=sentence, evidences=[Evidence(quote=sentence, source="resume")]))
    for line in transcript_text.splitlines():
        if re.search(r"\b(I led|I built|I designed|I implemented|I managed|I have \d+ years|we built|my team)\b", line, re.IGNORECASE):
            s = line.strip()
            if s:
                claims.append(Claim(text=s, evidences=[Evidence(quote=s, source="transcript")]))
    return claims

def build_candidate_profile(resume_text: str, transcript_text: str) -> CandidateProfile:
    name, email = parse_basic_contact(resume_text)
    skills = extract_skills(resume_text)
    experiences = extract_experiences(resume_text)
    claims = extract_claims(resume_text, transcript_text)
    education = []
    m = re.search(r"(B\.S\.|Bachelors|M\.S\.|Masters|Ph\.D|Bachelor of|Master of)[^\n]*", resume_text, re.IGNORECASE)
    if m:
        education.append(m.group(0).strip())
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

# -----------------------
# LLM client using OpenAI (with safe JSON parsing) and fallback to dummy
# -----------------------
class LLMClient:
    def __init__(self, provider: str = "openai"):
        # provider may be 'openai' or 'dummy'
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gpt-4")
        if self.api_key and openai is not None:
            openai.api_key = self.api_key
            self.provider = 'openai'
        else:
            self.provider = 'dummy'

    def _call_openai(self, prompt: str) -> str:
        # call ChatCompletion with retries
        last_exc = None
        for attempt in range(3):
            try:
                resp = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=1000,
                )
                return resp['choices'][0]['message']['content']
            except Exception as e:
                last_exc = e
                sleep(1 + attempt)
        raise last_exc

    def _extract_json(self, text: str) -> Any:
        # Try to extract the first JSON object from text robustly
        # 1) If the text is JSON already, load it
        try:
            return json.loads(text)
        except Exception:
            pass
        # 2) Try to find the first {...} block
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            candidate = m.group(0)
            try:
                return json.loads(candidate)
            except Exception:
                # attempt to fix common issues: replace single quotes with double quotes
                candidate2 = candidate.replace("'", '"')
                try:
                    return json.loads(candidate2)
                except Exception:
                    pass
        # 3) Could not parse JSON
        raise ValueError("Could not parse JSON from model output")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        # If OpenAI provider available, call it and parse JSON output.
        if self.provider == 'openai':
            # ask the model to produce JSON - rely on our prompt templates to request JSON
            txt = self._call_openai(prompt)
            try:
                parsed = self._extract_json(txt)
                if isinstance(parsed, dict):
                    return parsed
                # If model returned something else, wrap it
                return {"decision": "maybe", "score": 60, "confidence": 0.5, "rationale": str(parsed), "evidences": []}
            except Exception as e:
                # fallback: return a cautious default
                return {"decision": "maybe", "score": 60, "confidence": 0.5, "rationale": f"Could not parse model JSON: {e}", "evidences": []}

        # Dummy behaviour (previous simulated logic)
        low = prompt.lower()
        is_debate = ("respond by addressing" in low) or ("other agents' opinions" in low) or ("reply to" in low)
        if is_debate:
            # Attempt to extract role like before
            m = re.search(r"you are\s+([A-Za-z ]+?)\s+agent", prompt, re.IGNORECASE)
            role = m.group(1).strip().lower() if m else "unknown"
            if "technical" in role:
                return {
                    "reply_to": "agent_skeptic",
                    "text": "Technical: I acknowledge verification concerns, but the transcript/resume shows relevant tech and measurable claims; recommend targeted technical follow-up in interview.",
                    "updated_opinion": {
                        "decision": "maybe",
                        "score": 65,
                        "confidence": 0.65,
                        "rationale": "Technical reduces score slightly because of missing design details, but retains partial confidence due to relevant skills.",
                        "evidences": [
                            {"quote": "I built a distributed logging service in Python that handled 100k events per second.", "source": "transcript:00:15"},
                            {"quote": "Skills: Python, Docker, Kubernetes, SQL, React", "source": "resume"}
                        ]
                    }
                }
            if "hr" in role:
                return {
                    "reply_to": "agent_skeptic",
                    "text": "HR: Leadership and communication are clear (led a team, performance improvements); these are positive cultural signals even if artifacts are requested.",
                    "updated_opinion": {
                        "decision": "hire",
                        "score": 77,
                        "confidence": 0.7,
                        "rationale": "Strong communication and leadership signals in transcript and resume.",
                        "evidences": [
                            {"quote": "I led a team of 3 engineers.", "source": "transcript:00:15"},
                            {"quote": "Led backend services, improved performance 4x.", "source": "resume"}
                        ]
                    }
                }
            if "hiring" in role:
                return {
                    "reply_to": "agent_tech",
                    "text": "HiringManager: The candidate looks like a plausible fit; recommend proceeding to a technical interview to validate the claims.",
                    "updated_opinion": {
                        "decision": "maybe",
                        "score": 70,
                        "confidence": 0.65,
                        "rationale": "Role-fit plausible; recommend follow-up technical validation in interview.",
                        "evidences": [
                            {"quote": "Senior Software Engineer at Acme Corp (2019-2024)", "source": "resume"},
                            {"quote": "I have 6 years of backend experience.", "source": "transcript"}
                        ]
                    }
                }
            if "skeptic" in role:
                return {
                    "reply_to": "agent_hm",
                    "text": "Skeptic: Throughput/improvement claims lack corroborating evidence and could be exaggerated; request artifacts, logs, or specifics before hiring.",
                }
            return {"reply_to": "agent_hm", "text": "I question unverifiable claims; please provide more detail or evidence.", "updated_opinion": {"decision": "maybe", "score": 60, "confidence": 0.5, "rationale": "Generic debate adjustment.", "evidences": [{"quote": "I have 6 years of backend experience.", "source": "transcript"}]}}

        # initial opinions (dummy)
        if "technical" in low:
            return {"decision": "maybe", "score": 70, "confidence": 0.7, "rationale": "Candidate shows technical claims ...", "evidences": [{"quote": "I built a distributed logging service in Python that handled 100k events per second.", "source": "transcript:00:15"}]}
        if "hr" in low or "culture" in low:
            return {"decision": "hire", "score": 75, "confidence": 0.65, "rationale": "Communication appears clear...", "evidences": [{"quote": "I led a team of 3 engineers.", "source": "transcript:00:15"}]}
        if "hiringmanager" in low or "hiring manager" in low:
            return {"decision": "maybe", "score": 68, "confidence": 0.6, "rationale": "Role-fit plausible...", "evidences": [{"quote": "Senior Software Engineer at Acme Corp (2019-2024)", "source": "resume"}]}
        if "skeptic" in low:
            return {"decision": "reject", "score": 25, "confidence": 0.8, "rationale": "Strong claims without artifacts...", "evidences": [{"quote": "I built a distributed logging service...", "source": "transcript"}]}
        return {"decision": "maybe", "score": 60, "confidence": 0.5, "rationale": "Generic simulated opinion.", "evidences": [{"quote": "I have 6 years of backend experience.", "source": "transcript"}]}
