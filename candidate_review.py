#!/usr/bin/env python3
"""
Single-file Candidate Review prototype for hackathon.
- Build candidate profile from resume + transcript text
- Run 4 independent agents (Technical, HR, HiringManager, Skeptic)
- Run a multi-round debate where agents must reply to others
- Produce a reasoned final report (evidence-weighted, not simple averaging)

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
    source: str  # e.g., "resume:L12" or "transcript:line_5"
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
    initial_opinions: List[AgentOpinion] = field(default_factory=list)
    debate_history: List[DebateReply] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)

# -----------------------
# Profile builder (now records evidence anchors)
# -----------------------
def parse_basic_contact(resume_text: str):
    name = None
    email = None
    lines = [l.rstrip() for l in resume_text.splitlines() if l.strip()]
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
    for idx, line in enumerate(resume_text.splitlines(), start=1):
        if re.search(r"\b(Engineer|Developer|Manager|Intern|Consultant|SWE)\b", line, re.IGNORECASE):
            if len(line.strip()) > 5:
                exps.append(line.strip())
    return exps


def extract_claims(resume_text: str, transcript_text: str) -> List[Claim]:
    claims: List[Claim] = []
    # Resume claims with line anchors
    for i, line in enumerate(resume_text.splitlines(), start=1):
        for m in re.finditer(r"([^.]*\b(?:years|led|improved|built|designed|implemented|managed)\b[^.]*\.)", line, re.IGNORECASE):
            sentence = m.group(1).strip()
            claims.append(Claim(text=sentence, evidences=[Evidence(quote=sentence, source=f"resume:L{i}")]))
    # Transcript claims with line anchors and speaker
    for i, line in enumerate(transcript_text.splitlines(), start=1):
        if re.search(r"\b(I led|I built|I designed|I implemented|I managed|I have \d+ years|we built|my team)\b", line, re.IGNORECASE):
            s = line.strip()
            if s:
                # prefix speaker if present (e.g., "Candidate:")
                speaker = None
                sp = re.match(r"^(\w+):\s*(.*)$", s)
                if sp:
                    speaker = sp.group(1)
                    quote = sp.group(2)
                else:
                    quote = s
                claims.append(Claim(text=quote, evidences=[Evidence(quote=quote, source=f"transcript:{speaker or 'line'}@L{i}")]))
    return claims


def build_candidate_profile(resume_text: str, transcript_text: str) -> CandidateProfile:
    name, email = parse_basic_contact(resume_text)
    skills = extract_skills(resume_text)
    experiences = extract_experiences(resume_text)
    claims = extract_claims(resume_text, transcript_text)
    education = []
    for i, line in enumerate(resume_text.splitlines(), start=1):
        m = re.search(r"(B\.S\.|Bachelors|M\.S\.|Masters|Ph\.D|Bachelor of|Master of)(.*)", line, re.IGNORECASE)
        if m:
            education.append(line.strip())
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
# LLM client (dummy + openai compatibility)
# -----------------------
class LLMClient:
    def __init__(self, provider: str = "openai"):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.client = None
        if self.api_key and openai is not None:
            try:
                if hasattr(openai, "OpenAI"):
                    self.client = openai.OpenAI(api_key=self.api_key)
                else:
                    openai.api_key = self.api_key
                self.provider = 'openai'
            except Exception:
                self.provider = 'dummy'
        else:
            self.provider = 'dummy'

    def _call_openai(self, prompt: str) -> str:
        last_exc = None
        for attempt in range(3):
            try:
                if self.provider != 'openai':
                    raise RuntimeError("OpenAI provider not configured")
                if hasattr(openai, 'ChatCompletion'):
                    resp = openai.ChatCompletion.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=1000,
                    )
                    return resp['choices'][0]['message']['content']
                if hasattr(openai, 'OpenAI'):
                    client = self.client or openai.OpenAI(api_key=self.api_key)
                    resp = client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0,
                        max_tokens=1000,
                    )
                    try:
                        return resp['choices'][0]['message']['content']
                    except Exception:
                        return resp.choices[0].message.content
                raise RuntimeError("No compatible OpenAI API found")
            except Exception as e:
                last_exc = e
                sleep(1 + attempt)
        raise last_exc

    def _extract_json(self, text: str) -> Any:
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            candidate = m.group(0)
            try:
                return json.loads(candidate)
            except Exception:
                candidate2 = candidate.replace("'", '"')
                try:
                    return json.loads(candidate2)
                except Exception:
                    pass
        raise ValueError("Could not parse JSON from model output")

    def _validate_opinion(self, parsed: Any) -> Dict[str, Any]:
        # Ensure parsed is a dict with required keys; if not, return a cautious fallback
        if not isinstance(parsed, dict):
            return {"decision": "maybe", "score": 60, "confidence": 0.5, "rationale": "Invalid agent output", "evidences": []}
        required = {"decision", "score", "confidence", "rationale", "evidences"}
        if not required.issubset(set(parsed.keys())):
            # try to coerce
            parsed2 = {k: parsed.get(k) for k in list(required & set(parsed.keys()))}
            parsed2.setdefault('decision', 'maybe')
            parsed2.setdefault('score', 60)
            parsed2.setdefault('confidence', 0.5)
            parsed2.setdefault('rationale', 'Incomplete output, using conservative defaults')
            parsed2.setdefault('evidences', [])
            return parsed2
        return parsed

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Dict[str, Any]:
        if self.provider == 'openai':
            txt = self._call_openai(prompt)
            try:
                parsed = self._extract_json(txt)
                parsed = self._validate_opinion(parsed)
                return parsed
            except Exception as e:
                return {"decision": "maybe", "score": 60, "confidence": 0.5, "rationale": f"Could not parse model JSON: {e}", "evidences": []}

        # Dummy behaviour: deterministic per-role outputs and debate replies
        low = prompt.lower()
        is_debate = ("respond by addressing" in low) or ("other agents' opinions" in low) or ("reply to" in low)
        if is_debate:
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
                            {"quote": "I built a distributed logging service in Python that handled 100k events per second.", "source": "transcript:L2"},
                            {"quote": "Skills: Python, Docker, Kubernetes, SQL, React", "source": "resume:L3"}
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
                            {"quote": "I led a team of 3 engineers.", "source": "transcript:L2"},
                            {"quote": "Led backend services, improved performance 4x.", "source": "resume:L6"}
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
                            {"quote": "Senior Software Engineer at Acme Corp (2019-2024)", "source": "resume:L6"},
                            {"quote": "I have 6 years of backend experience.", "source": "transcript:L3"}
                        ]
                    }
                }
            if "skeptic" in role:
                return {
                    "reply_to": "agent_hm",
                    "text": "Skeptic: Throughput/improvement claims lack corroborating evidence and could be exaggerated; request artifacts, logs, or specifics before hiring.",
                }
            return {"reply_to": "agent_hm", "text": "I question unverifiable claims; please provide more detail or evidence.", "updated_opinion": {"decision": "maybe", "score": 60, "confidence": 0.5, "rationale": "Generic debate adjustment.", "evidences": [{"quote": "I have 6 years of backend experience.", "source": "transcript:L3"}]}}

        # initial opinions (dummy)
        if "technical" in low:
            return {"decision": "maybe", "score": 70, "confidence": 0.7, "rationale": "Candidate shows technical claims but lacks detailed design or reproducibility artifacts.", "evidences": [{"quote": "I built a distributed logging service in Python that handled 100k events per second.", "source": "transcript:L2"}, {"quote": "Skills: Python, Docker, Kubernetes, SQL, React", "source": "resume:L3"}]}
        if "hr" in low or "culture" in low:
            return {"decision": "hire", "score": 75, "confidence": 0.65, "rationale": "Communication appears clear; candidate claims to have led a team and improved performance.", "evidences": [{"quote": "I led a team of 3 engineers.", "source": "transcript:L2"}, {"quote": "Led backend services, improved performance 4x.", "source": "resume:L6"}]}
        if "hiringmanager" in low or "hiring manager" in low:
            return {"decision": "maybe", "score": 68, "confidence": 0.6, "rationale": "Role-fit plausible given backend experience but need to verify scale and depth for the specific role.", "evidences": [{"quote": "Senior Software Engineer at Acme Corp (2019-2024)", "source": "resume:L6"}, {"quote": "I have 6 years of backend experience.", "source": "transcript:L3"}]}
        if "skeptic" in low:
            return {"decision": "reject", "score": 25, "confidence": 0.8, "rationale": "There are strong claims of throughput and improvements but no corroborating details or artifacts; risk of exaggeration.", "evidences": [{"quote": "I built a distributed logging service in Python that handled 100k events per second.", "source": "transcript:L2"}, {"quote": "improved performance 4x", "source": "resume:L6"}]}
        return {"decision": "maybe", "score": 60, "confidence": 0.5, "rationale": "Generic simulated opinion.", "evidences": [{"quote": "I have 6 years of backend experience.", "source": "transcript:L3"}]}

# -----------------------
# Agent implementation
# -----------------------
ROLE_GUIDANCE = {
    "Technical": "- Assess technical skills, depth, relevant projects, measurable impact.",
    "HR": "- Assess communication, teamwork, honesty, cultural fit.",
    "HiringManager": "- Assess role fit, potential, ability to deliver on role requirements.",
    "Skeptic": "- Try to find contradictions, exaggerations, missing details, or red flags."
}

PROMPT_TEMPLATE = """
You are the {role} agent. Evaluate the candidate based ONLY on the provided CandidateProfile and excerpts.
Rules:
1) Do NOT use external knowledge or hallucinate — only cite verbatim text from the provided profile/resume/transcript.
2) For every claim you make, include at least one Evidence object with fields: quote and source (e.g., "resume:L12" or "transcript:speaker@00:15").
3) Output EXACTLY a JSON-like object with keys:
   decision (one of "hire","maybe","reject"),
   score (0-100),
   confidence (0.0-1.0),
   rationale (string),
   evidences (list of {{"quote": "...", "source": "..."}}).
CandidateProfile:
{profile_json}
Questions to consider for your role:
{role_guidance}
"""

class Agent:
    def __init__(self, agent_id: str, role: str, llm_client: LLMClient):
        self.agent_id = agent_id
        self.role = role
        self.llm = llm_client

    def build_prompt(self, profile_json: str):
        guidance = ROLE_GUIDANCE.get(self.role, "")
        safe_profile = profile_json.replace("{", "{{").replace("}", "}}")
        return PROMPT_TEMPLATE.format(role=self.role, profile_json=safe_profile, role_guidance=guidance)

    def run_initial_opinion(self, profile_json: str) -> AgentOpinion:
        prompt = self.build_prompt(profile_json)
        raw = self.llm.generate(prompt)
        parsed = raw if isinstance(raw, dict) else json.loads(raw)
        parsed = self.llm._validate_opinion(parsed)
        evidences = []
        for e in parsed.get("evidences", []):
            evidences.append(Evidence(quote=e.get("quote", ""), source=e.get("source", "")))
        return AgentOpinion(
            agent_id=self.agent_id,
            role=self.role,
            decision=parsed.get("decision", "maybe"),
            score=float(parsed.get("score", 0)),
            confidence=float(parsed.get("confidence", 0.0)),
            rationale=parsed.get("rationale", ""),
            evidences=evidences
        )

    def run_debate_response(self, profile_json: str, other_opinions: List[AgentOpinion]) -> DebateReply:
        others_summary = "\n".join([
            f"{o.agent_id} ({o.role}): decision={o.decision}, score={o.score}, evidences={[e.quote for e in o.evidences]}"
            for o in other_opinions
        ])
        debate_prompt = (
            f"You are {self.role} agent. Given the profile and the other agents' opinions:\n{others_summary}\n"
            f"Respond by addressing at least one other agent's point. Include any quote from the source if you update your opinion. "
            f"Output JSON-like with keys: reply_to (agent_id), text, updated_opinion (optional: decision,score,confidence,rationale,evidences).\n"
            f"Profile:\n{profile_json}"
        )
        raw = self.llm.generate(debate_prompt)
        parsed = raw if isinstance(raw, dict) else json.loads(raw)
        # parsed might be a debate reply dict with reply_to/text/updated_opinion
        reply_to = parsed.get("reply_to", "")
        text = parsed.get("text", "")
        updated_opinion = None
        if parsed.get("updated_opinion"):
            u = parsed["updated_opinion"]
            u = self.llm._validate_opinion(u)
            evids = []
            for e in u.get("evidences", []):
                evids.append(Evidence(quote=e.get("quote", ""), source=e.get("source", "")))
            updated_opinion = AgentOpinion(
                agent_id=self.agent_id,
                role=self.role,
                decision=u.get("decision", "maybe"),
                score=float(u.get("score", 0)),
                confidence=float(u.get("confidence", 0.0)),
                rationale=u.get("rationale", ""),
                evidences=evids
            )
        return DebateReply(agent_id=self.agent_id, replies_to=reply_to, text=text, updated_opinion=updated_opinion)

# -----------------------
# Orchestration
# -----------------------
def run_pipeline(resume_text: str, transcript_text: str) -> FinalReport:
    profile = build_candidate_profile(resume_text, transcript_text)
    profile_json = json.dumps(asdict(profile), indent=2)

    llm_client = LLMClient(provider="dummy")

    agent_defs = [
        ("agent_tech", "Technical"),
        ("agent_hr", "HR"),
        ("agent_hm", "HiringManager"),
        ("agent_skeptic", "Skeptic"),
    ]

    # Initial independent opinions
    initial_opinions: List[AgentOpinion] = []
    for aid, role in agent_defs:
        agent = Agent(aid, role, llm_client)
        op = agent.run_initial_opinion(profile_json)
        initial_opinions.append(op)

    # Start debate: multi-round (3 rounds) to encourage opinion changes
    opinions = [op for op in initial_opinions]
    debate_replies: List[DebateReply] = []
    rounds = 3
    for r in range(rounds):
        for aid, role in agent_defs:
            agent = Agent(aid, role, llm_client)
            others = [o for o in opinions if o.agent_id != aid]
            reply = agent.run_debate_response(profile_json, others)
            debate_replies.append(reply)
            if reply.updated_opinion:
                # replace opinion for that agent
                for idx, op in enumerate(opinions):
                    if op.agent_id == aid:
                        opinions[idx] = reply.updated_opinion
                        break

    # Aggregation: evidence-weighted scoring with role weights, produce reasoning
    role_weights = {"HiringManager": 1.4, "Technical": 1.2, "HR": 1.0, "Skeptic": 1.0}
    weighted_sum = 0.0
    weight_total = 0.0
    reasoning: List[str] = []
    # Gather top evidences
    evidence_scores: Dict[str, float] = {}
    for o in opinions:
        # evidence strength = min(1.0, 0.2 * len(e.evidences) + o.confidence)
        ev_count = max(0, len(o.evidences))
        evidence_strength = (o.confidence * 0.6) + (min(3, ev_count) / 3.0 * 0.4)
        role_w = role_weights.get(o.role, 1.0)
        weight = o.confidence * role_w * (1.0 + ev_count / 4.0)
        weighted_sum += o.score * weight
        weight_total += weight
        # attribute evidence contributions for reasoning
        for e in o.evidences:
            key = f"{e.quote} || {e.source}"
            evidence_scores[key] = evidence_scores.get(key, 0.0) + (o.score * o.confidence * role_w)
    final_score = weighted_sum / weight_total if weight_total else 60
    # Map to recommendation
    if final_score >= 75:
        recommendation = "hire"
    elif final_score >= 55:
        recommendation = "maybe"
    else:
        recommendation = "reject"
    # Confidence is normalized from confidences weighted by same weights
    conf_numer = sum([o.confidence * role_weights.get(o.role, 1.0) for o in opinions])
    conf_denom = sum([role_weights.get(o.role, 1.0) for o in opinions])
    final_confidence = (conf_numer / conf_denom) if conf_denom else 0.5

    # Build decisive evidence list (top 3 contributing evidence items)
    top_evidence = sorted(evidence_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    decisive_evidence = []
    for k, v in top_evidence:
        quote, source = k.split(" || ")
        decisive_evidence.append(Evidence(quote=quote, source=source))
        reasoning.append(f"Evidence '{quote}' from {source} contributed score weight {v:.1f}")

    # Skeptic override (if strong and well-supported)
    skeptic = next((o for o in opinions if o.role == "Skeptic"), None)
    hm = next((o for o in opinions if o.role == "HiringManager"), None)
    if skeptic and skeptic.score < 30 and skeptic.confidence > 0.75:
        # if hiring manager strongly favors, soften; else use skeptic decision
        if hm and hm.score >= 85:
            recommendation = "maybe"
            reasoning.append("Skeptic raised a high-confidence low score but HiringManager is strongly favorable; compromise to 'maybe'.")
        else:
            recommendation = "reject"
            reasoning.append("Skeptic raised a high-confidence low score and no HiringManager override; final decision set to 'reject'.")

    # Collect unresolved disagreements
    decisions = set([o.decision for o in opinions])
    unresolved = []
    if len(decisions) > 1:
        unresolved = [f"{o.agent_id}({o.role}) => {o.decision} (score={o.score}, conf={o.confidence})" for o in opinions]

    strengths = [f"Skill: {s}" for s in profile.skills][:10]
    concerns = []
    for e in decisive_evidence:
        lowq = e.quote.lower()
        if any(w in lowq for w in ["no ", "not ", "lack", "unverified", "risk"]):
            concerns.append(e.quote)
    if not concerns:
        concerns = [e.quote for e in decisive_evidence][:5]

    report = FinalReport(
        recommendation=recommendation,
        confidence=round(float(final_confidence), 3),
        decisive_evidence=decisive_evidence,
        strengths=strengths,
        concerns=concerns,
        agent_summaries=opinions,
        unresolved_disagreements=unresolved,
        initial_opinions=initial_opinions,
        debate_history=debate_replies,
        reasoning=reasoning,
    )
    return report

# -----------------------
# Convenience wrapper + sample
# -----------------------
def evaluate(resume_text: str, transcript_text: str) -> Dict[str, Any]:
    report = run_pipeline(resume_text, transcript_text)
    def conv(obj):
        if isinstance(obj, list):
            return [conv(i) for i in obj]
        if hasattr(obj, "__dict__") or hasattr(obj, "__dataclass_fields__"):
            d = asdict(obj)
            for k, v in list(d.items()):
                d[k] = conv(v)
            return d
        return obj
    return conv(report)

SAMPLE_RESUME = """John Doe
Email: john.doe@example.com

Skills: Python, Docker, Kubernetes, SQL, React

Experience
- Senior Software Engineer at Acme Corp (2019-2024)
  Led backend services, improved performance 4x.
- Software Engineer at Beta Inc (2016-2019)
  Built ETL pipelines and APIs.

Education
B.S. Computer Science
"""

SAMPLE_TRANSCRIPT = """Interviewer: Tell me about a project you're proud of.
Candidate: I built a distributed logging service in Python that handled 100k events per second. I led a team of 3 engineers.
Candidate: I have 6 years of backend experience.
Candidate: I improved latency by 60% in my last role.
"""

def pretty_print_report(report_dict: Dict[str, Any]):
    print(json.dumps(report_dict, indent=2))


def main():
    print("Running example evaluation with sample resume + transcript...\n")
    res = evaluate(SAMPLE_RESUME, SAMPLE_TRANSCRIPT)
    pretty_print_report(res)
    print("\n--- Done ---\n")
    print("If you want to run with your own text, call evaluate(resume_text, transcript_text) from Python.")

if __name__ == "__main__":
    main()
