"""Send a resume to an LLM (Google Gemini) and return a validated analysis."""

import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

from google import genai
from google.genai import types

# Tried in order; the first model the API key can use wins.
DEFAULT_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.8-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.6-flash-lite",
    "gemini-3.5-flash",
]

CATEGORIES = [
    "Impact & metrics",
    "Clarity & structure",
    "Technical skills",
    "Projects",
    "Experience relevance",
    "ATS-friendliness",
]

MAX_INPUT_CHARS = 15_000


def build_prompt(resume: str, role: str = "", job_description: str = "") -> str:
    """Build the instruction prompt sent to the model."""
    resume = resume[:MAX_INPUT_CHARS]
    job_description = job_description[:MAX_INPUT_CHARS]

    context = ""
    if role:
        context += f"TARGET ROLE: {role}\n"
    if job_description:
        context += f"JOB DESCRIPTION:\n{job_description}\n\n"

    match_shape = (
        '{"score": <0-100>, "matched": [<JD keywords found in resume>], '
        '"missing": [<important JD keywords missing>], "summary": "<1-2 sentences>"}'
        if job_description
        else "null"
    )
    categories = ",\n    ".join(
        f'{{"name": "{c}", "score": <0-10>, "note": "<one short sentence>"}}' for c in CATEGORIES
    )

    return f"""You are an expert technical recruiter and ATS (applicant tracking system) specialist.
Review the resume below. Be honest, specific and constructive. Base every point on the
actual resume text and never invent experience.

{context}RESUME:
{resume}

Return JSON in exactly this shape:
{{
  "overall": <integer 0-100>,
  "headline": "<short verdict, max 8 words>",
  "summary": "<2 sentences on the resume overall>",
  "categories": [
    {categories}
  ],
  "strengths": [<3-4 specific strengths>],
  "issues": [{{"severity": "high|medium|low", "issue": "<what's wrong>", "fix": "<concrete fix>"}}],
  "rewrites": [{{"before": "<exact bullet from the resume>", "after": "<stronger version: action verb + what + result; use [X] for numbers the candidate must fill in>"}}],
  "match": {match_shape}
}}
Give 4-7 issues ordered by severity and 3-4 rewrites."""


def _analyze(api_key: str, resume: str, role: str = "", job_description: str = "",
                   models: list[str] | None = None, time_budget: float = 60) -> dict:
    """Call Gemini and return a cleaned-up analysis dict.

    Tries each model once (30 s limit per call) and gives up after `time_budget`
    seconds in total, so visitors never wait forever. If a model rejects JSON
    mode, it retries that model once with a plain request.
    """
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))
    prompt = build_prompt(resume, role, job_description)
    json_config = types.GenerateContentConfig(response_mime_type="application/json")
    started = time.monotonic()

    errors = []
    for model in models or DEFAULT_MODELS:
        for config in (json_config, None):
            if time.monotonic() - started > time_budget:
                errors.append("Stopped: time limit reached")
                raise RuntimeError("The AI service didn't respond.\n" + "\n".join(errors))
            try:
                response = client.models.generate_content(model=model, contents=prompt, config=config)
                return validate(parse_json(response.text), has_job=bool(job_description))
            except Exception as exc:
                message = f"{model} ({'json' if config else 'plain'}): {str(exc)[:200]}"
                print("Model failed ->", message)  # shows in Streamlit Cloud logs
                errors.append(message)
                if config is None or "INVALID_ARGUMENT" not in str(exc):
                    break  # only JSON-setting errors are worth a plain retry
    raise RuntimeError("The AI service didn't respond.\n" + "\n".join(errors))


_executor = ThreadPoolExecutor(max_workers=4)
HARD_LIMIT_SECONDS = 75


def analyze_resume(api_key: str, resume: str, role: str = "", job_description: str = "") -> dict:
    """Run the analysis with a hard time limit, so the page can never spin forever."""
    future = _executor.submit(_analyze, api_key, resume, role, job_description)
    try:
        return future.result(timeout=HARD_LIMIT_SECONDS)
    except FutureTimeout:
        print("Scan hit the hard time limit")
        raise RuntimeError("The AI service didn't respond.\nDEADLINE: no answer within "
                           f"{HARD_LIMIT_SECONDS} seconds (Google's servers may be overloaded).")


def parse_json(text: str) -> dict:
    """Parse model output, tolerating stray markdown fences."""
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


def validate(data: dict, has_job: bool) -> dict:
    """Clamp scores and fill missing fields so the UI never breaks."""

    def clamp(value, low, high):
        try:
            return max(low, min(high, int(round(float(value)))))
        except (TypeError, ValueError):
            return low

    by_name = {c.get("name"): c for c in data.get("categories", []) if isinstance(c, dict)}
    categories = [
        {
            "name": name,
            "score": clamp(by_name.get(name, {}).get("score"), 0, 10),
            "note": str(by_name.get(name, {}).get("note", "")),
        }
        for name in CATEGORIES
    ]

    issues = []
    for item in data.get("issues", []):
        if isinstance(item, dict) and item.get("issue"):
            severity = str(item.get("severity", "medium")).lower()
            issues.append({
                "severity": severity if severity in {"high", "medium", "low"} else "medium",
                "issue": str(item["issue"]),
                "fix": str(item.get("fix", "")),
            })
    order = {"high": 0, "medium": 1, "low": 2}
    issues.sort(key=lambda i: order[i["severity"]])

    rewrites = [
        {"before": str(r.get("before", "")), "after": str(r.get("after", ""))}
        for r in data.get("rewrites", [])
        if isinstance(r, dict) and r.get("after")
    ]

    match = None
    raw_match = data.get("match")
    if has_job and isinstance(raw_match, dict):
        match = {
            "score": clamp(raw_match.get("score"), 0, 100),
            "matched": [str(k) for k in raw_match.get("matched", [])],
            "missing": [str(k) for k in raw_match.get("missing", [])],
            "summary": str(raw_match.get("summary", "")),
        }

    return {
        "overall": clamp(data.get("overall"), 0, 100),
        "headline": str(data.get("headline", "Resume analysis")),
        "summary": str(data.get("summary", "")),
        "categories": categories,
        "strengths": [str(s) for s in data.get("strengths", [])],
        "issues": issues,
        "rewrites": rewrites,
        "match": match,
    }
