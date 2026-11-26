"""AI Resume Scanner — a Streamlit app that reviews resumes with an LLM."""

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from analyzer import analyze_resume
from resume_parser import ResumeParseError, extract_text

MAX_SCANS_PER_SESSION = 5
SEVERITY_ICON = {"high": "🔴", "medium": "🟠", "low": "🟢"}

st.set_page_config(page_title="AI Resume Scanner", page_icon="📄", layout="centered")


# ---------- helpers ----------
def score_color(pct: float) -> str:
    if pct >= 75:
        return "#16a34a"
    if pct >= 55:
        return "#d97706"
    return "#dc2626"


def gauge(value: int, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": "", "font": {"size": 40}},
        title={"text": title, "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "visible": False},
            "bar": {"color": score_color(value), "thickness": 0.8},
            "bgcolor": "rgba(128,128,128,0.15)",
            "borderwidth": 0,
        },
    ))
    fig.update_layout(height=200, margin=dict(l=10, r=10, t=40, b=0),
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def radar(categories: list[dict]) -> go.Figure:
    names = [c["name"] for c in categories]
    scores = [c["score"] for c in categories]
    fig = go.Figure(go.Scatterpolar(
        r=scores + scores[:1], theta=names + names[:1],
        fill="toself", line_color="#4f46e5", fillcolor="rgba(79,70,229,0.25)",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 10], tickvals=[2, 4, 6, 8, 10])),
        showlegend=False, height=340, margin=dict(l=50, r=50, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def get_api_key() -> str | None:
    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return None


def show_results(result: dict) -> None:
    has_match = result.get("match") is not None

    # Overview
    cols = st.columns(2 if has_match else 1)
    cols[0].plotly_chart(gauge(result["overall"], "Overall score"), width="stretch",
                         config={"displayModeBar": False})
    if has_match:
        cols[1].plotly_chart(gauge(result["match"]["score"], "Job match"),
                             width="stretch", config={"displayModeBar": False})
    st.subheader(result["headline"])
    st.write(result["summary"])

    tabs = st.tabs(["📊 Scores", "🛠️ Fixes", "✍️ Rewrites", "💪 Strengths"]
                   + (["🎯 Job match"] if has_match else []))

    with tabs[0]:
        st.plotly_chart(radar(result["categories"]), width="stretch",
                        config={"displayModeBar": False})
        for c in result["categories"]:
            st.markdown(f"**{c['name']}** — {c['score']}/10")
            st.progress(c["score"] / 10)
            if c["note"]:
                st.caption(c["note"])

    with tabs[1]:
        for issue in result["issues"]:
            st.markdown(f"{SEVERITY_ICON[issue['severity']]} **{issue['issue']}**")
            st.markdown(f"→ {issue['fix']}")
            st.divider()

    with tabs[2]:
        st.caption("Replace [X] with your real numbers. Never add results you can't back up.")
        for r in result["rewrites"]:
            with st.container(border=True):
                st.markdown("**Before**")
                st.markdown(f"~~{r['before']}~~")
                st.markdown("**After**")
                st.code(r["after"], language=None, wrap_lines=True)

    with tabs[3]:
        for s in result["strengths"]:
            st.markdown(f"✅ {s}")

    if has_match:
        with tabs[4]:
            m = result["match"]
            st.write(m["summary"])
            left, right = st.columns(2)
            left.markdown("**✓ You have**")
            left.markdown("\n".join(f"- {k}" for k in m["matched"]) or "—")
            right.markdown("**✗ Missing**")
            right.markdown("\n".join(f"- {k}" for k in m["missing"]) or "—")


# ---------- page ----------
st.title("📄 AI Resume Scanner")
st.write(
    "Upload a resume to get a score, the most important fixes, and stronger bullet points. "
    "Paste a job posting to see how well the resume matches it."
)

if "scans" not in st.session_state:
    st.session_state.scans = 0

with st.form("scan"):
    uploaded = st.file_uploader("Resume", type=["pdf", "docx", "txt"])
    with st.expander("…or paste the resume text"):
        pasted = st.text_area("Resume text", height=200, label_visibility="collapsed")
    role = st.text_input("Target role (optional)", placeholder="e.g. Software Engineering Intern")
    job = st.text_area("Job description (optional)", height=150,
                       placeholder="Paste a job posting to get a match score")
    submitted = st.form_submit_button("Scan resume", type="primary", width="stretch")

if st.button("Or see an example result", width="stretch"):
    sample = json.loads((Path(__file__).parent / "sample_result.json").read_text())
    st.session_state.result = sample
    st.session_state.is_sample = True

if submitted:
    api_key = get_api_key()
    if not api_key:
        st.error("This app isn't configured with an API key yet.")
    elif st.session_state.scans >= MAX_SCANS_PER_SESSION:
        st.warning("You've hit the scan limit for this session. Refresh the page to start over.")
    else:
        try:
            resume_text = pasted.strip() or (
                extract_text(uploaded.name, uploaded.getvalue()) if uploaded else ""
            )
            if not resume_text:
                st.error("Upload a resume or paste its text first.")
            else:
                with st.spinner("Reading your resume like a recruiter would…"):
                    st.session_state.result = analyze_resume(api_key, resume_text, role.strip(), job.strip())
                st.session_state.is_sample = False
                st.session_state.scans += 1
        except ResumeParseError as exc:
            st.error(str(exc))
        except Exception as exc:
            print(f"Scan failed: {exc!r}")
            st.error("The AI service is busy or unavailable right now. Please try again in a minute.")
            with st.expander("Error details"):
                st.code(str(exc)[:600], language=None, wrap_lines=True)

if "result" in st.session_state:
    st.divider()
    if st.session_state.get("is_sample"):
        st.info("Example result for a sample student resume.")
    show_results(st.session_state.result)

st.divider()
st.caption("Built with Python, Streamlit and Google Gemini. Resumes aren't stored.")
