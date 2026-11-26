# 📄 AI Resume Scanner

An LLM-powered web app that reviews a resume the way a recruiter and an applicant tracking system (ATS) would. It scores the resume, points out what to fix first, and rewrites weak bullet points. Paste in a job posting to get a match score and a list of missing keywords.

**Live demo:** _add your Streamlit link here_

## Features

- **Upload PDF, DOCX or TXT**, or paste the text directly
- **Overall score (0–100)** plus a radar chart across six areas: impact & metrics, clarity, technical skills, projects, experience relevance and ATS-friendliness
- **Prioritized fixes**, each tagged high, medium or low severity with a concrete suggestion
- **Bullet-point rewrites** shown before and after, using `[X]` placeholders so no results get invented
- **Job match mode**: paste a job description to see a match score and the keywords you have and the ones you're missing
- **Example result** so visitors can see the output without uploading anything

## How it works

```
Resume file ──► resume_parser.py ──► plain text
                                        │
           role + job description ──────┤
                                        ▼
                                 analyzer.py
                     builds a structured prompt, calls Gemini
                     in JSON mode, validates and cleans the output
                                        │
                                        ▼
                                    app.py
                    Streamlit UI: gauges, radar chart, tabs
```

- **`resume_parser.py`** extracts and normalizes text with `pypdf` and `python-docx`, and rejects scanned or empty files with a helpful message.
- **`analyzer.py`** handles prompt engineering and the LLM call. It uses Gemini's JSON response mode, then `validate()` clamps scores, fills in missing fields and sorts issues, so a malformed model response can't break the UI. If one model isn't available, it falls back to the next.
- **`app.py`** is the Streamlit front end, with Plotly charts and a per-session scan limit to protect the API quota.
- **`test_analyzer.py`** has unit tests for the parsing and validation logic.

## Tech stack

Python · Streamlit · Google Gemini API (`google-genai`) · Plotly · pypdf · python-docx · pytest

## Run it locally

```bash
git clone https://github.com/muqadas-fatima16/ai-resume-scanner.git
cd ai-resume-scanner
pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then paste your key inside
streamlit run app.py
```

Get a free Gemini API key at [Google AI Studio](https://aistudio.google.com/apikey).

Run the tests with `python -m pytest`.

## Privacy

Resumes are processed in memory for a single request and are never saved by the app.

## Ideas for next steps

- Side-by-side comparison of two versions of a resume
- Export the report as a PDF
- Bias-aware screening mode for recruiters (hide names and schools before scoring)
