"""Unit tests for the parts that don't need the API. Run with: python -m pytest"""

from analyzer import CATEGORIES, build_prompt, parse_json, validate


def test_parse_json_strips_fences():
    assert parse_json('```json\n{"overall": 80}\n```') == {"overall": 80}


def test_validate_clamps_and_fills_defaults():
    result = validate({"overall": 140, "categories": [{"name": "Projects", "score": 12}]}, has_job=False)
    assert result["overall"] == 100
    assert [c["name"] for c in result["categories"]] == CATEGORIES
    assert next(c for c in result["categories"] if c["name"] == "Projects")["score"] == 10
    assert result["match"] is None


def test_validate_sorts_issues_by_severity():
    data = {"issues": [
        {"severity": "low", "issue": "a"},
        {"severity": "high", "issue": "b"},
        {"severity": "weird", "issue": "c"},
    ]}
    assert [i["severity"] for i in validate(data, has_job=False)["issues"]] == ["high", "medium", "low"]


def test_prompt_includes_job_only_when_given():
    assert "JOB DESCRIPTION" not in build_prompt("resume text")
    assert "JOB DESCRIPTION" in build_prompt("resume text", job_description="Python intern")
