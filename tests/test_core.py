import json
from types import SimpleNamespace

import pytest

import tools
from check_citations import check
from research import slugify, summarize


def test_with_retry_uses_backoff_and_stops(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)
    monkeypatch.setattr(tools.random, "uniform", lambda _a, _b: 0)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise tools.RetryableError("temporary")
        return "ok"

    assert tools.with_retry(flaky, attempts=3, base=2, cap=10) == "ok"
    assert sleeps == [2, 4]


def test_with_retry_honors_retry_after_and_no_final_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools.time, "sleep", sleeps.append)

    def always_fails():
        raise tools.RetryableError("busy", retry_after=99)

    with pytest.raises(tools.RetryableError):
        tools.with_retry(always_fails, attempts=2, cap=7)
    assert sleeps == [7]


def test_slugify_is_safe_and_bounded():
    assert slugify("../../x") == "x"
    assert slugify("!!!") == "topic"
    assert len(slugify("a" * 100)) == 60


def test_citation_validator_accepts_consistent_report():
    sources = [
        {"n": 1, "url": "https://arxiv.org/abs/1"},
        {"n": 2, "url": "https://example.org/two"},
    ]
    report = "Claim [1] and another [2].\n\n## References\n[1] A. arxiv. https://arxiv.org/abs/1\n[2] B. web. https://example.org/two\n"
    assert check(report, sources) == []


def test_citation_validator_detects_mismatch():
    sources = [{"n": 1, "url": "https://example.org/one"}]
    report = "Claim [2].\n\n## References\n[1] A. web. https://example.org/wrong\n"
    problems = check(report, sources)
    assert "[2] cited but missing from sources.json" in problems
    assert "source [1] never cited" in problems
    assert "reference [1] URL does not match sources.json" in problems


def test_citation_validator_allows_parentheses_inside_url():
    url = "https://example.org/fulltext/S2666-3899(25)00218-1"
    sources = [{"n": 1, "url": url}]
    report = f"Claim [1].\n\n## References\n[1] Paper. web. {url} (n.d.)\n"
    assert check(report, sources) == []


def test_summarize_counts_lead_calls_and_tokens():
    messages = [SimpleNamespace(
        tool_calls=[{"name": "task"}, {"name": "task"}, {"name": "execute"}],
        usage_metadata={"input_tokens": 10, "output_tokens": 4},
    )]
    summary = summarize(messages, 1.26, "model")
    assert summary["subagent_calls"] == 2
    assert summary["tool_calls"] == {"execute": 1, "task": 2}
    assert summary["tokens"] == {"input": 10, "output": 4}
    assert summary["elapsed_s"] == 1.3
