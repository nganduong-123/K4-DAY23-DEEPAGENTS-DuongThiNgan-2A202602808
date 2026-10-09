"""research.py - STUDENT IMPLEMENTS.  The main script.   Guide: GUIDE.md, part 3.

Usage:  python research.py "survey about world model"
Result: reports/<slug>.md   reports/<slug>.sources.json   reports/<slug>.meta.json
"""
import json  # noqa: F401
import os  # noqa: F401
import re  # noqa: F401
import sys
import time  # noqa: F401
from collections import Counter  # noqa: F401
from pathlib import Path

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain.chat_models import init_chat_model

from agents import FINALIZER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent  # noqa: F401
from model import make_model  # noqa: F401
from sandbox import download, open_sandbox, upload  # noqa: F401

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"   # provided: uploaded next to your validator


def slugify(topic):
    """Turn a topic into a safe file name: lower case, runs of non-word characters become one "-", max 60 chars,
    never empty (fall back to "topic"). The topic is user input: "../../x" must not escape reports/."""
    slug = re.sub(r"[^\w]+", "-", str(topic).lower(), flags=re.UNICODE).strip("-_")
    slug = slug[:60].rstrip("-_")
    return slug or "topic"


def build_prompt(topic):
    """The user message sent to the lead agent."""
    return f"""Research this topic and produce the required cited survey: {topic}

Use at least three researcher delegations in parallel and ensure the final retained sources span at least three source families. Follow the complete workflow in your system prompt, including finalization, validation, and citation spot checks. Do not finish until the report and sources files are valid."""


def summarize(messages, elapsed, model_name):
    """Return {"model", "elapsed_s", "subagent_calls", "tool_calls": {name: count}, "tokens": {"input", "output"}}.

    PSEUDO-CODE: walk the lead's messages; for every message with tool_calls count call["name"] (subagent_calls = the
    count of "task"); add the input/output token counts from each message's usage_metadata when present.
    (Lead messages only: subagent tokens are not included, so this undercounts the real cost.)
    elapsed_s rounded to 0.1.
    """
    tool_calls = Counter()
    input_tokens = output_tokens = 0
    for message in messages:
        calls = getattr(message, "tool_calls", None)
        if calls is None and isinstance(message, dict):
            calls = message.get("tool_calls") or message.get("additional_kwargs", {}).get("tool_calls")
        for call in calls or []:
            name = call.get("name") or call.get("function", {}).get("name")
            if name:
                tool_calls[name] += 1
        usage = getattr(message, "usage_metadata", None)
        if usage is None and isinstance(message, dict):
            usage = message.get("usage_metadata") or {}
        usage = usage or {}
        input_tokens += int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or usage.get("completion_tokens") or 0)
    return {
        "model": model_name,
        "elapsed_s": round(float(elapsed), 1),
        "subagent_calls": tool_calls.get("task", 0),
        "tool_calls": dict(sorted(tool_calls.items())),
        "tokens": {"input": input_tokens, "output": output_tokens},
    }


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS):
    """Download the report from the sandbox and write the three files into reports_dir. Return the report path.

    PSEUDO-CODE:
      files = download(backend, [REPORT_PATH, SOURCES_PATH])
      if the report is missing/empty or sources.json is missing/invalid JSON: raise RuntimeError and WRITE NOTHING
          (a failed run must never leave an empty or half-written report behind)
      write <slug>.sources.json, <slug>.meta.json (topic + summarize(...) + n_sources + source_families: the sorted
      distinct "source" values of sources.json) and <slug>.md
    """
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_bytes, source_bytes = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_bytes or not report_bytes.strip():
        raise RuntimeError("report.md is missing or empty")
    if not source_bytes or not source_bytes.strip():
        raise RuntimeError("sources.json is missing or empty")
    try:
        sources = json.loads(source_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"sources.json is invalid: {exc}") from exc
    if not isinstance(sources, list) or not sources:
        raise RuntimeError("sources.json must be a non-empty JSON list")

    reports_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(topic)
    report_path = reports_dir / f"{slug}.md"
    sources_path = reports_dir / f"{slug}.sources.json"
    meta_path = reports_dir / f"{slug}.meta.json"
    meta = {
        "topic": topic,
        **summarize(messages, elapsed, model_name),
        "n_sources": len(sources),
        "source_families": sorted({str(item.get("source")) for item in sources if item.get("source")}),
    }

    temporary = []
    try:
        for path, data in (
            (report_path, report_bytes),
            (sources_path, json.dumps(sources, ensure_ascii=False, indent=2).encode("utf-8")),
            (meta_path, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")),
        ):
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_bytes(data)
            temporary.append((tmp, path))
        for tmp, path in temporary:
            tmp.replace(path)
    except Exception:
        for tmp, _ in temporary:
            tmp.unlink(missing_ok=True)
        raise
    return report_path


def main(topic):
    """Return the process exit code (0 ok, 1 failed run, 2 no topic).

    PSEUDO-CODE:
      empty topic -> print usage to stderr, return 2
      model = make_model(); start = time.monotonic()
      with open_sandbox() as backend:                # the sandbox is always cleaned up, even on errors
          backend.execute("mkdir -p <WORKDIR>/research/notes <WORKDIR>/report")
          upload(backend, {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes()})
          agent = build_lead_agent(backend, model)
          result = agent.invoke({"messages": [{"role": "user", "content": build_prompt(topic)}]},
                                config={"recursion_limit": 1000})
          save_outputs(...); on RuntimeError print "FAILED: ..." to stderr and return 1
      print where the report was saved; return 0
    """
    topic = str(topic).strip()
    if not topic:
        print('Usage: python research.py "survey topic"', file=sys.stderr)
        return 2
    try:
        model = make_model()
        rpm = float(os.getenv("LAB_MODEL_RPM", "0"))
        if rpm > 0 and hasattr(model, "rate_limiter"):
            model.rate_limiter = InMemoryRateLimiter(
                requests_per_second=rpm / 60.0,
                check_every_n_seconds=0.1,
                max_bucket_size=1,
            )
        # OpenAI-compatible gateways may otherwise reserve the model's entire
        # context window as output, which can exceed per-key credit limits.
        max_tokens = int(os.getenv("LAB_MAX_TOKENS", "8192"))
        if hasattr(model, "max_tokens"):
            model.max_tokens = max_tokens
        if hasattr(model, "max_output_tokens"):
            model.max_output_tokens = max_tokens
        model_name = getattr(model, "model_name", None) or getattr(model, "model", None) or os.getenv("LAB_MODEL", "unknown")
        started = time.monotonic()
        with open_sandbox() as backend:
            result = backend.execute(f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            if result.exit_code:
                raise RuntimeError(f"cannot initialize sandbox: {result.output}")
            upload(backend, {
                VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(),
                FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            })
            subagent_model = None
            subagent_name = (os.getenv("LAB_SUB_MODEL") or "").strip()
            if subagent_name:
                subagent_model = init_chat_model(subagent_name)
                if rpm > 0 and hasattr(subagent_model, "rate_limiter"):
                    subagent_model.rate_limiter = InMemoryRateLimiter(
                        requests_per_second=rpm / 60.0,
                        check_every_n_seconds=0.1,
                        max_bucket_size=1,
                    )
            agent = build_lead_agent(backend, model, subagent_model)
            outcome = agent.invoke(
                {"messages": [{"role": "user", "content": build_prompt(topic)}]},
                config={"recursion_limit": 1000},
            )
            messages = outcome.get("messages", []) if isinstance(outcome, dict) else []
            elapsed = time.monotonic() - started
            path = save_outputs(backend, topic, messages, elapsed, str(model_name))
        print(f"Saved report: {path}")
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(" ".join(sys.argv[1:])))
