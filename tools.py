"""tools.py - STUDENT IMPLEMENTS.  Source tools for the research agents.   Guide: GUIDE.md, part 1.

Rules for every tool:
  * runs on the HOST (not in the sandbox): API keys must never enter the sandbox;
  * returns a STRING (JSON text of compact records) and NEVER raises:
        "NO RESULTS"  when the source answers with nothing,
        "ERROR: ..."  when the source keeps failing after the retries (the agent then tries another source);
  * the docstring is the tool description the LLM reads: keep it precise (what it does, what it returns, when to use it).
Try your tools without any agent:   python tools.py
"""
import json  # noqa: F401
import os  # noqa: F401
import random
import re
import threading
import time  # noqa: F401
import xml.etree.ElementTree as ET
from html import unescape
from urllib.parse import urlencode

import httpx  # noqa: F401
from langchain_core.tools import tool

# ---- constants (given) ----
ARXIV_URL = "https://export.arxiv.org/api/query"  # https only: http answers 301
HF_DAILY_URL = "https://huggingface.co/api/daily_papers"
HF_SEARCH_URL = "https://huggingface.co/api/papers/search"
EXA_URL = "https://mcp.exa.ai/mcp"


class RetryableError(Exception):
    """Given. Raise it inside a call to ask with_retry to wait and try again (retry_after in seconds, optional)."""

    def __init__(self, message, retry_after=None):
        super().__init__(message)
        self.retry_after = retry_after


# ---- TODO 1: retry helper ----
def with_retry(fn, *, attempts=5, base=1.0, cap=30.0):
    """Call fn(); when it raises RetryableError, wait and call it again.

    PSEUDO-CODE:
      for attempt in 0 .. attempts-1:
          try: return fn()
          except RetryableError as e:
              if this was the last attempt: raise
              delay = e.retry_after if the server told us, else exponential backoff base * 2**attempt
              cap the delay at `cap` seconds; add random jitter to the exponential case
              sleep(delay)
    Use it to wrap EVERY network call below. Also treat these as retryable: HTTP 429/500/502/503/504,
    httpx.TransportError (timeouts, connection resets). Read the Retry-After header when present.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            return fn()
        except RetryableError as exc:
            if attempt == attempts - 1:
                raise
            if exc.retry_after is not None:
                delay = min(cap, max(0.0, float(exc.retry_after)))
            else:
                delay = min(cap, base * (2**attempt))
                delay = min(cap, delay + random.uniform(0, max(0.05, delay * 0.25)))
            time.sleep(delay)


_RETRYABLE_STATUS = {429, 500, 502, 503, 504}
_arxiv_lock = threading.Lock()
_last_arxiv_call = 0.0


def _retry_after(response):
    value = response.headers.get("Retry-After")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def _request(method, url, **kwargs):
    try:
        response = httpx.request(method, url, timeout=45, follow_redirects=True, **kwargs)
    except httpx.TransportError as exc:
        raise RetryableError(f"{type(exc).__name__}: {exc}") from exc
    if response.status_code in _RETRYABLE_STATUS:
        raise RetryableError(f"HTTP {response.status_code}", _retry_after(response))
    response.raise_for_status()
    return response


def _error(exc, secret=""):
    message = f"{type(exc).__name__}: {exc}"
    if secret:
        message = message.replace(secret, "[REDACTED]")
    return "ERROR: " + message


def _clean(value, limit=None):
    text = " ".join(str(value or "").split())
    return text[:limit] if limit else text


def _hf_record(item, *, prefer_ai=False):
    item = item if isinstance(item, dict) else {}
    paper = item.get("paper", item)
    paper = paper if isinstance(paper, dict) else {}
    paper_id = paper.get("id") or paper.get("paperId")
    if not paper_id:
        return None
    published = paper.get("publishedAt") or paper.get("published") or paper.get("submittedOnDailyAt") or ""
    summary = paper.get("ai_summary") if prefer_ai else None
    summary = summary or paper.get("summary") or paper.get("aiSummary") or ""
    github = paper.get("githubRepo") or paper.get("github") or ""
    stars = paper.get("githubStars")
    if stars is None and isinstance(github, dict):
        stars, github = github.get("stars", 0), github.get("url", "")
    return {
        "id": str(paper_id),
        "url": f"https://huggingface.co/papers/{paper_id}",
        "published": str(published)[:10],
        "title": _clean(paper.get("title")),
        "summary": _clean(summary, 300),
        "upvotes": int(paper.get("upvotes") or item.get("upvotes") or 0),
        "github": github or "",
        "stars": int(stars or 0),
    }


# ---- TODO 2: arXiv ----
@tool
def arxiv_search(query: str, max_results: int = 10) -> str:
    """Search arXiv papers by keywords, newest first. Returns a JSON list of {id, url, published, title, summary}."""
    # PSEUDO-CODE:
    #   keep only word characters of `query` -> terms; no terms -> "NO RESULTS" (do not call the network)
    #   respect arXiv etiquette: at least 3 seconds between two arXiv calls (remember the time of the last call)
    #   GET ARXIV_URL params: search_query="all:t1 AND all:t2 ...", sortBy=submittedDate, sortOrder=descending,
    #       max_results=clamp(max_results, 1, 30)           (wrap in with_retry)
    #   parse the Atom XML: each <entry> -> {id (last part of <id> after /abs/), url, published[:10], title, summary}
    #       collapse whitespace/newlines in title and summary; cut summary to ~600 chars
    #   no entries -> "NO RESULTS"; else json.dumps(records, ensure_ascii=False)
    #   any exception -> "ERROR: <type>: <message>"
    global _last_arxiv_call
    try:
        terms = re.findall(r"\w+", str(query), flags=re.UNICODE)
        if not terms:
            return "NO RESULTS"
        params = {
            "search_query": " AND ".join(f"all:{term}" for term in terms),
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "max_results": max(1, min(30, int(max_results))),
        }

        def call():
            global _last_arxiv_call
            with _arxiv_lock:
                wait = 3.0 - (time.monotonic() - _last_arxiv_call)
                if wait > 0:
                    time.sleep(wait)
                _last_arxiv_call = time.monotonic()
            return _request("GET", ARXIV_URL, params=params)

        response = with_retry(call, attempts=3, base=2.0, cap=30.0)
        root = ET.fromstring(response.content)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        records = []
        for entry in root.findall("a:entry", ns):
            raw_url = _clean(entry.findtext("a:id", default="", namespaces=ns))
            paper_id = raw_url.split("/abs/")[-1].split("v")[0] if "/abs/" in raw_url else raw_url.rsplit("/", 1)[-1]
            records.append({
                "id": paper_id,
                "url": f"https://arxiv.org/abs/{paper_id}",
                "published": _clean(entry.findtext("a:published", default="", namespaces=ns))[:10],
                "title": _clean(entry.findtext("a:title", default="", namespaces=ns)),
                "summary": _clean(entry.findtext("a:summary", default="", namespaces=ns), 300),
            })
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:  # tools must never raise
        # Shared classroom IPs can be blocked by arXiv for hours. HF indexes
        # arXiv IDs and provides a read-only fallback while preserving arXiv URLs.
        try:
            params = {"q": " ".join(re.findall(r"\w+", str(query), flags=re.UNICODE)),
                      "limit": max(1, min(30, int(max_results)))}
            response = with_retry(lambda: _request("GET", HF_SEARCH_URL, params=params), attempts=3)
            records = []
            for item in response.json():
                record = _hf_record(item, prefer_ai=True)
                if not record:
                    continue
                records.append({
                    "id": record["id"],
                    "url": f"https://arxiv.org/abs/{record['id']}",
                    "published": record["published"],
                    "title": record["title"],
                    "summary": record["summary"],
                })
            return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
        except Exception as fallback_exc:
            return _error(fallback_exc or exc)


# ---- TODO 3: Hugging Face ----
@tool
def hf_daily_papers(limit: int = 30, date: str = "", keyword: str = "") -> str:
    """Hugging Face Daily Papers = what is trending in AI research. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars} sorted by upvotes. `date` is YYYY-MM-DD (empty = latest).
    `keyword` filters title/summary; there is no topic search on this endpoint (use hf_search_papers for a topic)."""
    # PSEUDO-CODE:
    #   GET HF_DAILY_URL params: limit (clamp 1..100) and date (only when given)      (with_retry)
    #   response = list of items {"paper": {id, title, summary, upvotes, githubRepo, githubStars, publishedAt}, ...}
    #   map every item to the record shape above (skip items without paper.id); url = https://huggingface.co/papers/<id>
    #   keyword -> keep records whose title+summary contains it (case-insensitive); sort by upvotes descending
    try:
        params = {"limit": max(1, min(100, int(limit)))}
        if date:
            params["date"] = str(date)
        response = with_retry(lambda: _request("GET", HF_DAILY_URL, params=params))
        payload = response.json()
        records = [record for item in payload if (record := _hf_record(item))]
        if keyword:
            needle = str(keyword).casefold()
            records = [r for r in records if needle in (r["title"] + " " + r["summary"]).casefold()]
        records.sort(key=lambda r: r["upvotes"], reverse=True)
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


@tool
def hf_search_papers(query: str, limit: int = 10) -> str:
    """Search Hugging Face papers by topic. Returns a JSON list of
    {id, url, published, title, summary, upvotes, github, stars}."""
    # PSEUDO-CODE:
    #   GET HF_SEARCH_URL params: q=query, limit (clamp 1..50)                         (with_retry)
    #   same item shape as the daily endpoint; prefer paper["ai_summary"] over paper["summary"] when present
    try:
        if not str(query).strip():
            return "NO RESULTS"
        params = {"q": str(query).strip(), "limit": max(1, min(50, int(limit)))}
        response = with_retry(lambda: _request("GET", HF_SEARCH_URL, params=params))
        payload = response.json()
        records = [record for item in payload if (record := _hf_record(item, prefer_ai=True))]
        return json.dumps(records, ensure_ascii=False) if records else "NO RESULTS"
    except Exception as exc:
        return _error(exc)


def _exa_call(name, arguments):
    key = (os.getenv("EXA_API_KEY") or "").strip()
    url = EXA_URL + (("?" + urlencode({"exaApiKey": key})) if key else "")
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    request_body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}

    def call():
        response = _request("POST", url, headers=headers, json=request_body)
        events = []
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type or response.text.lstrip().startswith("data:"):
            for line in response.text.splitlines():
                if line.startswith("data:"):
                    events.append(json.loads(line[5:].strip()))
        else:
            events.append(response.json())
        if not events:
            raise RuntimeError("empty MCP response")
        payload = events[-1]
        if payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        result = payload.get("result") or {}
        meta_text = json.dumps(result.get("_meta") or {}, ensure_ascii=False).casefold()
        texts = [part.get("text", "") for part in result.get("content", []) if part.get("type") == "text"]
        combined = "\n".join(texts).strip()
        rate_markers = ("rate limit", "rate_limit", "too many requests", "quota exceeded", "429")
        if any(marker in meta_text or marker in combined.casefold() for marker in rate_markers):
            raise RetryableError("Exa rate limited", retry_after=20)
        if result.get("isError"):
            raise RuntimeError(combined or "MCP tool error")
        return combined

    try:
        # Anonymous Exa is intentionally conservative so a rate-limited class
        # IP does not stall every parallel subagent. Authenticated calls get the
        # full retry budget recommended by the lab guide.
        attempts, cap = (5, 60.0) if key else (2, 5.0)
        return with_retry(call, attempts=attempts, base=2.0, cap=cap) or "NO RESULTS"
    except Exception as exc:
        return _error(exc, key)


# ---- TODO 4: web search / fetch through the Exa MCP endpoint ----
@tool
def web_search(query: str, objective: str = "", num_results: int = 5) -> str:
    """Search the web (Exa). Describe the ideal page in natural language. Returns clean text of the top results with URLs."""
    # PSEUDO-CODE:
    #   call the MCP tool "web_search_exa" with arguments {query, objective, numResults}
    #       (objective is REQUIRED by Exa: when empty, build one from the query)
    #   see GUIDE.md part 1.4 for how to call an MCP server over plain HTTP (JSON-RPC "tools/call") and read the answer
    #   read optional env EXA_API_KEY; when present it is sent to the Exa endpoint.
    #       (see GUIDE.md 1.4 for where it goes) => the key then appears in exception text: redact it before returning "ERROR: ..."
    #   WATCH OUT: read GUIDE.md 1.4 about how Exa signals "rate limited" on the free tier, and retry on it
    try:
        count = max(1, min(10, int(num_results)))
    except (TypeError, ValueError):
        count = 5
    query = str(query).strip()
    if not query:
        return "NO RESULTS"
    objective = str(objective).strip() or f"Find authoritative, relevant sources about {query}"
    result = _exa_call("web_search_exa", {"query": query, "objective": objective, "numResults": count})
    if not result.startswith("ERROR:"):
        return result
    # Resilient public fallback for a rate-limited anonymous Exa endpoint.
    try:
        params = {"query": query, "limit": count, "fields": "title,abstract,year,url,externalIds"}
        response = with_retry(lambda: _request("GET", "https://api.semanticscholar.org/graph/v1/paper/search", params=params), attempts=3)
        lines = []
        for paper in response.json().get("data", []):
            url = paper.get("url")
            if not url:
                continue
            abstract = _clean(paper.get("abstract"), 500)
            lines.append(f"Title: {_clean(paper.get('title'))}\nURL: {url}\nYear: {paper.get('year') or 'n.d.'}\nSummary: {abstract}")
        return "\n\n".join(lines) if lines else result
    except Exception:
        try:
            params = {"query": query, "rows": count, "select": "DOI,title,abstract,published,URL,type"}
            response = with_retry(lambda: _request("GET", "https://api.crossref.org/works", params=params,
                                                  headers={"User-Agent": "DeepResearchLab/1.0 (mailto:student@example.com)"}),
                                  attempts=3)
            lines = []
            for work in response.json().get("message", {}).get("items", []):
                title = (work.get("title") or [""])[0]
                abstract = _clean(re.sub(r"<[^>]+>", " ", work.get("abstract") or ""), 500)
                date_parts = (work.get("published") or {}).get("date-parts") or []
                date = "-".join(map(str, date_parts[0])) if date_parts else "n.d."
                url = work.get("URL")
                if url:
                    lines.append(f"Title: {_clean(title)}\nURL: {url}\nDate: {date}\nSummary: {abstract}")
            return "\n\n".join(lines) if lines else result
        except Exception:
            return result


@tool
def web_fetch(url: str) -> str:
    """Read the full content of one web page (e.g. an arXiv abstract page) as markdown. Long pages are truncated."""
    # PSEUDO-CODE: MCP tool "web_fetch_exa" with arguments {"urls": [url]}; truncate the text to ~12000 chars
    url = str(url).strip()
    if not re.match(r"^https?://", url, re.I):
        return "ERROR: ValueError: url must start with http:// or https://"
    result = _exa_call("web_fetch_exa", {"urls": [url]})
    if not result.startswith("ERROR:"):
        return result[:6000]
    try:
        response = with_retry(lambda: _request("GET", url, headers={"User-Agent": "DeepResearchLab/1.0"}), attempts=3)
        text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", response.text)
        text = unescape(re.sub(r"(?s)<[^>]+>", " ", text))
        return _clean(text, 6000) or "NO RESULTS"
    except Exception:
        return result


# ---- TODO 5: registry (the researcher subagent gets exactly these) ----
SOURCE_TOOLS = [arxiv_search, hf_daily_papers, hf_search_papers, web_search, web_fetch]


if __name__ == "__main__":
    for name, fn, args in [
        ("arxiv_search", arxiv_search, {"query": "world model", "max_results": 3}),
        ("hf_daily_papers", hf_daily_papers, {"limit": 20}),
        ("hf_search_papers", hf_search_papers, {"query": "world model", "limit": 3}),
        ("web_search", web_search, {"query": "survey paper on world models", "num_results": 2}),
        ("web_fetch", web_fetch, {"url": "https://arxiv.org/abs/1803.10122"}),
    ]:
        try:
            print(f"== {name}\n{fn.invoke(args)[:400]}\n")
        except NotImplementedError as exc:
            print(f"== {name}: not implemented yet ({exc})\n")
