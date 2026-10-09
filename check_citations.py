"""check_citations.py - STUDENT IMPLEMENTS `check`.   Runs INSIDE the sandbox (standard library only).

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def check(report_text, sources):
    """Return a list of problem strings (empty list = OK).

    PSEUDO-CODE:
      problems = []
      if sources is empty: return ["no sources in sources.json"]
      for each source entry:
          n must be an int                       -> problem if not
          url must start with http:// or https://-> problem if not
          the same url must not appear twice     -> problem if duplicated
      split report_text at the heading "## References":
          body = text before it; if the heading is missing -> problem
      cited = set of numbers found as [n] in the BODY only (not in the reference list; use a regex)
      every number in `cited` must exist in sources -> problem "[n] cited but missing from sources.json"
      every source number must be in `cited`        -> problem "source [n] never cited"
      the lines of the References section that start with "[n]" (regex) are the reference lines:
          every source needs exactly ONE reference line (none missing, no number twice, no number that is not a source)
          each reference line holds exactly ONE http(s) URL and it must equal that source's url
          (a line bundling several sources under one number is a problem)
      return problems
    """
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["no sources in sources.json"]

    by_n = {}
    seen_urls = set()
    for entry in sources:
        if not isinstance(entry, dict):
            problems.append("source entry is not an object")
            continue
        n, url = entry.get("n"), entry.get("url")
        if not isinstance(n, int) or isinstance(n, bool):
            problems.append(f"source n={n!r} is not an int")
        elif n in by_n:
            problems.append(f"source number [{n}] appears more than once")
        else:
            by_n[n] = entry
        if not isinstance(url, str) or not re.match(r"^https?://", url):
            problems.append(f"source [{n}] has invalid url")
        elif url in seen_urls:
            problems.append(f"duplicate source url: {url}")
        else:
            seen_urls.add(url)

    heading = re.search(r"(?m)^##[ \t]+References[ \t]*$", report_text)
    if not heading:
        problems.append("missing ## References heading")
        body, references = report_text, ""
    else:
        body, references = report_text[:heading.start()], report_text[heading.end():]

    # Ignore fenced/inline code and Markdown link labels such as [1](url).
    body = re.sub(r"```.*?```|`[^`\n]*`", "", body, flags=re.S)
    cited = set()
    for match in re.finditer(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()", body):
        for part in re.split(r"\s*,\s*", match.group(1)):
            span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
            if span:
                a, b = map(int, span.groups())
                cited.update(range(a, b + 1) if 0 <= b - a <= 200 else (a, b))
            else:
                cited.add(int(part))
    for n in sorted(cited - set(by_n)):
        problems.append(f"[{n}] cited but missing from sources.json")
    for n in sorted(set(by_n) - cited):
        problems.append(f"source [{n}] never cited")

    reference_lines = {}
    for line in references.splitlines():
        match = re.match(r"^\s*\[(\d+)\]\s*(.*)$", line)
        if not match:
            continue
        n, rest = int(match.group(1)), match.group(2)
        reference_lines.setdefault(n, []).append(rest)
    for n in sorted(by_n):
        lines = reference_lines.get(n, [])
        if not lines:
            problems.append(f"source [{n}] has no reference line")
        elif len(lines) > 1:
            problems.append(f"source [{n}] has more than one reference line")
    for n in sorted(set(reference_lines) - set(by_n)):
        problems.append(f"reference [{n}] is not a source")
    for n, lines in reference_lines.items():
        if len(lines) != 1 or n not in by_n:
            continue
        urls = re.findall(r"https?://\S+", lines[0])
        urls = [url.rstrip(".,;") for url in urls]
        if len(urls) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL")
        elif urls[0] != by_n[n].get("url"):
            problems.append(f"reference [{n}] URL does not match sources.json")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
