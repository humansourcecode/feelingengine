"""Context QA — generate a review ledger for a written article.

Combines deterministic rule checks (regex-based) with optional LLM-based
semantic analysis to identify places in an article where a first-time
reader might lack context. Outputs a review ledger in the format consumed
by review_wizard.py.

Run:
    python -m feeling_engine.qa.context_qa article.md               # stdout
    python -m feeling_engine.qa.context_qa article.md -o ledger.md
    python -m feeling_engine.qa.context_qa article.md --no-llm      # rules only

Checks:

Deterministic (no LLM):
  - placeholders: [TODO], [your-repo], TKTK, XXX, {{...}}
  - link-missing: known service/tool names without nearby inline markdown links
  - acronyms: all-caps 3+ letter words used without glossed first-use
  - bare-urls: URLs not wrapped in markdown link syntax

Semantic (requires ANTHROPIC_API_KEY):
  - context-gap: proper nouns introduced without context
  - jargon: technical terms used without gloss
  - first-use: entity referenced before being explained
  - reference-inconsistency: same entity called different things

Each issue becomes a block in the ledger with a stable id, status=pending,
location in the article, and suggested action.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


# ─── Data model ────────────────────────────────────────────────

@dataclass
class Issue:
    """A single QA finding to be reviewed."""
    id: str
    type: str        # placeholder | link-missing | acronym | bare-url | context-gap | jargon | first-use | reference-inconsistency
    location: str    # human-readable location (e.g., "Line 42" or "Section III")
    current: str     # the text snippet being flagged
    why: str         # why it's a problem
    action: str      # what the user should do
    status: str = "pending"
    amendment: str = ""


# ─── Deterministic rule checks ─────────────────────────────────

PLACEHOLDER_PATTERNS = [
    (r"\[TODO[^\]]*\]",          "generic TODO marker"),
    (r"\[your-[^\]]+\]",          "placeholder marker"),
    (r"\[placeholder[^\]]*\]",    "placeholder marker"),
    (r"\bTKTK\b",                 "journalism placeholder"),
    (r"\bXXX\b",                  "XXX placeholder"),
    (r"\{\{[^}]+\}\}",             "template variable"),
    (r"\bLOREM IPSUM\b",           "lorem ipsum placeholder"),
]

KNOWN_SERVICES = {
    # name: (canonical url to link to, pattern for first-mention detection)
    "TRIBE v2":       "https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/",
    "TRIBE":          "https://ai.meta.com/blog/tribe-v2-brain-predictive-foundation-model/",
    "Claude":         "https://claude.ai",
    "Anthropic":      "https://anthropic.com",
    "Feeling Engine": "https://github.com/humansourcecode/feelingengine",
    "Human Source Code": "",  # channel, no fixed link yet
    "Meta Research":  "https://ai.meta.com/research/",
    "Modal":          "https://modal.com",
    "ElevenLabs":     "https://elevenlabs.io",
    "HuggingFace":    "https://huggingface.co",
    "OpenAI":         "https://openai.com",
    "GitHub":         "https://github.com",
}

ACRONYM_RE = re.compile(r"\b([A-Z]{3,})\b")
# Acronyms that shouldn't be flagged (common English + common tech + common bio)
ACRONYM_IGNORE = {
    "USA", "UK", "USB", "API", "CEO", "CTO", "CFO", "NASA", "CSS", "HTML", "URL",
    "JSON", "FAQ", "DNA", "RNA", "CPU", "GPU", "PDF", "PNG", "JPG", "SVG",
    "AI", "ML", "NLP", "SQL", "CLI", "UI", "UX", "MVP", "SaaS", "TTS", "STT",
    "EEG", "MRI", "ETA", "RSS", "ATM", "PIN", "DVD", "CD",
}
# Roman numerals I-XX — for section headers like "## III."
ROMAN_RE = re.compile(r"^[IVXLCDM]+$")
# Emphatic uppercase (common intensifier words)
EMPHATIC_WORDS = {"ONLY", "NEVER", "ALWAYS", "EVERY", "NOTHING", "NONE", "ALL", "NOT", "YES", "NO"}

BARE_URL_RE = re.compile(
    r"(?<!\]\()(?<!\]: )(https?://[^\s\]\)]+)(?!\))"
)

MD_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _line_of(text: str, idx: int) -> int:
    """Return 1-based line number for a character offset."""
    return text[:idx].count("\n") + 1


def _first_occurrence(text: str, needle: str) -> Optional[int]:
    """Case-sensitive first char-index of needle in text."""
    idx = text.find(needle)
    return idx if idx >= 0 else None


def check_placeholders(text: str) -> List[Issue]:
    """Detect leftover placeholder / TODO / template markers."""
    out: List[Issue] = []
    seq = 1
    for pattern, label in PLACEHOLDER_PATTERNS:
        for m in re.finditer(pattern, text):
            line = _line_of(text, m.start())
            snippet = _context_snippet(text, m.start(), m.end())
            out.append(Issue(
                id=f"p{seq}",
                type="placeholder",
                location=f"Line {line}",
                current=snippet,
                why=f"Unresolved placeholder detected ({label}). Should not appear in the final article.",
                action=f"Replace the placeholder with the intended content, or remove it.",
            ))
            seq += 1
    return out


def check_links(text: str) -> List[Issue]:
    """Flag known service names that appear without a nearby inline link on first mention."""
    out: List[Issue] = []
    seq = 1
    # Build index of inline-link destinations (so we know which entities already have links)
    linked = set()
    for m in MD_LINK_RE.finditer(text):
        anchor_text = m.group(1).strip().lower()
        linked.add(anchor_text)

    for name, canonical_url in KNOWN_SERVICES.items():
        # sort by length so we match longer first (TRIBE v2 before TRIBE)
        first = _first_occurrence(text, name)
        if first is None:
            continue
        # look 120 chars before/after for a markdown link referencing this name
        window_start = max(0, first - 120)
        window_end = min(len(text), first + len(name) + 120)
        window = text[window_start:window_end]
        has_link = bool(re.search(rf"\[[^\]]*{re.escape(name)}[^\]]*\]\([^)]+\)", window)) \
                   or bool(re.search(rf"\[{re.escape(name)}\]", window))
        # also check if a link to the canonical URL appears in the window
        if canonical_url and canonical_url in window:
            has_link = True
        if not has_link:
            line = _line_of(text, first)
            snippet = _context_snippet(text, first, first + len(name))
            out.append(Issue(
                id=f"l{seq}",
                type="link-missing",
                location=f"Line {line}",
                current=snippet,
                why=f"'{name}' is a named service / tool that should link to its canonical source on first mention so readers can verify or explore.",
                action=(
                    f"Add an inline markdown link on first mention. Suggested URL: {canonical_url}"
                    if canonical_url else
                    f"Add an inline markdown link on first mention to an authoritative source."
                ),
            ))
            seq += 1
    return out


def check_acronyms(text: str) -> List[Issue]:
    """Flag acronyms used without a glossed first-use."""
    out: List[Issue] = []
    seq = 1
    seen: dict[str, int] = {}
    for m in ACRONYM_RE.finditer(text):
        acr = m.group(1)
        if acr in ACRONYM_IGNORE:
            continue
        if acr in EMPHATIC_WORDS:
            continue
        if ROMAN_RE.match(acr):
            continue
        if acr in seen:
            continue
        seen[acr] = m.start()
        # check for gloss: "(expansion)" immediately after, or "Expansion ... (ACR)" immediately before
        tail = text[m.end():m.end()+120]
        head = text[max(0, m.start()-120):m.start()]
        has_gloss = bool(re.match(r"\s*\(\s*[A-Za-z][^)]+\)", tail)) \
                    or bool(re.search(rf"\([^)]*{re.escape(acr)}[^)]*\)", head))
        # also skip if the acronym is explicitly explained inline
        if not has_gloss:
            line = _line_of(text, m.start())
            snippet = _context_snippet(text, m.start(), m.end())
            out.append(Issue(
                id=f"a{seq}",
                type="acronym",
                location=f"Line {line}",
                current=snippet,
                why=f"Acronym '{acr}' used without a glossed first-mention. Reader may not know what it stands for.",
                action=f"On first use, add the expansion in parentheses — e.g., '{acr} (full expansion)' — or skip if the acronym is universally known in the target audience.",
            ))
            seq += 1
    return out


def check_bare_urls(text: str) -> List[Issue]:
    """Flag bare URLs not wrapped in markdown link syntax."""
    out: List[Issue] = []
    seq = 1
    for m in BARE_URL_RE.finditer(text):
        line = _line_of(text, m.start())
        snippet = _context_snippet(text, m.start(), m.end())
        out.append(Issue(
            id=f"u{seq}",
            type="bare-url",
            location=f"Line {line}",
            current=snippet,
            why="Bare URL found outside markdown link syntax. For readability, wrap in [text](url) form.",
            action=f"Wrap in markdown link syntax, e.g. [{m.group(1)[:40]}...]({m.group(1)})",
        ))
        seq += 1
    return out


# ─── Semantic check via Claude ─────────────────────────────────

SEMANTIC_PROMPT = """You are reviewing an article for context adequacy. Identify places where a first-time reader — someone landing on the article with no prior knowledge of the writer or the project — would struggle to understand what is being referenced.

Flag these specific issue types:

- **context-gap**: A proper noun (person, product, tool, place, organization) that is used before it's introduced with enough context for a cold reader.
- **jargon**: A technical term, field-specific concept, or insider vocabulary used without a gloss. "knowledge graph", "atomic design", "fMRI", and non-English terms are examples; determine if each case needs a gloss based on likely audience.
- **first-use**: An entity, concept, or claim referenced before it's explained elsewhere in the article.
- **reference-inconsistency**: The same entity referred to by different names inconsistently (e.g., "Jobs" sometimes, "Steve Jobs" other times, without pattern).

Rules:
- Do NOT flag placeholder / TODO / link-missing issues — those are handled separately.
- Do NOT flag things that ARE introduced properly earlier in the article.
- Do NOT flag a term that is clearly explained inline in the same sentence or next one.
- Focus on issues that would genuinely confuse a cold reader, not stylistic preferences.
- Be conservative — only flag real context gaps.

Return a JSON array. Each element:
{
  "type": "context-gap" | "jargon" | "first-use" | "reference-inconsistency",
  "line_approx": <int line number, 1-based, best estimate>,
  "current": "<verbatim snippet, 5-15 words containing the problem>",
  "why": "<one-sentence explanation for a reader>",
  "action": "<concrete fix suggestion>"
}

If no issues found, return [].

Respond with JSON only — no prose, no markdown fences.

Article:
---
{article}
---
"""


def check_semantic(text: str, anthropic_client=None) -> List[Issue]:
    """Claude-based context-gap detection. Returns [] if no API key or client."""
    if anthropic_client is None:
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                return []
            anthropic_client = anthropic.Anthropic(api_key=api_key)
        except ImportError:
            return []

    prompt = SEMANTIC_PROMPT.replace("{article}", text)
    try:
        resp = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        # strip code fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
    except (Exception,) as e:
        print(f"[warn] semantic check failed: {e}", file=sys.stderr)
        return []

    out: List[Issue] = []
    for i, item in enumerate(data, start=1):
        out.append(Issue(
            id=f"s{i}",
            type=item.get("type", "context-gap"),
            location=f"Line {item.get('line_approx', '?')} (approx)",
            current=item.get("current", "").strip(),
            why=item.get("why", "").strip(),
            action=item.get("action", "").strip(),
        ))
    return out


# ─── Helpers ───────────────────────────────────────────────────

def _context_snippet(text: str, start: int, end: int, pad: int = 40) -> str:
    s = max(0, start - pad)
    e = min(len(text), end + pad)
    snippet = text[s:e].strip()
    # collapse whitespace
    snippet = re.sub(r"\s+", " ", snippet)
    if s > 0: snippet = "…" + snippet
    if e < len(text): snippet = snippet + "…"
    return snippet


# ─── Ledger emission ───────────────────────────────────────────

LEDGER_HEADER = """---
artifact: {artifact}
qa_run: context_qa v1
issue_types:
  placeholder:             ⚠ — unresolved placeholder / TODO / template text
  link-missing:            🔗 — named service without inline link
  acronym:                 🔤 — acronym without glossed first-use
  bare-url:                🔓 — URL not wrapped in markdown link syntax
  context-gap:             🔍 — proper noun or concept referenced without introduction
  jargon:                  📖 — technical term used without gloss
  first-use:               ⏳ — entity used before it's explained
  reference-inconsistency: ♻ — same entity called different names
status_values: [pending, reviewed, amended]
---

# Context QA Ledger — {artifact_name}

Generated automatically. Each issue uses the strict field format parseable by review_wizard.py.

Run the wizard:
    python review_wizard.py {ledger_path}

Loop: amend → re-run context_qa → until issue_count = 0.

---
"""


def write_ledger(issues: List[Issue], artifact_path: Path, ledger_path: Optional[Path] = None) -> str:
    """Serialize issues into a markdown review ledger."""
    artifact_name = artifact_path.stem
    ledger_rel = ledger_path.name if ledger_path else f"{artifact_name}_qa_ledger.md"

    lines = [LEDGER_HEADER.format(
        artifact=str(artifact_path),
        artifact_name=artifact_name,
        ledger_path=ledger_rel,
    )]

    # Group issues by type for predictable ordering
    type_order = [
        "placeholder", "link-missing", "acronym", "bare-url",
        "context-gap", "jargon", "first-use", "reference-inconsistency",
    ]
    by_type: dict[str, List[Issue]] = {}
    for iss in issues:
        by_type.setdefault(iss.type, []).append(iss)

    # Stable unique IDs across the ledger
    counter = 1
    for t in type_order:
        for iss in by_type.get(t, []):
            lines.append(f"## [q{counter}] {iss.type} · {iss.location}\n")
            lines.append(f"- **Status:** {iss.status}")
            lines.append(f"- **Location:** {iss.location}")
            lines.append(f"- **Current:** {iss.current}")
            lines.append(f"- **Why:** {iss.why}")
            lines.append(f"- **Action:** {iss.action}")
            lines.append(f"- **Amendment:** {iss.amendment}\n")
            counter += 1

    if counter == 1:
        lines.append("## 🎉 Zero issues found.\n")
        lines.append("The article passed context QA. No placeholder, link, acronym, or semantic gaps detected.\n")

    return "\n".join(lines)


# ─── Orchestration ─────────────────────────────────────────────

def run_qa(article_text: str, use_llm: bool = True, anthropic_client=None) -> List[Issue]:
    """Run all QA checks on an article and return the combined issue list."""
    issues: List[Issue] = []
    issues.extend(check_placeholders(article_text))
    issues.extend(check_links(article_text))
    issues.extend(check_acronyms(article_text))
    issues.extend(check_bare_urls(article_text))
    if use_llm:
        issues.extend(check_semantic(article_text, anthropic_client=anthropic_client))
    return issues


# ─── CLI ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Generate a context QA review ledger for an article.")
    ap.add_argument("article", type=Path, help="path to the article markdown file")
    ap.add_argument("-o", "--output", type=Path, help="write ledger to this path (default: stdout)")
    ap.add_argument("--no-llm", action="store_true", help="skip semantic LLM check (rules only)")
    ap.add_argument("--stats", action="store_true", help="print summary counts to stderr")
    args = ap.parse_args()

    if not args.article.exists():
        print(f"File not found: {args.article}", file=sys.stderr)
        sys.exit(1)

    text = args.article.read_text()
    issues = run_qa(text, use_llm=not args.no_llm)

    if args.stats:
        by_type: dict[str, int] = {}
        for iss in issues:
            by_type[iss.type] = by_type.get(iss.type, 0) + 1
        total = len(issues)
        print(f"QA complete — {total} issues", file=sys.stderr)
        for t, c in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {t:<28} {c:>3}", file=sys.stderr)

    out_path = args.output or Path(args.article.stem + "_qa_ledger.md")
    ledger = write_ledger(issues, args.article, ledger_path=out_path)
    if args.output:
        out_path.write_text(ledger)
        print(f"wrote ledger → {out_path}", file=sys.stderr)
    else:
        print(ledger)


if __name__ == "__main__":
    main()
