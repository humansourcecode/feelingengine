"""Apply QA-ledger amendments back into an article.

For each issue in the ledger with status='amended', extract the
replacement text from the Amendment field and replace the Current text
in the article.

Amendment-field parsing supports two forms:

    - "Replace with: <text>"         → replaces Current with <text>
    - "Replace with: \"<text>\""      → replaces Current with <text>
    - "Remove"                         → deletes Current from article

Anything in [square brackets] at the end of Amendment is treated as
meta / reviewer note and discarded from the replacement text.

Usage:
    python -m feeling_engine.qa.apply_amendments <ledger.md> <article.md>
    python -m feeling_engine.qa.apply_amendments <ledger.md> <article.md> --dry-run
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


FIELD_RE = re.compile(r"^- \*\*(?P<key>[^:]+):\*\* ?(?P<value>.*)$")
ISSUE_HEADER_RE = re.compile(r"^## \[(?P<id>[^\]]+)\] (?P<type>[a-z-]+)")


@dataclass
class LedgerIssue:
    id: str
    type: str
    status: str = "pending"
    current: str = ""
    amendment: str = ""


def parse_ledger(path: Path) -> List[LedgerIssue]:
    """Parse a review ledger, extracting issues needing application."""
    issues: List[LedgerIssue] = []
    current: Optional[LedgerIssue] = None
    cur_key: Optional[str] = None

    for line in path.read_text().splitlines():
        hdr = ISSUE_HEADER_RE.match(line)
        if hdr:
            if current is not None:
                issues.append(current)
            current = LedgerIssue(id=hdr.group("id"), type=hdr.group("type"))
            cur_key = None
            continue
        if current is None:
            continue
        if line.startswith("## ") or line.startswith("# "):
            issues.append(current)
            current = None
            cur_key = None
            continue
        fld = FIELD_RE.match(line)
        if fld:
            cur_key = fld.group("key").strip()
            val = fld.group("value")
            if cur_key == "Status":
                current.status = val.strip().lower()
            elif cur_key == "Current":
                current.current = val
            elif cur_key == "Amendment":
                current.amendment = val
        elif cur_key and line.strip():
            # continuation of multi-line field
            if cur_key == "Current":
                current.current += " " + line.strip()
            elif cur_key == "Amendment":
                current.amendment += " " + line.strip()

    if current is not None:
        issues.append(current)
    return issues


# ─── Amendment parsing ─────────────────────────────────────────

REPLACE_PATTERNS = [
    # "Replace with: \"<quoted content>\" [notes]"
    re.compile(r'^Replace with:\s*"(?P<text>[^"]+)"\s*(?:\[.*\])?\s*$', re.DOTALL),
    # "Replace with: <unquoted content> [notes]"
    re.compile(r'^Replace with:\s*(?P<text>.+?)(?:\s*\[[^\]]+\])?\s*$', re.DOTALL),
]

REMOVE_KEYWORD = re.compile(r"^\s*Remove", re.IGNORECASE)


def parse_amendment(amendment: str) -> Optional[tuple[str, Optional[str]]]:
    """Return (action, replacement_text) — action is 'replace' or 'remove'.
    Returns None if the amendment couldn't be parsed."""
    amendment = amendment.strip()
    if not amendment:
        return None
    if REMOVE_KEYWORD.match(amendment):
        return ("remove", None)
    for pat in REPLACE_PATTERNS:
        m = pat.match(amendment)
        if m:
            text = m.group("text").strip()
            # strip enclosing quotes if present
            if text.startswith('"') and text.endswith('"'):
                text = text[1:-1]
            return ("replace", text)
    return None


# ─── Article editing ───────────────────────────────────────────

def apply_to_article(article_text: str, issues: List[LedgerIssue],
                     dry_run: bool = False) -> tuple[str, List[dict]]:
    """Apply amended issues to article text. Returns (new_text, report)."""
    new_text = article_text
    report: List[dict] = []

    for iss in issues:
        if iss.status != "amended":
            continue
        parsed = parse_amendment(iss.amendment)
        if parsed is None:
            report.append({"id": iss.id, "type": iss.type, "result": "unparseable-amendment",
                           "detail": iss.amendment[:80]})
            continue

        action, replacement = parsed
        current = iss.current.strip()

        if action == "remove":
            if current in new_text:
                if not dry_run:
                    new_text = new_text.replace(current, "", 1)
                report.append({"id": iss.id, "type": iss.type, "result": "removed"})
            else:
                report.append({"id": iss.id, "type": iss.type, "result": "current-not-found",
                               "detail": current[:80]})
            continue

        # action == "replace"
        if current and current in new_text:
            if not dry_run:
                new_text = new_text.replace(current, replacement, 1)
            report.append({"id": iss.id, "type": iss.type, "result": "replaced"})
        else:
            report.append({"id": iss.id, "type": iss.type, "result": "current-not-found",
                           "detail": f"Looked for: {current[:80]}"})

    return new_text, report


# ─── CLI ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Apply ledger amendments back into article.")
    ap.add_argument("ledger", type=Path, help="path to the QA or review ledger")
    ap.add_argument("article", type=Path, help="path to the article file to edit")
    ap.add_argument("--dry-run", action="store_true", help="report what would change, no write")
    ap.add_argument("-o", "--output", type=Path, help="write updated article to this path (default: overwrite)")
    args = ap.parse_args()

    if not args.ledger.exists():
        print(f"Ledger not found: {args.ledger}", file=sys.stderr); sys.exit(1)
    if not args.article.exists():
        print(f"Article not found: {args.article}", file=sys.stderr); sys.exit(1)

    issues = parse_ledger(args.ledger)
    article_text = args.article.read_text()
    new_text, report = apply_to_article(article_text, issues, dry_run=args.dry_run)

    # Print report to stderr
    counts = {"replaced": 0, "removed": 0, "current-not-found": 0, "unparseable-amendment": 0}
    for r in report:
        counts[r["result"]] = counts.get(r["result"], 0) + 1

    print(f"\nAmendment apply report — {len(report)} amended issues processed", file=sys.stderr)
    print(f"  {'replaced':<30} {counts.get('replaced', 0):>3}", file=sys.stderr)
    print(f"  {'removed':<30} {counts.get('removed', 0):>3}", file=sys.stderr)
    print(f"  {'current-not-found':<30} {counts.get('current-not-found', 0):>3}", file=sys.stderr)
    print(f"  {'unparseable-amendment':<30} {counts.get('unparseable-amendment', 0):>3}", file=sys.stderr)

    # Detail failures
    for r in report:
        if r["result"] in ("current-not-found", "unparseable-amendment"):
            print(f"  [{r['id']}] {r['result']}: {r.get('detail', '')}", file=sys.stderr)

    if not args.dry_run:
        out = args.output or args.article
        out.write_text(new_text)
        print(f"\nWrote updated article → {out}", file=sys.stderr)
    else:
        print(f"\n[dry-run] no changes written", file=sys.stderr)


if __name__ == "__main__":
    main()
