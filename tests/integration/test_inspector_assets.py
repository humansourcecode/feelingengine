"""Inspector static-asset sanity checks.

Verifies:
  - All three inspector files exist and are non-empty
  - HTML references to scripts/stylesheets exist
  - Every arc option in the dropdown has a matching bundle file
  - All cross-references from the inspector to arc bundles resolve
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent.parent
INSPECTOR_DIR = REPO_ROOT / "examples" / "inspector"
ARCS_DIR = REPO_ROOT / "examples" / "arcs"


def test_all_three_files_present():
    for name in ("index.html", "app.js", "style.css"):
        p = INSPECTOR_DIR / name
        assert p.exists(), f"missing {p}"
        assert p.stat().st_size > 0, f"empty {p}"


def test_html_references_its_local_assets():
    html = (INSPECTOR_DIR / "index.html").read_text()
    assert 'href="style.css"' in html
    assert 'src="app.js"' in html


def test_every_dropdown_option_has_matching_bundle():
    """For each `<option value="X">` in the arc-select dropdown, check that
    examples/arcs/X.json exists."""
    html = (INSPECTOR_DIR / "index.html").read_text()
    select_block = re.search(
        r'<select id="arc-select"[^>]*>(.*?)</select>', html, re.DOTALL)
    assert select_block, "could not find arc-select dropdown in HTML"
    values = re.findall(r'<option\s+value="([^"]+)"', select_block.group(1))
    assert values, "no options found in arc-select dropdown"
    for slug in values:
        bundle_path = ARCS_DIR / f"{slug}.json"
        assert bundle_path.exists(), (
            f"dropdown option '{slug}' has no matching bundle at {bundle_path}"
        )


def test_app_js_loads_arcs_from_expected_path():
    """The loader should point at `../arcs/{slug}.json`. Catches regressions
    where the data path gets moved without updating the inspector."""
    app_js = (INSPECTOR_DIR / "app.js").read_text()
    assert "../arcs/" in app_js, (
        "inspector no longer fetches from ../arcs/ — "
        "either the path changed or the loader was removed"
    )


def test_bundled_arcs_have_parseable_json():
    """The bundles the inspector will load must be parseable."""
    import json
    arc_files = list(ARCS_DIR.glob("*.json"))
    assert len(arc_files) >= 4, f"expected ≥4 bundled arcs, got {len(arc_files)}"
    for p in arc_files:
        try:
            json.loads(p.read_text())
        except json.JSONDecodeError as e:
            pytest.fail(f"{p} is not valid JSON: {e}")
