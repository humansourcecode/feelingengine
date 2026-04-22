"""Unit tests for rendering.platform_export — all three exporters."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from feeling_engine.rendering.analysis_assets import AnalysisBundle
from feeling_engine.rendering.platform_export import (
    export_substack,
    export_x_thread,
    export_youtube_script,
    export_all,
    _mmss,
)


def _build_bundle(tmp_path: Path) -> AnalysisBundle:
    """Construct a minimal AnalysisBundle without running the real renderer."""
    moments = [
        {
            "index": 1,
            "label": "body-turn",
            "start_sec": 10.0,
            "end_sec": 12.0,
            "peak_sec": 11.0,
            "intensity": 0.9,
            "confidence": 0.9,
            "score": 0.81,
            "evidence": "breath catch, inward turn",
            "in_sequence": "intimacy-deepening",
            "co_firing": ["vulnerability-transfer"],
            "clip_path": "clips/moment_01.mp4",
            "still_path": "stills/moment_01.jpg",
            "brain_path": "brains/moment_01.png",
        },
        {
            "index": 2,
            "label": "recognition",
            "start_sec": 30.0,
            "end_sec": 32.0,
            "peak_sec": 31.0,
            "intensity": 0.8,
            "confidence": 0.9,
            "score": 0.72,
            "evidence": "familiar phrase lands",
            "in_sequence": None,
            "co_firing": [],
            "clip_path": "clips/moment_02.mp4",
            "still_path": "stills/moment_02.jpg",
            "brain_path": "brains/moment_02.png",
        },
    ]
    return AnalysisBundle(
        video_source="https://example.com/videos/test-clip",
        duration_sec=60.0,
        total_labels=20,
        total_sequences=1,
        sequences=[{
            "name": "intimacy-deepening",
            "start_sec": 5.0, "end_sec": 35.0,
            "matched_labels": ["body-turn", "recognition"],
            "positions": [10, 30],
            "partial": False,
        }],
        key_moments=moments,
        mechanism_counts={"body-turn": 4, "recognition": 3, "drift": 2},
        out_dir=str(tmp_path),
    )


# ─── helpers ────────────────────────────────────────────────────

def test_mmss_formats_correctly():
    assert _mmss(0) == "0:00"
    assert _mmss(59) == "0:59"
    assert _mmss(60) == "1:00"
    assert _mmss(125.5) == "2:06"  # 125.5 rounds to 126 = 2:06


# ─── Substack ────────────────────────────────────────────────────

def test_export_substack_writes_article_md(tmp_path):
    bundle = _build_bundle(tmp_path)
    out = export_substack(bundle)
    assert out == tmp_path / "article.md"
    assert out.exists()

    text = out.read_text()
    # Title and opening
    assert "# Mechanism analysis" in text
    assert "60.0-minute piece" not in text  # it's 60 sec → 1.0 min
    assert "1.0-minute piece" in text
    # Moment headings
    assert "## Moment 1 — body-turn @ 0:11" in text
    assert "## Moment 2 — recognition @ 0:31" in text
    # Media references
    assert "clips/moment_01.mp4" in text
    assert "brains/moment_01.png" in text
    # Mechanism counts
    assert "| body-turn | 4 |" in text
    assert "| drift | 2 |" in text
    # Sequence block
    assert "intimacy-deepening" in text


def test_export_substack_respects_custom_out_path(tmp_path):
    bundle = _build_bundle(tmp_path)
    custom = tmp_path / "custom.md"
    out = export_substack(bundle, out_path=custom)
    assert out == custom
    assert custom.exists()


def test_export_substack_includes_evidence(tmp_path):
    bundle = _build_bundle(tmp_path)
    text = export_substack(bundle).read_text()
    assert "breath catch, inward turn" in text
    assert "familiar phrase lands" in text


# ─── X thread ────────────────────────────────────────────────────

def test_export_x_thread_writes_valid_json(tmp_path):
    bundle = _build_bundle(tmp_path)
    out = export_x_thread(bundle)
    assert out == tmp_path / "thread.json"
    data = json.loads(out.read_text())

    assert data["platform"] == "x"
    assert "posts" in data
    posts = data["posts"]
    # opening + 2 moments + closing = 4
    assert len(posts) == 4
    # index is sequential, 1-based
    assert [p["index"] for p in posts] == [1, 2, 3, 4]
    # Opening post contains thread hook
    assert "Feeling Engine" in posts[0]["text"]
    # Moment posts reference their labels
    assert "body-turn" in posts[1]["text"]
    assert "recognition" in posts[2]["text"]
    # Closing post contains video URL + CTA
    assert "example.com" in posts[3]["text"]
    assert "reply" in posts[3]["text"].lower()


def test_export_x_thread_attaches_media_refs_to_moment_posts(tmp_path):
    bundle = _build_bundle(tmp_path)
    data = json.loads(export_x_thread(bundle).read_text())
    moment_post = data["posts"][1]  # first moment post
    media_types = {m["type"] for m in moment_post["media"]}
    assert "image" in media_types  # still + brain
    assert "video" in media_types  # clip
    # Each media item has a path
    for m in moment_post["media"]:
        assert "path" in m


def test_export_x_thread_respects_max_posts(tmp_path):
    bundle = _build_bundle(tmp_path)
    # 2 moments → opening + 2 moment posts + closing = 4, well under max_posts=10
    data = json.loads(export_x_thread(bundle, max_posts=3).read_text())
    # max_posts=3 means only 1 moment post fits (opening + 1 + closing = 3)
    assert len(data["posts"]) == 3


# ─── YouTube script ──────────────────────────────────────────────

def test_export_youtube_script_writes_markdown(tmp_path):
    bundle = _build_bundle(tmp_path)
    out = export_youtube_script(bundle)
    assert out == tmp_path / "video_script.md"
    text = out.read_text()

    assert "# OBS session plan" in text
    # Per-moment sections
    assert "### Moment 1 — body-turn @ 0:11" in text
    assert "### Moment 2 — recognition @ 0:31" in text
    # Action cues
    assert "Play clip: `clips/moment_01.mp4`" in text
    assert "Show brain image: `brains/moment_01.png`" in text
    # Narration cues with evidence
    assert "breath catch, inward turn" in text
    # Outro references sequences
    assert "intimacy-deepening" in text


# ─── Orchestrator ────────────────────────────────────────────────

def test_export_all_runs_all_three_by_default(tmp_path):
    bundle = _build_bundle(tmp_path)
    results = export_all(bundle)
    assert set(results) == {"substack", "x", "youtube"}
    for path in results.values():
        assert path.exists()


def test_export_all_can_be_restricted(tmp_path):
    bundle = _build_bundle(tmp_path)
    results = export_all(bundle, platforms=["substack"])
    assert set(results) == {"substack"}
    assert (tmp_path / "article.md").exists()
    assert not (tmp_path / "thread.json").exists()


def test_export_all_ignores_unknown_platforms(tmp_path):
    bundle = _build_bundle(tmp_path)
    results = export_all(bundle, platforms=["substack", "bogus"])
    assert set(results) == {"substack"}


def test_exporters_accept_dict_bundle(tmp_path):
    """Callers that only have the JSON file (no dataclass) should still work."""
    bundle = _build_bundle(tmp_path)
    dict_bundle = bundle.to_dict()
    # dict bundle should produce the same output
    out = export_substack(dict_bundle)
    assert out.exists()
    assert "body-turn" in out.read_text()
