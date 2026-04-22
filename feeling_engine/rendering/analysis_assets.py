"""Analysis bundle — structured output + extracted clips, stills, brain images.

Given a video and its mechanism arc (from any detector — LLM, Tier 1+2/3, etc.),
produces a platform-agnostic asset bundle that downstream renderers (Substack
article, YouTube script, TikTok cuts, X thread) can consume without re-running
analysis.

Output layout:

    <out_dir>/
        analysis.json       # full structured analysis
        clips/              # 10-15s mp4 cuts at each key moment
        stills/             # representative jpg frames
        brains/             # canonical brain renders per key moment

Key moments are selected by ranking firings on intensity × confidence + a
preference for firings that contribute to named sequences, with temporal
spread enforced so selections aren't clustered.

Commercial-safe: uses the arc + local ffmpeg + our brain renderer.
No TRIBE at runtime.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from feeling_engine.mechanisms.vocabulary import (
    LabelApplication, SequenceMatch,
)
from feeling_engine.rendering.brain_renderer import render_mechanism_brain


DEFAULT_CLIP_PRE_SEC = 3.0
DEFAULT_CLIP_POST_SEC = 2.0
DEFAULT_KEY_MOMENTS = 5


@dataclass
class KeyMoment:
    """A selected salient moment worth rendering as article/video/clip/brain."""
    index: int                     # 1-based
    label: str                     # dominant mechanism at this moment
    start_sec: float
    end_sec: float
    peak_sec: float                # representative timestamp (midpoint by default)
    intensity: float
    confidence: float
    score: float                   # intensity * confidence * bonuses
    evidence: str                  # from the detector
    in_sequence: Optional[str] = None   # name of named sequence, if any
    co_firing: list = field(default_factory=list)  # labels firing in the same window
    clip_path: Optional[str] = None
    still_path: Optional[str] = None
    brain_path: Optional[str] = None


@dataclass
class AnalysisBundle:
    """Everything a downstream platform renderer needs."""
    video_source: str                       # path or URL
    duration_sec: float
    total_labels: int
    total_sequences: int
    sequences: list                         # list of SequenceMatch dicts
    key_moments: list                       # list of KeyMoment dicts
    mechanism_counts: dict                  # label → firing count
    out_dir: str

    def to_dict(self) -> dict:
        return asdict(self)


# ─── Key-moment selection ────────────────────────────────────────────

def _score_firing(app: LabelApplication, in_seq_names: set) -> float:
    """Rank-order score for a single LabelApplication.

    intensity × confidence, with +20% bonus if the firing contributes to any
    named sequence. Tier-2/3 firings are slightly deprioritized vs Tier 1
    because Tier 1 are brain-grounded.
    """
    base = app.intensity * app.confidence
    if app.label in in_seq_names:
        base *= 1.2
    return base


def _labels_in_sequences(sequences) -> set:
    """Collect all label names that appear in any detected sequence."""
    names = set()
    for s in sequences:
        for lbl in getattr(s, "matched_labels", []) or []:
            names.add(lbl)
    return names


def _enforce_temporal_spread(
    sorted_candidates: list,
    duration_sec: float,
    k: int,
) -> list:
    """Pick k candidates with at least min_gap between selections.

    Greedy: iterate the score-sorted list, accept each candidate unless it
    falls within min_gap of an already-selected timestamp. If fewer than k
    survive, relax the gap and fill.
    """
    if duration_sec <= 0 or not sorted_candidates:
        return sorted_candidates[:k]

    min_gap = duration_sec / (k * 2)
    selected = []
    for cand in sorted_candidates:
        peak = (cand.start_sec + cand.end_sec) / 2
        if all(abs(peak - ((s.start_sec + s.end_sec) / 2)) >= min_gap
               for s in selected):
            selected.append(cand)
            if len(selected) == k:
                return selected

    # Fallback: fill remaining slots without gap constraint
    for cand in sorted_candidates:
        if cand not in selected:
            selected.append(cand)
            if len(selected) == k:
                break
    return selected


def extract_key_moments(
    arc: list,
    sequences: list,
    duration_sec: float,
    n: int = DEFAULT_KEY_MOMENTS,
) -> list:
    """Return top-n key moments from a mechanism arc, with temporal spread.

    Args:
        arc: list of LabelApplication
        sequences: list of SequenceMatch
        duration_sec: total video duration
        n: max number of key moments

    Returns:
        list of KeyMoment (indexed 1..n, sorted by start_sec)
    """
    if not arc:
        return []

    seq_label_names = _labels_in_sequences(sequences)
    seq_ranges = [(s.name, s.start_sec, s.end_sec) for s in sequences]

    scored = [(app, _score_firing(app, seq_label_names)) for app in arc]
    scored.sort(key=lambda x: -x[1])
    sorted_apps = [app for app, _ in scored]

    chosen = _enforce_temporal_spread(sorted_apps, duration_sec, n)
    chosen.sort(key=lambda a: a.start_sec)

    moments = []
    for i, app in enumerate(chosen, 1):
        peak = (app.start_sec + app.end_sec) / 2
        # Which named sequence (if any) contains this moment?
        in_seq = None
        for name, s, e in seq_ranges:
            if s <= peak <= e:
                in_seq = name
                break
        # Which other labels fire in the same ~2-second window?
        co_firing = [
            other.label for other in arc
            if other is not app
            and abs(((other.start_sec + other.end_sec) / 2) - peak) <= 2.0
        ]
        moments.append(KeyMoment(
            index=i,
            label=app.label,
            start_sec=app.start_sec,
            end_sec=app.end_sec,
            peak_sec=peak,
            intensity=app.intensity,
            confidence=app.confidence,
            score=_score_firing(app, seq_label_names),
            evidence=str(app.signals.get("evidence", "")) if app.signals else "",
            in_sequence=in_seq,
            co_firing=co_firing,
        ))
    return moments


# ─── Asset extraction via ffmpeg ─────────────────────────────────────

def _probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip()) if result.stdout.strip() else 0.0


def extract_clip(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    out_path: Path,
    pre_sec: float = DEFAULT_CLIP_PRE_SEC,
    post_sec: float = DEFAULT_CLIP_POST_SEC,
) -> Path:
    """Extract a clip around a moment, with lead-in / lead-out padding."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    start = max(0.0, start_sec - pre_sec)
    end = end_sec + post_sec
    dur = max(0.5, end - start)
    # re-encode for seekable output
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-ss", f"{start:.3f}",
         "-i", str(video_path),
         "-t", f"{dur:.3f}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "22",
         "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart",
         str(out_path)],
        check=True, capture_output=True,
    )
    return out_path


def extract_still(
    video_path: Path,
    at_sec: float,
    out_path: Path,
) -> Path:
    """Extract a single JPEG frame at the given timestamp."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-ss", f"{at_sec:.3f}",
         "-i", str(video_path),
         "-vframes", "1",
         "-q:v", "3",
         str(out_path)],
        check=True, capture_output=True,
    )
    return out_path


# ─── Bundle builder ──────────────────────────────────────────────────

def render_analysis_bundle(
    video_path,
    arc: list,
    out_dir,
    sequences: Optional[list] = None,
    n_key_moments: int = DEFAULT_KEY_MOMENTS,
    source_url: Optional[str] = None,
    clip_pre_sec: float = DEFAULT_CLIP_PRE_SEC,
    clip_post_sec: float = DEFAULT_CLIP_POST_SEC,
    render_brains: bool = True,
    verbose: bool = True,
) -> AnalysisBundle:
    """Build a full asset bundle from a video + mechanism arc.

    Args:
        video_path: path to the source video/audio file.
        arc: list of LabelApplication (from any detector).
        out_dir: directory to populate. Will be created if missing.
        sequences: optional list of SequenceMatch. If None, none attached.
        n_key_moments: how many key moments to extract (default 5).
        source_url: optional URL pointer for the content.
        render_brains: if True, render canonical brain PNG per key moment.
        verbose: progress logging.

    Returns:
        AnalysisBundle with paths populated.
    """
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = out_dir / "clips"
    stills_dir = out_dir / "stills"
    brains_dir = out_dir / "brains"

    sequences = sequences or []
    duration = _probe_duration(video_path)

    if verbose:
        print(f"  Video: {video_path.name} ({duration:.1f}s), "
              f"arc={len(arc)} apps, sequences={len(sequences)}", flush=True)

    moments = extract_key_moments(arc, sequences, duration, n=n_key_moments)
    if verbose:
        print(f"  Key moments selected: {len(moments)}", flush=True)

    for m in moments:
        stem = f"moment_{m.index:02d}"
        if verbose:
            print(f"    [{m.index}] {m.label} @ {m.peak_sec:.1f}s "
                  f"(i={m.intensity:.2f}, c={m.confidence:.2f})", flush=True)

        clip_path = clips_dir / f"{stem}.mp4"
        extract_clip(video_path, m.start_sec, m.end_sec, clip_path,
                     pre_sec=clip_pre_sec, post_sec=clip_post_sec)
        m.clip_path = str(clip_path.relative_to(out_dir))

        still_path = stills_dir / f"{stem}.jpg"
        extract_still(video_path, m.peak_sec, still_path)
        m.still_path = str(still_path.relative_to(out_dir))

        if render_brains:
            brain_path = brains_dir / f"{stem}.png"
            render_mechanism_brain(
                label=m.label,
                output_path=brain_path,
                view="lateral_both",
                intensity=m.intensity,
                title=f"{m.label} @ {m.peak_sec:.1f}s",
            )
            m.brain_path = str(brain_path.relative_to(out_dir))

    # Full mechanism distribution
    mechanism_counts: dict = {}
    for app in arc:
        mechanism_counts[app.label] = mechanism_counts.get(app.label, 0) + 1

    bundle = AnalysisBundle(
        video_source=source_url or str(video_path),
        duration_sec=duration,
        total_labels=len(arc),
        total_sequences=len(sequences),
        sequences=[asdict(s) for s in sequences],
        key_moments=[asdict(m) for m in moments],
        mechanism_counts=dict(sorted(
            mechanism_counts.items(), key=lambda x: -x[1],
        )),
        out_dir=str(out_dir),
    )

    analysis_json_path = out_dir / "analysis.json"
    analysis_json_path.write_text(json.dumps(bundle.to_dict(), indent=2))
    if verbose:
        print(f"  Wrote {analysis_json_path}", flush=True)

    return bundle
