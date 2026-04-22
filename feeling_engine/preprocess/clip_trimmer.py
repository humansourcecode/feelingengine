"""Clip trimmer — remove non-scene content via dense per-frame classification.

YouTube clips often carry channel intros, outros, subscribe overlays, or
commentary bookends that pollute mechanism detection (a branded end card
reads as body-surge + pattern-break; a "SUBSCRIBE" outro fires fake release).

This module samples frames at regular intervals through the video, asks Gemini
to classify each frame as scene vs non-scene (branded / title / commentary /
ad / transition), smooths single-frame noise, and ffmpeg-concats the scene
regions. No segmentation step — no reliance on scene-cut detection, no
mixed-segment problem, no "1 segment" failure mode.

Commercial-safe: Gemini API + ffmpeg. No TRIBE.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


GEMINI_CLASSIFY_MODEL_DEFAULT = "gemini-2.5-flash"
DEFAULT_SAMPLE_INTERVAL = 2.0     # seconds between sampled frames
DEFAULT_MIN_NONSCENE_RUN = 2      # min contiguous non-scene frames to trim
DEFAULT_CLASSIFY_CHUNK = 10       # frames per Gemini call — small batches
                                  # localize PROHIBITED_CONTENT blocks
SCENE_LABEL = "scene"
NON_SCENE_LABELS = {"branded", "title", "commentary", "ad", "transition"}


@dataclass
class RemoveSegment:
    """One non-content segment to remove."""
    start_sec: float
    end_sec: float
    reason: str


@dataclass
class TrimResult:
    """Metadata about one trim operation."""
    original_path: str
    trimmed_path: str
    original_duration: float
    trimmed_duration: float
    removed_segments: list = field(default_factory=list)
    kept_ranges: list = field(default_factory=list)
    model_used: str = ""
    token_usage_in: int = 0
    token_usage_out: int = 0
    cost_usd: float = 0.0
    trimmed_ratio: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


# ─── ffprobe / ffmpeg helpers ─────────────────────────────────────

def _probe_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(video_path)],
        capture_output=True, text=True, timeout=30,
    )
    return float(result.stdout.strip()) if result.stdout.strip() else 0.0


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n", "", text)
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def _ffmpeg_copy_through(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-i", str(src), "-c", "copy", str(dst)],
        check=True, capture_output=True,
    )


def _ffmpeg_single_range(src: Path, dst: Path, start: float, end: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end - start)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-ss", f"{start:.3f}",
         "-i", str(src),
         "-t", f"{duration:.3f}",
         "-c:v", "libx264", "-preset", "fast", "-crf", "22",
         "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart",
         str(dst)],
        check=True, capture_output=True,
    )


def _ffmpeg_multi_range_concat(src: Path, dst: Path, keeps: list) -> None:
    """Cut multiple keep ranges from a single source and concat into one output."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = len(keeps)
    parts = []
    for i, (s, e) in enumerate(keeps):
        parts.append(
            f"[0:v]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{i}];"
            f"[0:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS[a{i}]"
        )
    concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
    parts.append(f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]")
    filtergraph = ";".join(parts)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-i", str(src),
         "-filter_complex", filtergraph,
         "-map", "[outv]", "-map", "[outa]",
         "-c:v", "libx264", "-preset", "fast", "-crf", "22",
         "-c:a", "aac", "-b:a", "128k",
         "-movflags", "+faststart",
         str(dst)],
        check=True, capture_output=True,
    )


# ─── Dense frame sampling ────────────────────────────────────────

def _sample_frames_at_interval(
    video_path: Path, interval: float, out_dir: Path,
) -> list:
    """Extract one frame every `interval` seconds via a single ffmpeg call.

    Returns [(timestamp_sec, frame_path), ...] sorted by timestamp.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-i", str(video_path),
         "-vf", f"fps=1/{interval}",
         "-q:v", "4",
         str(out_dir / "frame_%04d.jpg")],
        check=True, capture_output=True, timeout=600,
    )
    frames = []
    for fp in sorted(out_dir.glob("frame_*.jpg")):
        m = re.search(r"frame_(\d+)\.jpg", fp.name)
        if not m:
            continue
        idx = int(m.group(1)) - 1   # ffmpeg numbers from 1
        ts = idx * interval
        frames.append((ts, fp))
    return frames


# ─── Gemini classification ───────────────────────────────────────

_CLASSIFY_PROMPT = """You are classifying frames from a {content_type} video.

Each frame is tagged with its timestamp. Classify EACH frame as exactly ONE of:

- "scene": actual {content_type} content — dialogue, action, performance, atmospheric cinematography, dark or quiet moments that are part of the cinematic content. A dark or near-empty scene frame WITHOUT text/logo/template overlay is "scene" (atmospheric darkness is content).
- "branded": channel-branded template — YouTube end cards, subscribe overlays, channel logos, horror-channel plates (FEAR, Fandango Movieclips outros), thumbnail-link grids, designed templates with subscribe buttons.
- "title": static title cards, movie-title reveals, credits, copyright notices, rating screens, solid-color blank cards, studio splash screens.
- "commentary": talking-head analysis, reaction overlays, on-screen commentary layout.
- "ad": sponsor advertisement, product placement card, pre-roll ad frame.
- "transition": brief bumpers or designed transitional visuals between scenes.

Hard rules:
1. A frame showing text overlays, subscribe buttons, or branded templates is "branded" or "title" regardless of cinematic music under it.
2. A dark frame WITHOUT text/logo is "scene", not "title".
3. Classify based on the visual content of the frame, not what you think the video is about.

Return JSON: {{"classifications": [{{"frame_index": N, "label": "scene|branded|title|commentary|ad|transition"}}, ...]}}

One entry per input frame, in input order. No prose outside the JSON. No markdown fences.
"""


def _classify_frames_chunk(
    client, chunk, content_type, model, thinking_budget, verbose,
):
    """Classify one small batch of frames. Raises on block; returns
    (labels_by_idx, in_tok, out_tok)."""
    from google.genai import types
    from feeling_engine._gemini_retry import gemini_with_retry

    contents = [_CLASSIFY_PROMPT.format(content_type=content_type)]
    for global_idx, ts, fp in chunk:
        contents.append(f"\nFrame {global_idx}: t={ts:.1f}s")
        contents.append(types.Part.from_bytes(
            mime_type="image/jpeg", data=fp.read_bytes(),
        ))
    contents.append("\nClassify each frame. JSON only.")

    safety = [
        types.SafetySetting(
            category=c,
            threshold=types.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        )
        for c in (
            types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        )
    ]

    response = gemini_with_retry(
        client.models.generate_content,
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            max_output_tokens=2000,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            safety_settings=safety,
        ),
        verbose=False,
    )

    if not response.candidates:
        reason = getattr(response, "prompt_feedback", None)
        raise RuntimeError(f"blocked: {reason}")
    candidate = response.candidates[0]
    if candidate.content is None or candidate.content.parts is None:
        finish = getattr(candidate, "finish_reason", None)
        raise RuntimeError(f"empty (finish_reason={finish})")

    text = ""
    for part in candidate.content.parts:
        if hasattr(part, "text") and part.text:
            text += part.text
    text = _strip_code_fences(text)
    parsed = json.loads(text)
    entries = parsed.get("classifications", []) if isinstance(parsed, dict) else parsed
    if not isinstance(entries, list):
        raise ValueError("Expected 'classifications' array")

    labels = {}
    for ent in entries:
        try:
            idx = int(ent["frame_index"])
            label = str(ent.get("label", SCENE_LABEL)).strip().lower()
        except (KeyError, TypeError, ValueError):
            continue
        labels[idx] = label

    usage = response.usage_metadata
    return labels, (usage.prompt_token_count or 0), (usage.candidates_token_count or 0)


def _classify_frames(
    client,
    frames: list,
    content_type: str,
    model: str,
    verbose: bool = True,
    thinking_budget: Optional[int] = None,
    chunk_size: int = DEFAULT_CLASSIFY_CHUNK,
) -> tuple:
    """Classify all sampled frames in small batches.

    Small batches (default 10 frames) localize Gemini's non-configurable
    PROHIBITED_CONTENT blocks — if one frame trips the hard filter, only
    that chunk fails, and the other chunks still produce labels. Failed
    chunks default their frames to "scene" (bias-to-keep).

    Returns (labels_by_idx, in_tok, out_tok).
    """
    if thinking_budget is None:
        thinking_budget = 0 if "flash" in model.lower() else 2048

    # Enumerate global indices once, then chunk
    indexed = [(i, ts, fp) for i, (ts, fp) in enumerate(frames)]
    chunks = [indexed[i:i + chunk_size] for i in range(0, len(indexed), chunk_size)]

    if verbose:
        print(f"  Classifying {len(frames)} frame(s) in {len(chunks)} "
              f"chunk(s) via {model}...", end="", flush=True)
    t0 = time.time()

    all_labels = {}
    total_in = 0
    total_out = 0
    blocked_chunks = 0
    for ci, chunk in enumerate(chunks):
        try:
            labels, in_tok, out_tok = _classify_frames_chunk(
                client, chunk, content_type, model, thinking_budget, verbose,
            )
            all_labels.update(labels)
            total_in += in_tok
            total_out += out_tok
        except (RuntimeError, ValueError, json.JSONDecodeError) as e:
            blocked_chunks += 1
            # Default to scene for this chunk — bias to keep content
            for global_idx, _, _ in chunk:
                all_labels[global_idx] = SCENE_LABEL
            if verbose:
                print(f"\n    ⚠ chunk {ci+1}/{len(chunks)} "
                      f"({len(chunk)} frames) → scene fallback ({e})",
                      end="", flush=True)

    if verbose:
        blocked_note = f", {blocked_chunks} chunk(s) blocked" if blocked_chunks else ""
        print(f"\n  done in {time.time()-t0:.0f}s, "
              f"{total_in:,} in / {total_out:,} out{blocked_note}", flush=True)
    return all_labels, total_in, total_out


# ─── Label smoothing + range construction ────────────────────────

def _smooth_frame_labels(labels: list, min_nonscene_run: int) -> list:
    """Replace isolated non-scene frames with their surrounding scene label.

    Prevents a single mislabeled frame (e.g. a briefly-dark scene moment
    tagged "title") from producing a spurious trim. Non-scene runs shorter
    than `min_nonscene_run` get flipped back to "scene".

    Scene runs are never squashed — the asymmetry is deliberate: we bias
    toward keeping content when ambiguous.
    """
    n = len(labels)
    if n == 0:
        return labels
    smoothed = list(labels)
    i = 0
    while i < n:
        if smoothed[i] == SCENE_LABEL:
            i += 1
            continue
        j = i
        while j < n and smoothed[j] != SCENE_LABEL:
            j += 1
        run_len = j - i
        if run_len < min_nonscene_run:
            # Short non-scene island → convert to scene
            for k in range(i, j):
                smoothed[k] = SCENE_LABEL
        i = j
    return smoothed


def _frame_labels_to_removes(
    frames: list, labels: list, duration: float,
) -> list:
    """From per-frame labels build RemoveSegment objects for non-scene runs.

    Boundaries fall at the midpoint between differently-labeled frames.
    """
    removes = []
    n = len(frames)
    i = 0
    while i < n:
        if labels[i] == SCENE_LABEL:
            i += 1
            continue
        # Start of a non-scene run
        run_start_idx = i
        run_label = labels[i]
        # All reasons we pick up during this run
        reasons = {labels[i]}
        j = i
        while j < n and labels[j] != SCENE_LABEL:
            reasons.add(labels[j])
            j += 1
        # Time boundary BEFORE this run
        if run_start_idx == 0:
            boundary_start = 0.0
        else:
            boundary_start = (frames[run_start_idx - 1][0] + frames[run_start_idx][0]) / 2.0
        # Time boundary AFTER this run
        if j >= n:
            boundary_end = duration
        else:
            boundary_end = (frames[j - 1][0] + frames[j][0]) / 2.0
        removes.append(RemoveSegment(
            start_sec=max(0.0, boundary_start),
            end_sec=min(duration, boundary_end),
            reason="+".join(sorted(reasons)),
        ))
        i = j
    return removes


def _merge_overlapping(segments: list) -> list:
    """Sort by start, merge overlapping / touching segments."""
    if not segments:
        return []
    ordered = sorted(segments, key=lambda s: s.start_sec)
    merged = [ordered[0]]
    for s in ordered[1:]:
        last = merged[-1]
        if s.start_sec <= last.end_sec:
            merged[-1] = RemoveSegment(
                start_sec=last.start_sec,
                end_sec=max(last.end_sec, s.end_sec),
                reason=f"{last.reason}; {s.reason}".strip("; "),
            )
        else:
            merged.append(s)
    return merged


def _compute_keep_ranges(duration: float, removes: list) -> list:
    """Given sorted non-overlapping removes, return [(start, end), ...] to keep."""
    if not removes:
        return [(0.0, duration)]
    keeps = []
    cursor = 0.0
    for r in removes:
        if r.start_sec > cursor:
            keeps.append((cursor, r.start_sec))
        cursor = r.end_sec
    if cursor < duration:
        keeps.append((cursor, duration))
    return keeps


# ─── Public API ──────────────────────────────────────────────────

def trim_by_dense_classification(
    video_path,
    output_path,
    content_type: str = "movie scene",
    sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
    min_nonscene_run: int = DEFAULT_MIN_NONSCENE_RUN,
    model: Optional[str] = None,
    thinking_budget: Optional[int] = None,
    verbose: bool = True,
) -> TrimResult:
    """Trim non-scene content via dense per-frame classification.

    Samples frames every `sample_interval` seconds. Sends all frames to Gemini
    in one batch for classification. Smooths single-frame noise. ffmpeg-concats
    the scene regions.

    Args:
        video_path: input video.
        output_path: where to write the trimmed mp4.
        content_type: what the video is of — "movie scene", "pitch", "speech".
        sample_interval: seconds between sampled frames (default 2.0). Smaller
            = better boundary precision, more cost.
        min_nonscene_run: minimum contiguous non-scene frames to actually trim
            (default 2). A single frame flip is treated as noise. With 2s
            sample interval this means trim only for ≥ ~4s of non-scene.
        model: classifier Gemini model (default gemini-2.5-flash).
        thinking_budget: explicit thinking token budget.
        verbose: progress logs.

    Returns:
        TrimResult with removed segments, kept ranges, cost.
    """
    from google import genai

    video_path = Path(video_path)
    output_path = Path(output_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    api_key = os.environ.get("GOOGLE_AI_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_AI_API_KEY not set")

    model_name = model or GEMINI_CLASSIFY_MODEL_DEFAULT
    duration = _probe_duration(video_path)
    if verbose:
        print(f"  {video_path.name}: duration={duration:.1f}s", flush=True)

    # 1. Sample frames densely
    tmpdir = Path(tempfile.mkdtemp(prefix="trim_dense_frames_"))
    try:
        frames = _sample_frames_at_interval(video_path, sample_interval, tmpdir)
        if verbose:
            print(f"  Sampled {len(frames)} frame(s) at {sample_interval}s interval",
                  flush=True)
        if len(frames) < 2:
            # Too short for meaningful classification — just copy through.
            _ffmpeg_copy_through(video_path, output_path)
            trimmed_duration = _probe_duration(output_path)
            return TrimResult(
                original_path=str(video_path),
                trimmed_path=str(output_path),
                original_duration=duration,
                trimmed_duration=trimmed_duration,
                model_used=model_name,
                trimmed_ratio=1.0,
            )

        # 2. Classify all frames in one Gemini call
        client = genai.Client(api_key=api_key)
        labels_by_idx, in_tok, out_tok = _classify_frames(
            client, frames, content_type, model_name,
            verbose=verbose, thinking_budget=thinking_budget,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # Align labels with frames; any missing → default scene (bias to keep)
    raw_labels = [
        labels_by_idx.get(i, SCENE_LABEL) for i in range(len(frames))
    ]

    # 3. Smooth single-frame noise
    smoothed = _smooth_frame_labels(raw_labels, min_nonscene_run)

    # 4. Build remove segments from smoothed labels
    removes = _frame_labels_to_removes(frames, smoothed, duration)
    merged = _merge_overlapping(removes)
    keeps = _compute_keep_ranges(duration, merged)

    if not keeps:
        raise RuntimeError("All frames classified as non-scene — refusing to write empty output")

    # 5. Execute ffmpeg
    if not merged:
        _ffmpeg_copy_through(video_path, output_path)
        if verbose:
            print(f"  All frames scene — copy-through", flush=True)
    elif len(keeps) == 1:
        s, e = keeps[0]
        _ffmpeg_single_range(video_path, output_path, s, e)
        if verbose:
            print(f"  Removed {len(merged)} region(s); kept {s:.1f}-{e:.1f}s",
                  flush=True)
    else:
        _ffmpeg_multi_range_concat(video_path, output_path, keeps)
        if verbose:
            total_kept = sum(e - s for s, e in keeps)
            print(f"  Removed {len(merged)} region(s); stitched "
                  f"{len(keeps)} keep-ranges = {total_kept:.1f}s", flush=True)

    trimmed_duration = _probe_duration(output_path)

    if verbose:
        from collections import Counter
        breakdown = Counter(smoothed)
        parts = [f"{k}:{v}" for k, v in breakdown.most_common()]
        print(f"  Frame labels: {', '.join(parts)}", flush=True)

    # Gemini pricing (verified 2026-04-22 from ai.google.dev/gemini-api/docs/pricing).
    # Trim pipeline uses only text+image inputs (no audio), so text rate applies.
    rates = {
        "gemini-2.5-flash":        (0.30, 2.50),
        "gemini-3-flash-preview":  (0.50, 3.00),
        "gemini-2.5-pro":          (1.25, 10.00),
    }
    rate_in, rate_out = rates.get(model_name, (0.30, 2.50))
    cost = (in_tok * rate_in + out_tok * rate_out) / 1_000_000

    return TrimResult(
        original_path=str(video_path),
        trimmed_path=str(output_path),
        original_duration=duration,
        trimmed_duration=trimmed_duration,
        removed_segments=[asdict(s) for s in merged],
        kept_ranges=[list(k) for k in keeps],
        model_used=model_name,
        token_usage_in=in_tok,
        token_usage_out=out_tok,
        cost_usd=round(cost, 4),
        trimmed_ratio=round(trimmed_duration / duration, 3) if duration > 0 else 0.0,
    )
