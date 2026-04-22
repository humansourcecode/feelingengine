"""Arc Mining — extract mechanism arcs from outlier content.

Pipeline:
    URL -> yt-dlp (audio + metadata + auto-subs)
        -> Modal TRIBE (research path, CC BY-NC)
        -> detect_mechanisms (vocabulary form, your IP)
        -> detect_sequences
        -> sqlite arc_library.db

Downstream consumption of the arc library is commercial-safe because
arcs are stored in vocabulary form. Raw TRIBE outputs are not persisted.

Usage:
    # One URL
    python -m feeling_engine.mining.arc_miner mine <URL> --db arc_library.db --niche history

    # Batch from a file (one URL per line, optional tab-separated niche)
    python -m feeling_engine.mining.arc_miner mine-batch seeds.tsv --db arc_library.db

    # Inspect what's in the library
    python -m feeling_engine.mining.arc_miner list --db arc_library.db

Architecture: mining uses TRIBE (research path, CC BY-NC 4.0) to produce
per-video mechanism arcs, then stores them in vocabulary form (your IP).
Downstream consumers work off the vocabulary arcs, keeping commercial
uses insulated from TRIBE's non-commercial license. Mining code is MIT
so any user can build their own arc library; the library itself remains
a private asset of whoever mined it.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ─── Schema ──────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS arcs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT NOT NULL UNIQUE,
    medium TEXT NOT NULL,
    modality TEXT,                   -- 'video' (trimodal), 'audio', 'text'
    channel TEXT,
    channel_handle TEXT,
    channel_subs INTEGER,
    title TEXT,
    niche TEXT,
    duration_sec REAL,
    view_count INTEGER,
    channel_median_views REAL,
    outlier_ratio REAL,
    pub_date TEXT,
    arc_json TEXT NOT NULL,           -- absolute-threshold applications (Jobs-calibrated)
    normalized_arc_json TEXT,         -- time-pct normalized (for compose)
    sigma_arc_json TEXT,              -- σ-threshold applications (per-video z-normalized)
    sigma_sequences_json TEXT,        -- sequence matches on σ-mode arc
    axis_stats_json TEXT,             -- per-axis mean+std (value + deriv) for this video
    sequences_json TEXT,
    n_labels INTEGER,
    n_sequences INTEGER,
    n_labels_sigma INTEGER,
    n_sequences_sigma INTEGER,
    tribe_grounded INTEGER DEFAULT 1,
    mined_at TEXT NOT NULL
);

-- Schema migration: add columns for σ-mode on existing databases
-- (SQLite will error-then-skip on duplicate; we swallow via try/except in init_db)

CREATE INDEX IF NOT EXISTS idx_arcs_channel ON arcs(channel_handle);
CREATE INDEX IF NOT EXISTS idx_arcs_niche ON arcs(niche);
CREATE INDEX IF NOT EXISTS idx_arcs_medium ON arcs(medium);
"""


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA_SQL)
    # Idempotent migration for pre-σ databases: add new columns if missing
    for ddl in (
        "ALTER TABLE arcs ADD COLUMN sigma_arc_json TEXT",
        "ALTER TABLE arcs ADD COLUMN sigma_sequences_json TEXT",
        "ALTER TABLE arcs ADD COLUMN axis_stats_json TEXT",
        "ALTER TABLE arcs ADD COLUMN n_labels_sigma INTEGER",
        "ALTER TABLE arcs ADD COLUMN n_sequences_sigma INTEGER",
    ):
        try:
            conn.execute(ddl)
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()


# ─── yt-dlp helpers ──────────────────────────────────────────────

def fetch_metadata(url: str) -> dict:
    print(f"  Fetching metadata...", end="", flush=True)
    result = subprocess.run(
        ["yt-dlp", "--dump-json", "--no-download", url],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp metadata failed: {result.stderr[:200]}")
    d = json.loads(result.stdout)
    handle = (d.get("uploader_id") or "").strip()
    if handle and not handle.startswith("@"):
        handle = f"@{handle}"
    meta = {
        "url": url,
        "video_id": d.get("id"),
        "title": d.get("title"),
        "channel": d.get("channel") or d.get("uploader"),
        "channel_handle": handle,
        "channel_url": d.get("channel_url") or d.get("uploader_url"),
        "channel_subs": int(d.get("channel_follower_count") or 0),
        "duration_sec": float(d.get("duration") or 0),
        "view_count": int(d.get("view_count") or 0),
        "pub_date": d.get("upload_date"),
    }
    print(f" {(meta['title'] or '')[:60]}", flush=True)
    return meta


def fetch_channel_median_views(
    channel_url: str,
    exclude_video_id: Optional[str] = None,
) -> Optional[dict]:
    """Compute a channel's median view count via yt-dlp --flat-playlist.

    Single Innertube API call; no per-video page fetches. Typically 5-15s.
    Returns {"median": int, "n_videos": int} or None on failure.
    Excludes the currently-mining video from the median if its ID is provided,
    so the outlier itself doesn't skew its own baseline.
    """
    import statistics
    try:
        result = subprocess.run(
            ["yt-dlp", "--flat-playlist",
             "--print", "%(id)s\t%(view_count)s",
             "--no-download", channel_url],
            capture_output=True, text=True, timeout=120,
        )
    except subprocess.TimeoutExpired:
        return None
    if result.returncode != 0:
        return None
    views = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("\t")
        if len(parts) != 2:
            continue
        vid, vc = parts
        if exclude_video_id and vid == exclude_video_id:
            continue
        try:
            views.append(int(vc))
        except ValueError:
            continue
    if not views:
        return None
    return {"median": int(statistics.median(views)), "n_videos": len(views)}


def download_video(url: str, out_dir: Path) -> Path:
    """Download video (mp4) via yt-dlp, capped at 720p to keep upload size reasonable.

    TRIBE v2 is trimodal — feeding mp4 (not audio-only) enables visual features
    (DINOv2 + V-JEPA2) alongside audio + text transcript. Audio-only inputs
    lose the visual signal entirely.
    """
    print(f"  Downloading video (720p mp4)...", end="", flush=True)
    template = str(out_dir / "%(id)s.%(ext)s")
    result = subprocess.run(
        [
            "yt-dlp",
            "-f", "best[height<=720][ext=mp4]/best[ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", template,
            url,
        ],
        capture_output=True, text=True, timeout=900,
    )
    if result.returncode != 0:
        raise RuntimeError(f"yt-dlp video download failed: {result.stderr[:200]}")
    mp4s = list(out_dir.glob("*.mp4"))
    if not mp4s:
        raise RuntimeError("No mp4 produced by yt-dlp")
    path = mp4s[0]
    size_mb = path.stat().st_size / 1024 / 1024
    print(f" {size_mb:.1f}MB", flush=True)
    return path


def fetch_transcript(url: str, out_dir: Path) -> Optional[dict]:
    """Return word-level transcript dict from YouTube auto-captions.

    Format: {"words": [{"word", "start", "end"}]}
    Returns None if captions unavailable.
    """
    print(f"  Fetching transcript...", end="", flush=True)
    template = str(out_dir / "%(id)s.%(ext)s")
    subprocess.run(
        [
            "yt-dlp", "--skip-download",
            "--write-auto-sub", "--sub-lang", "en",
            "--sub-format", "json3",
            "-o", template,
            url,
        ],
        capture_output=True, text=True, timeout=60,
    )
    sub_files = list(out_dir.glob("*.en.json3"))
    if not sub_files:
        print(" none found", flush=True)
        return None

    try:
        data = json.loads(sub_files[0].read_text())
    except Exception:
        print(" parse failed", flush=True)
        return None

    words = []
    for event in data.get("events", []):
        t_start = (event.get("tStartMs") or 0) / 1000.0
        for seg in event.get("segs", []):
            word = (seg.get("utf8") or "").strip()
            if not word or word in ("\n", " "):
                continue
            off = (seg.get("tOffsetMs") or 0) / 1000.0
            start = t_start + off
            words.append({"word": word, "start": start, "end": start + 0.3})

    # Fill end times using next word's start (better than flat 0.3)
    for i in range(len(words) - 1):
        nxt = words[i + 1]["start"]
        if nxt > words[i]["start"]:
            words[i]["end"] = nxt

    print(f" {len(words)} words", flush=True)
    return {"words": words} if words else None


# ─── TRIBE ───────────────────────────────────────────────────────

def run_tribe(content_path: Path) -> tuple[list, str]:
    """Call Modal TRIBE (trimodal), return (profiles, modality)."""
    print(f"  Running TRIBE trimodal on Modal...", end="", flush=True)
    from feeling_engine.adapters.compute.modal_tribe import ModalTRIBEAdapter
    adapter = ModalTRIBEAdapter()
    t0 = time.time()
    prediction = adapter.predict_from_file(content_path)
    runtime = time.time() - t0
    profiles = prediction.metadata.get("profiles", [])
    # modality is returned by predict_brain; adapter passes it through metadata
    # Fallback to extension inference if not present (older Modal deployment)
    modality = prediction.metadata.get("modality")
    if not modality:
        ext = content_path.suffix.lower()
        modality = "video" if ext in {".mp4", ".mov", ".webm", ".mkv"} else (
            "audio" if ext in {".mp3", ".wav", ".m4a", ".flac"} else "text"
        )
    print(f" {len(profiles)} timesteps ({modality}) in {runtime:.1f}s", flush=True)
    return profiles, modality


# ─── Normalization ───────────────────────────────────────────────

def normalize_arc(applications, duration_sec: float) -> list[dict]:
    """Convert each LabelApplication's absolute seconds to percent of duration."""
    result = []
    for app in applications:
        d = asdict(app)
        if duration_sec > 0:
            d["pct_start"] = round(app.start_sec / duration_sec, 4)
            d["pct_end"] = round(app.end_sec / duration_sec, 4)
        result.append(d)
    return result


# ─── DB write ────────────────────────────────────────────────────

def write_entry(db_path: str, row: dict) -> int:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO arcs
          (url, medium, modality, channel, channel_handle, channel_subs,
           title, niche, duration_sec, view_count,
           channel_median_views, outlier_ratio, pub_date,
           arc_json, normalized_arc_json, sequences_json,
           sigma_arc_json, sigma_sequences_json, axis_stats_json,
           n_labels, n_sequences, n_labels_sigma, n_sequences_sigma,
           tribe_grounded, mined_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        row["url"], row["medium"], row.get("modality"),
        row.get("channel"), row.get("channel_handle"),
        row.get("channel_subs"), row.get("title"), row.get("niche"),
        row.get("duration_sec"), row.get("view_count"),
        row.get("channel_median_views"), row.get("outlier_ratio"),
        row.get("pub_date"),
        row["arc_json"], row.get("normalized_arc_json"), row.get("sequences_json"),
        row.get("sigma_arc_json"), row.get("sigma_sequences_json"),
        row.get("axis_stats_json"),
        row.get("n_labels"), row.get("n_sequences"),
        row.get("n_labels_sigma"), row.get("n_sequences_sigma"),
        int(row.get("tribe_grounded", 1)),
        row["mined_at"],
    ))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


# ─── Pipeline ────────────────────────────────────────────────────

def _cache_path(db_path: str, video_id: str) -> Path:
    return Path(db_path).parent / "arc_cache" / f"{video_id}.json"


def _save_cache(db_path: str, video_id: str, payload: dict) -> None:
    p = _cache_path(db_path, video_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload))


def _load_cache(db_path: str, video_id: str) -> Optional[dict]:
    p = _cache_path(db_path, video_id)
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _mechanisms_and_write(
    profiles: list,
    transcript: Optional[dict],
    modality: str,
    db_path: str,
    url: str,
    medium: str,
    meta: dict,
    niche: Optional[str],
    channel_median: Optional[float],
) -> dict:
    """Run mechanism + sequence detection on TRIBE profiles and write a row.

    Shared core called by mine_url, mine_audio, mine_text. Contains the
    vocabulary-only steps (Tier 1/2/3 detection + sequence detection +
    normalization) + row construction + DB write. Source-specific ingestion
    (yt-dlp / local-audio / text-synthesis) is the caller's responsibility.
    """
    from feeling_engine.mechanisms.api import detect_mechanisms, detect_sequences
    from feeling_engine.mechanisms.tier1_detectors import compute_axis_stats

    print(f"  Detecting mechanisms (absolute)...", end="", flush=True)
    arc_abs = detect_mechanisms(tribe_categories=profiles, transcript=transcript)
    print(f" {len(arc_abs)} applications", flush=True)

    print(f"  Detecting mechanisms (σ-normalized)...", end="", flush=True)
    axis_stats = compute_axis_stats(profiles)
    arc_sigma = detect_mechanisms(
        tribe_categories=profiles, transcript=transcript, axis_stats=axis_stats,
    )
    print(f" {len(arc_sigma)} applications", flush=True)

    print(f"  Detecting sequences (both modes)...", end="", flush=True)
    sequences_abs = detect_sequences(arc_abs)
    sequences_sigma = detect_sequences(arc_sigma)
    print(f" abs={len(sequences_abs)} σ={len(sequences_sigma)}", flush=True)

    duration_sec = float(meta.get("duration_sec") or 0.0)
    normalized = normalize_arc(arc_abs, duration_sec)

    view_count = meta.get("view_count")
    outlier_ratio = None
    if channel_median and channel_median > 0 and view_count:
        outlier_ratio = round(view_count / channel_median, 2)

    # Sanity check: a real outlier should beat its channel median.
    # A ratio < 1.5 suggests either the content isn't actually an outlier
    # or the stored channel_median is wrong (the failure mode Task #84 caught).
    if outlier_ratio is not None and outlier_ratio < 1.5:
        print(f"  ⚠  WARNING: outlier_ratio={outlier_ratio} is < 1.5 — "
              f"this content appears to have UNDERPERFORMED its channel median "
              f"({int(channel_median):,}). Either it's not an outlier, or "
              f"channel_median_views is incorrect. Verify before trusting.",
              flush=True)

    row = {
        "url": url,
        "medium": medium,
        "modality": modality,
        "channel": meta.get("channel"),
        "channel_handle": meta.get("channel_handle"),
        "channel_subs": meta.get("channel_subs"),
        "title": meta.get("title"),
        "niche": niche,
        "duration_sec": duration_sec,
        "view_count": view_count,
        "channel_median_views": channel_median,
        "outlier_ratio": outlier_ratio,
        "pub_date": meta.get("pub_date"),
        "arc_json": json.dumps([asdict(a) for a in arc_abs]),
        "normalized_arc_json": json.dumps(normalized),
        "sequences_json": json.dumps([asdict(s) for s in sequences_abs]),
        "sigma_arc_json": json.dumps([asdict(a) for a in arc_sigma]),
        "sigma_sequences_json": json.dumps([asdict(s) for s in sequences_sigma]),
        "axis_stats_json": json.dumps(axis_stats),
        "n_labels": len(arc_abs),
        "n_sequences": len(sequences_abs),
        "n_labels_sigma": len(arc_sigma),
        "n_sequences_sigma": len(sequences_sigma),
        "tribe_grounded": 1,
        "mined_at": datetime.now(timezone.utc).isoformat(),
    }

    rid = write_entry(db_path, row)
    print(f"  ✓ Saved to {db_path} as id={rid} "
          f"(abs: {len(arc_abs)} labels / {len(sequences_abs)} seq · "
          f"σ: {len(arc_sigma)} labels / {len(sequences_sigma)} seq)")
    return {
        "id": rid, "url": url,
        "n_labels": len(arc_abs), "n_sequences": len(sequences_abs),
        "n_labels_sigma": len(arc_sigma), "n_sequences_sigma": len(sequences_sigma),
        "duration_sec": duration_sec, "view_count": view_count,
    }


def mine_url(
    url: str,
    db_path: str,
    niche: Optional[str] = None,
    channel_median: Optional[float] = None,
) -> dict:
    """Mine a YouTube URL end-to-end: yt-dlp → Modal TRIBE → detect → write."""
    print(f"\n▶ Mining: {url}")
    init_db(db_path)

    # Fetch metadata first so we have video_id for cache lookup
    meta = fetch_metadata(url)
    video_id = meta.get("video_id") or url.rsplit("=", 1)[-1]

    # Live-compute channel median by default. Manual override via
    # channel_median arg stays available for tests / deterministic replay.
    if channel_median is None and meta.get("channel_url"):
        print(f"  Computing channel median views (live)...", end="", flush=True)
        result = fetch_channel_median_views(meta["channel_url"],
                                            exclude_video_id=video_id)
        if result:
            channel_median = float(result["median"])
            print(f" {int(channel_median):,} from {result['n_videos']} videos",
                  flush=True)
            if result["n_videos"] < 10:
                print(f"  ⚠  Small catalog ({result['n_videos']} videos) — "
                      f"median may be statistically fragile",
                      flush=True)
        else:
            print(f" FAILED — no channel_median_views will be stored",
                  flush=True)

    cached = _load_cache(db_path, video_id)
    if cached:
        print(f"  Cache hit for {video_id} — skipping yt-dlp + Modal")
        profiles = cached["profiles"]
        modality = cached.get("modality", "video")
        transcript = cached.get("transcript")
    else:
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            video = download_video(url, tmp)
            transcript = fetch_transcript(url, tmp)
            profiles, modality = run_tribe(video)
        # Persist so a downstream failure doesn't cost another Modal call
        _save_cache(db_path, video_id, {
            "url": url, "video_id": video_id,
            "profiles": profiles, "modality": modality,
            "transcript": transcript, "meta": meta,
        })
        print(f"  Cached Modal result → arc_cache/{video_id}.json")

    return _mechanisms_and_write(
        profiles=profiles, transcript=transcript, modality=modality,
        db_path=db_path, url=url, medium="youtube_video",
        meta=meta, niche=niche, channel_median=channel_median,
    )


def mine_audio(
    audio_path,
    db_path: str,
    url: str,
    meta: dict,
    niche: Optional[str] = None,
    channel_median: Optional[float] = None,
    medium: str = "audio_upload",
    transcript: Optional[dict] = None,
) -> dict:
    """Mine a local audio file: Modal TRIBE → detect → write.

    Caller must supply `meta` (title, channel, duration_sec, etc.) — there's no
    yt-dlp to fetch it. If duration_sec is missing, it's probed from the audio.
    """
    from pathlib import Path as _Path
    audio_path = _Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    print(f"\n▶ Mining audio: {audio_path.name}")
    init_db(db_path)

    meta = dict(meta) if meta else {}
    if not meta.get("duration_sec"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(audio_path)],
                capture_output=True, text=True, timeout=30, check=True,
            )
            meta["duration_sec"] = float(out.stdout.strip())
        except Exception:
            meta["duration_sec"] = 0.0

    profiles, modality = run_tribe(audio_path)
    return _mechanisms_and_write(
        profiles=profiles, transcript=transcript, modality=modality,
        db_path=db_path, url=url, medium=medium,
        meta=meta, niche=niche, channel_median=channel_median,
    )


ELEVENLABS_MAX_CHARS = 10000
TTS_CHUNK_TARGET = 8000  # headroom under the ElevenLabs limit


def _chunk_text_for_tts(text: str, target: int = TTS_CHUNK_TARGET) -> list:
    """Split text into chunks ≤ target chars, respecting paragraph + sentence
    boundaries. Never splits mid-sentence."""
    import re as _re
    paragraphs = text.split("\n\n")
    chunks = []
    buf = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        if len(p) > target:
            if buf:
                chunks.append(buf.strip())
                buf = ""
            sentences = _re.split(r'(?<=[.!?])\s+', p)
            sub = ""
            for s in sentences:
                if len(sub) + len(s) + 1 <= target:
                    sub = (sub + " " + s).strip()
                else:
                    if sub:
                        chunks.append(sub)
                    sub = s
            if sub:
                chunks.append(sub)
            continue
        if len(buf) + len(p) + 2 <= target:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            chunks.append(buf.strip())
            buf = p
    if buf:
        chunks.append(buf.strip())
    return chunks


def text_to_audio(
    text: str,
    out_path,
    voice_id: Optional[str] = None,
    use_voice_picker: bool = True,
) -> dict:
    """Synthesize text → MP3 via ElevenLabs. Handles chunking + ffmpeg concat.

    Returns {'audio_path': Path, 'voice_id': str, 'voice_name': str,
    'rationale': str or None, 'duration_sec': float, 'n_chunks': int}.
    """
    from pathlib import Path as _Path
    from feeling_engine.adapters.tts.elevenlabs import ElevenLabsAdapter

    out_path = _Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tts = ElevenLabsAdapter()

    picker_info = {"name": None, "rationale": None, "method": "manual"}
    if voice_id is None and use_voice_picker:
        from feeling_engine.voice_picker import pick_voice
        print(f"  Picking voice...", flush=True)
        choice = pick_voice(text=text, tts_adapter=tts)
        voice_id = choice.voice_id
        picker_info = {"name": choice.name, "rationale": choice.rationale,
                       "method": choice.method}
        print(f"  Voice: {choice.name} ({choice.voice_id}) via {choice.method}",
              flush=True)

    chunks = _chunk_text_for_tts(text)
    print(f"  Text: {len(text)} chars → {len(chunks)} chunk(s) "
          f"(sizes: {[len(c) for c in chunks]})", flush=True)

    if len(chunks) == 1:
        tts.synthesize(text=chunks[0], output_path=str(out_path), voice_id=voice_id)
    else:
        chunks_dir = out_path.parent / (out_path.stem + "_chunks")
        chunks_dir.mkdir(exist_ok=True)
        chunk_paths = []
        for i, ch in enumerate(chunks, 1):
            p = chunks_dir / f"{out_path.stem}_chunk_{i:02d}.mp3"
            print(f"    [{i}/{len(chunks)}] synthesizing {len(ch)} chars → {p.name}",
                  flush=True)
            tts.synthesize(text=ch, output_path=str(p), voice_id=voice_id)
            chunk_paths.append(p)
        concat_list = chunks_dir / "concat.txt"
        concat_list.write_text("\n".join(f"file '{p}'" for p in chunk_paths))
        print(f"  Concatenating {len(chunk_paths)} chunks via ffmpeg...", flush=True)
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", str(concat_list), "-c", "copy", str(out_path)],
            check=True, capture_output=True,
        )

    # Probe duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(out_path)],
        capture_output=True, text=True, timeout=30,
    )
    duration_sec = float(probe.stdout.strip()) if probe.stdout.strip() else 0.0
    size_mb = out_path.stat().st_size / 1e6
    print(f"  Wrote {out_path.name} ({size_mb:.1f} MB, {duration_sec:.1f}s)",
          flush=True)
    return {
        "audio_path": out_path,
        "voice_id": voice_id,
        "voice_name": picker_info["name"],
        "rationale": picker_info["rationale"],
        "picker_method": picker_info["method"],
        "duration_sec": duration_sec,
        "n_chunks": len(chunks),
    }


def mine_text(
    text: str,
    db_path: str,
    url: str,
    meta: dict,
    niche: Optional[str] = None,
    channel_median: Optional[float] = None,
    voice_id: Optional[str] = None,
    medium: str = "text_synthesized",
    out_dir=None,
) -> dict:
    """Synthesize text → audio via ElevenLabs, then mine via Modal TRIBE.

    If out_dir is provided, audio + chunks are persisted there; otherwise a
    temp dir is used and cleaned up on exit. Adds `voice_id` / `voice_name`
    to meta for later audit.
    """
    from pathlib import Path as _Path
    print(f"\n▶ Mining text: {url}")
    init_db(db_path)

    if out_dir is not None:
        out_dir = _Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        audio_path = out_dir / "synth.mp3"
        synth = text_to_audio(text, audio_path, voice_id=voice_id)
        # Fall through — audio persists on disk
        meta_out = dict(meta) if meta else {}
        meta_out["duration_sec"] = synth["duration_sec"]
        meta_out["voice_id"] = synth["voice_id"]
        meta_out["voice_name"] = synth["voice_name"]
        return mine_audio(
            audio_path=synth["audio_path"], db_path=db_path, url=url,
            meta=meta_out, niche=niche, channel_median=channel_median,
            medium=medium,
        )

    with tempfile.TemporaryDirectory() as tmp:
        audio_path = _Path(tmp) / "synth.mp3"
        synth = text_to_audio(text, audio_path, voice_id=voice_id)
        meta_out = dict(meta) if meta else {}
        meta_out["duration_sec"] = synth["duration_sec"]
        meta_out["voice_id"] = synth["voice_id"]
        meta_out["voice_name"] = synth["voice_name"]
        return mine_audio(
            audio_path=synth["audio_path"], db_path=db_path, url=url,
            meta=meta_out, niche=niche, channel_median=channel_median,
            medium=medium,
        )


# ─── CLI ─────────────────────────────────────────────────────────

def _cmd_list(db_path: str) -> None:
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = list(conn.execute("""
        SELECT id, channel_handle, title, duration_sec, view_count,
               outlier_ratio, niche, n_labels, n_sequences,
               n_labels_sigma, n_sequences_sigma
        FROM arcs ORDER BY id
    """))
    if not rows:
        print(f"(empty — {db_path})")
        return
    print(f"{'id':>3} {'channel':<25} {'dur':>5} {'views':>10} {'ratio':>6} "
          f"{'niche':<18} {'abs':>4}/{'σ':<4} {'seq':>3}/{'σ':<3}  title")
    for r in rows:
        dur = f"{int(r['duration_sec'] or 0) // 60}m{int(r['duration_sec'] or 0) % 60:02d}"
        ratio = f"{r['outlier_ratio']:.1f}x" if r['outlier_ratio'] else "—"
        print(f"{r['id']:>3} {(r['channel_handle'] or '—')[:25]:<25} {dur:>5} "
              f"{r['view_count'] or 0:>10,} {ratio:>6} {(r['niche'] or '—')[:18]:<18} "
              f"{r['n_labels'] or 0:>4}/{r['n_labels_sigma'] or 0:<4} "
              f"{r['n_sequences'] or 0:>3}/{r['n_sequences_sigma'] or 0:<3}  "
              f"{(r['title'] or '')[:55]}")


def main():
    parser = argparse.ArgumentParser(
        description="Arc miner — extract mechanism arcs from outlier content.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("mine", help="Mine a single URL")
    m.add_argument("url")
    m.add_argument("--db", default="arc_library.db")
    m.add_argument("--niche")
    m.add_argument("--channel-median", type=float,
                   help="Channel median views (for outlier ratio)")

    b = sub.add_parser("mine-batch",
                       help="Mine multiple URLs from a TSV file "
                            "(url[\\tniche[\\tchannel_median]] per line)")
    b.add_argument("urls_file")
    b.add_argument("--db", default="arc_library.db")

    ma = sub.add_parser(
        "mine-audio",
        help="Mine a local audio file (skips yt-dlp; caller supplies meta)",
    )
    ma.add_argument("audio_path")
    ma.add_argument("--url", required=True,
                    help="Source URL for the content (stored in row)")
    ma.add_argument("--meta-json", required=True,
                    help="JSON string or path to JSON with title/channel/"
                         "channel_handle/pub_date/view_count")
    ma.add_argument("--db", default="arc_library.db")
    ma.add_argument("--niche")
    ma.add_argument("--channel-median", type=float,
                    help="Publication/channel median (views, restacks, etc.)")
    ma.add_argument("--medium", default="audio_upload",
                    help="Medium label (default: audio_upload)")

    mt = sub.add_parser(
        "mine-text",
        help="Mine from text: ElevenLabs synthesize → Modal TRIBE → detect → write",
    )
    mt.add_argument("text_path")
    mt.add_argument("--url", required=True,
                    help="Source URL for the content")
    mt.add_argument("--meta-json", required=True,
                    help="JSON string or path to JSON with metadata")
    mt.add_argument("--db", default="arc_library.db")
    mt.add_argument("--niche")
    mt.add_argument("--channel-median", type=float)
    mt.add_argument("--voice-id",
                    help="ElevenLabs voice ID (omit to use voice picker)")
    mt.add_argument("--medium", default="text_synthesized")
    mt.add_argument("--out-dir",
                    help="Persist synthesized audio + chunks here "
                         "(default: temp dir, cleaned up)")

    ls = sub.add_parser("list", help="List arcs in library")
    ls.add_argument("--db", default="arc_library.db")

    args = parser.parse_args()

    def _parse_meta(s):
        p = Path(s)
        if p.exists():
            return json.loads(p.read_text())
        return json.loads(s)

    if args.cmd == "mine":
        mine_url(args.url, args.db,
                 niche=args.niche, channel_median=args.channel_median)
    elif args.cmd == "mine-batch":
        lines = Path(args.urls_file).read_text().strip().splitlines()
        for i, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            url = parts[0]
            niche = parts[1] if len(parts) > 1 and parts[1] else None
            median = float(parts[2]) if len(parts) > 2 and parts[2] else None
            print(f"\n[{i}/{len(lines)}]")
            try:
                mine_url(url, args.db, niche=niche, channel_median=median)
            except Exception as e:
                print(f"  ✗ Failed {url}: {e}", file=sys.stderr)
    elif args.cmd == "mine-audio":
        meta = _parse_meta(args.meta_json)
        mine_audio(
            audio_path=args.audio_path, db_path=args.db, url=args.url,
            meta=meta, niche=args.niche,
            channel_median=args.channel_median, medium=args.medium,
        )
    elif args.cmd == "mine-text":
        meta = _parse_meta(args.meta_json)
        text = Path(args.text_path).read_text()
        mine_text(
            text=text, db_path=args.db, url=args.url, meta=meta,
            niche=args.niche, channel_median=args.channel_median,
            voice_id=args.voice_id, medium=args.medium,
            out_dir=args.out_dir,
        )
    elif args.cmd == "list":
        _cmd_list(args.db)


if __name__ == "__main__":
    main()
