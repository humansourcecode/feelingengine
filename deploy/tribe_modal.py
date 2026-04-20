"""TRIBE v2 — Modal Serverless GPU Runner

One-command brain prediction from any conversation.
No pods to manage, no SSH, no setup. Modal handles everything.

Setup (one time):
    pip install modal
    modal setup          # authenticate
    modal deploy tribe_modal.py   # deploy the function

Usage (every time):
    # Predict brain response from audio
    python tribe_modal.py predict audio.mp3

    # Compare two audio files
    python tribe_modal.py compare audio_a.mp3 audio_b.mp3

    # Results saved locally: prediction.npy, profiles.json, brain.png
"""

import json
import os
import sys
from pathlib import Path

import modal

# ─── Modal Image Definition ──────────────────────────────────
# This defines the container that runs on Modal's GPU.
# Built once, cached, reused on every call.

tribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git")
    .pip_install(
        "torch==2.4.0",
        "torchaudio==2.4.0",
        "torchvision==0.19.0",
        gpu="T4",
    )
    .pip_install(
        "tribev2 @ git+https://github.com/facebookresearch/tribev2.git",
        "whisperx",
        "pyannote.audio",
        "nltk",
        "matplotlib",
        "nilearn",
        "seaborn",
        "pyvista",
        "colorcet",
        "scikit-image",
    )
    .run_commands(
        # Download NLTK data
        "python3 -c \"import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)\"",
        # Patch TRIBE to use whisperx directly (not uvx) and force float32
        # compute type — subprocess GPU visibility for whisperx can flake on
        # Modal and the ctranslate2 fp16 check rejects, killing the whole run.
        "TRIBE_PATH=$(python3 -c 'import tribev2, os; print(os.path.dirname(tribev2.__file__))') && "
        "sed -i 's/\"uvx\",//' ${TRIBE_PATH}/eventstransforms.py && "
        "sed -i 's/compute_type = \"float16\"/compute_type = \"float32\"/' ${TRIBE_PATH}/eventstransforms.py",
    )
    .add_local_file(
        str(Path(__file__).parent / "brain_regions.py"),
        "/root/brain_regions.py",
        copy=True,
    )
    # HCP-MMP parcellation files — figshare blocks Modal's egress (HTTP 403),
    # so we bundle them in the image. MNE's fetch_hcp_mmp_parcellation skips
    # download if the files already exist at the expected SUBJECTS_DIR path.
    .add_local_file(
        str(Path(__file__).parent / "hcp_mmp_data" / "lh.HCPMMP1.annot"),
        "/root/hcp_mmp_data/lh.HCPMMP1.annot",
        copy=True,
    )
    .add_local_file(
        str(Path(__file__).parent / "hcp_mmp_data" / "rh.HCPMMP1.annot"),
        "/root/hcp_mmp_data/rh.HCPMMP1.annot",
        copy=True,
    )
)

app = modal.App("tribe-v2", image=tribe_image)

# ─── Modal Volume for caching model weights ──────────────────
# Llama weights are ~15GB. Cache them so they don't re-download.
model_cache = modal.Volume.from_name("tribe-model-cache", create_if_missing=True)


# ─── GPU Function: Predict ────────────────────────────────────

@app.function(
    gpu="A100-80GB",
    timeout=7200,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def predict_brain(content_bytes: bytes, filename: str) -> dict:
    """Run TRIBE prediction on content (video/audio/text). Returns prediction + profiles.

    Modality is routed by file extension:
      .mp4/.mov/.webm/.mkv → video_path (full trimodal: vision + audio + text)
      .mp3/.wav/.m4a/.flac/.ogg → audio_path (audio + text transcript)
      other → text_path (converted to speech internally)

    A100-80GB selected: trimodal needs ≥40GB VRAM per TRIBE docs;
    80GB gives headroom. Empirically A100 is faster than H100 for
    this workload (see KB Ref #25).
    """
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.WARNING)

    import time
    from pathlib import Path
    import numpy as np
    import torch

    # Set cache dir and auth HuggingFace
    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TORCH_HOME"] = "/cache/torch"
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

    # Stage bundled HCP-MMP parcellation files to MNE's expected location.
    # Avoids the figshare-403 network issue on Modal egress.
    import shutil
    import mne
    mne_subjects_dir = (
        mne.get_config("SUBJECTS_DIR")
        or os.environ.get("SUBJECTS_DIR")
        or os.path.expanduser("~/mne_data/MNE-fsaverage-data")
    )
    os.environ["SUBJECTS_DIR"] = mne_subjects_dir
    hcp_dest = os.path.join(mne_subjects_dir, "fsaverage", "label")
    os.makedirs(hcp_dest, exist_ok=True)
    for hemi in ("lh", "rh"):
        src = f"/root/hcp_mmp_data/{hemi}.HCPMMP1.annot"
        dst = os.path.join(hcp_dest, f"{hemi}.HCPMMP1.annot")
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copy(src, dst)

    # Save content to temp file
    content_path = f"/tmp/{filename}"
    with open(content_path, "wb") as f:
        f.write(content_bytes)

    ext = Path(filename).suffix.lower()
    VIDEO_EXTS = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
    AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}

    # Load model
    from tribev2 import TribeModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TribeModel.from_pretrained("facebook/tribev2", device=device)

    # Route by modality
    start = time.time()
    if ext in VIDEO_EXTS:
        modality = "video"
        events = model.get_events_dataframe(video_path=content_path)
    elif ext in AUDIO_EXTS:
        modality = "audio"
        events = model.get_events_dataframe(audio_path=content_path)
    else:
        modality = "text"
        events = model.get_events_dataframe(text_path=content_path)
    event_time = time.time() - start

    pred_start = time.time()
    predictions, info = model.predict(events)
    pred_time = time.time() - pred_start
    total = time.time() - start

    # Emotional profiling
    sys.path.insert(0, "/root")
    from brain_regions import (
        get_emotional_profile,
        get_top_emotional_regions,
        format_emotional_profile,
    )

    # Profile each timestep
    profiles = []
    for t in range(predictions.shape[0]):
        p = get_emotional_profile(predictions[t])
        profiles.append({
            "timestep": t,
            "categories": {
                cat: round(data["mean_activation"], 4)
                for cat, data in p.items()
            },
        })

    # Peak timestep analysis
    mean_acts = [abs(predictions[t].mean()) for t in range(predictions.shape[0])]
    peak_t = int(np.argmax(mean_acts))
    peak_profile = get_emotional_profile(predictions[peak_t])
    top_regions = get_top_emotional_regions(predictions[peak_t], k=10)

    # Render brain image
    brain_png = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from tribev2.plotting.cortical import PlotBrainNilearn
        import io

        plotter = PlotBrainNilearn(mesh="fsaverage5")
        fig, axes = plotter.get_fig_axes(views=["left", "right"])
        plotter.plot_surf(
            predictions[peak_t], axes=axes, views=["left", "right"],
            cmap="hot", norm_percentile=95,
        )
        plt.suptitle(f"Brain Activation — t={peak_t}s", fontsize=12)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        brain_png = buf.getvalue()
    except Exception as e:
        print(f"Render failed: {e}")

    # Commit cached model weights
    model_cache.commit()

    return {
        "prediction": predictions.tobytes(),
        "prediction_shape": list(predictions.shape),
        "prediction_dtype": str(predictions.dtype),
        "profiles": profiles,
        "modality": modality,
        "peak_timestep": peak_t,
        "peak_profile": {
            cat: round(data["mean_activation"], 4)
            for cat, data in peak_profile.items()
        },
        "top_regions": [
            {"name": r["name"], "category": r["category"],
             "activation": round(r["activation"], 4), "role": r["role"]}
            for r in top_regions
        ],
        "timing": {
            "event_extraction": round(event_time, 1),
            "brain_prediction": round(pred_time, 1),
            "total": round(total, 1),
        },
        "brain_png": brain_png,
        "device": f"cuda ({torch.cuda.get_device_name(0)})" if device == "cuda" else "cpu",
    }


@app.function(
    gpu="A100-40GB",
    timeout=900,
    volumes={"/cache": model_cache},
    secrets=[modal.Secret.from_name("huggingface-token")],
)
def compare_brains(audio_a_bytes: bytes, filename_a: str,
                   audio_b_bytes: bytes, filename_b: str) -> dict:
    """Run TRIBE on two audio files and compare brain activation."""
    import warnings
    warnings.filterwarnings("ignore")
    import logging
    logging.disable(logging.WARNING)

    import time
    import numpy as np
    import torch

    os.environ["HF_HOME"] = "/cache/huggingface"
    os.environ["TORCH_HOME"] = "/cache/torch"
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)

    # Save audio files
    path_a = f"/tmp/{filename_a}"
    path_b = f"/tmp/{filename_b}"
    with open(path_a, "wb") as f:
        f.write(audio_a_bytes)
    with open(path_b, "wb") as f:
        f.write(audio_b_bytes)

    # Load model once
    from tribev2 import TribeModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TribeModel.from_pretrained("facebook/tribev2", device=device)

    # Predict both
    start = time.time()
    events_a = model.get_events_dataframe(audio_path=path_a)
    pred_a, _ = model.predict(events_a)

    events_b = model.get_events_dataframe(audio_path=path_b)
    pred_b, _ = model.predict(events_b)
    total = time.time() - start

    # Compare
    sys.path.insert(0, "/root")
    from brain_regions import compare_predictions, get_emotional_profile

    mean_a = pred_a.mean(axis=0)
    mean_b = pred_b.mean(axis=0)
    comparison = compare_predictions(mean_a, mean_b)

    # Serialize comparison (convert numpy types)
    comparison_clean = json.loads(json.dumps(comparison, default=lambda x: float(x) if hasattr(x, 'item') else str(x)))

    model_cache.commit()

    return {
        "prediction_a_shape": list(pred_a.shape),
        "prediction_b_shape": list(pred_b.shape),
        "comparison": comparison_clean,
        "profile_a": {
            cat: round(data["mean_activation"], 4)
            for cat, data in get_emotional_profile(mean_a).items()
        },
        "profile_b": {
            cat: round(data["mean_activation"], 4)
            for cat, data in get_emotional_profile(mean_b).items()
        },
        "timing_total": round(total, 1),
    }


# ─── Local CLI ────────────────────────────────────────────────
# This runs on YOUR machine, sends audio to Modal, gets results back.

def _save_results(result: dict, output_dir: str):
    """Save prediction results locally."""
    os.makedirs(output_dir, exist_ok=True)

    # Reconstruct numpy array
    import numpy as np
    pred = np.frombuffer(result["prediction"], dtype=result["prediction_dtype"])
    pred = pred.reshape(result["prediction_shape"])
    np.save(os.path.join(output_dir, "prediction.npy"), pred)

    # Save profiles
    with open(os.path.join(output_dir, "profiles.json"), "w") as f:
        json.dump(result["profiles"], f, indent=2)

    # Save brain image
    if result.get("brain_png"):
        with open(os.path.join(output_dir, "brain.png"), "wb") as f:
            f.write(result["brain_png"])

    print(f"  Saved to {output_dir}/")


def cli_predict(audio_path: str, output_dir: str = None):
    """Run TRIBE prediction via Modal."""
    if output_dir is None:
        output_dir = Path(audio_path).stem + "_tribe"

    print(f"  Uploading {audio_path} to Modal...")
    audio_bytes = Path(audio_path).read_bytes()
    filename = Path(audio_path).name

    print(f"  Running TRIBE on GPU (serverless)...")
    fn = modal.Function.from_name("tribe-v2", "predict_brain")
    result = fn.remote(audio_bytes, filename)

    print(f"  Device: {result['device']}")
    print(f"  Timing: {result['timing']['total']}s total")
    print(f"  Result: {result['prediction_shape'][0]}t x {result['prediction_shape'][1]}v")
    print(f"  Peak: t={result['peak_timestep']}")
    print()

    # Show profile
    print("  Emotional Profile (peak):")
    for cat, val in sorted(result["peak_profile"].items(), key=lambda x: -abs(x[1])):
        bar = "█" * int(abs(val) * 200)
        sign = "+" if val > 0 else "-"
        print(f"    {cat:20s} {sign}{abs(val):.4f} {bar}")
    print()

    print("  Top Regions:")
    for r in result["top_regions"]:
        print(f"    {r['name']:40s} {r['activation']:+.4f}  [{r['category']}]")

    _save_results(result, output_dir)


def cli_compare(audio_a: str, audio_b: str):
    """Compare two audio files via Modal."""
    print(f"  Uploading {audio_a} and {audio_b}...")
    bytes_a = Path(audio_a).read_bytes()
    bytes_b = Path(audio_b).read_bytes()

    print(f"  Running TRIBE comparison on GPU...")
    fn = modal.Function.from_name("tribe-v2", "compare_brains")
    result = fn.remote(
        bytes_a, Path(audio_a).name,
        bytes_b, Path(audio_b).name,
    )

    print(f"  Total time: {result['timing_total']}s")
    print()

    comp = result["comparison"]
    print(f"  Correlation: {comp['overall_correlation']:.3f}")
    print(f"  Cosine:      {comp['overall_cosine']:.3f}")
    print(f"  Verdict:     {comp['verdict'].upper()}")
    print()

    print("  Category comparison:")
    for cat, data in comp.get("category_comparison", {}).items():
        icon = "✓" if data["match"] else "✗"
        print(f"    {icon} {cat:20s}  A:{data['activation_a']:+.4f}  B:{data['activation_b']:+.4f}")

    # Save comparison
    with open("comparison.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved comparison.json")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TRIBE v2 — Modal Serverless GPU")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("predict", help="Predict brain response from audio")
    p.add_argument("audio", help="Path to audio file")
    p.add_argument("--output", help="Output directory")

    c = sub.add_parser("compare", help="Compare two audio files")
    c.add_argument("audio_a", help="First audio file")
    c.add_argument("audio_b", help="Second audio file")

    args = parser.parse_args()

    if args.command == "predict":
        cli_predict(args.audio, args.output)
    elif args.command == "compare":
        cli_compare(args.audio_a, args.audio_b)
    else:
        parser.print_help()
