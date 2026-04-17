"""Modal serverless GPU adapter for TRIBE v2 brain prediction.

Wraps the existing tribe_modal.py deployment on Modal to conform
to the ComputeAdapter interface. Sends audio to Modal's A100 GPU,
runs TRIBE v2 inference, returns brain predictions.

Requires:
    - Modal account with $30/mo free credits (Starter plan)
    - TRIBE deployed on Modal: `modal deploy tribe_modal.py`
    - HuggingFace token for TRIBE model weight downloads

Cost: ~$0.08 per 60-second clip (A100-40GB at $0.000583/sec, ~140s runtime).

Usage:
    adapter = ModalTRIBEAdapter()
    prediction = adapter.predict(audio_bytes, "speech.mp3")
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from feeling_engine.adapters.compute.base import BrainPrediction, ComputeAdapter


class ModalTRIBEAdapter(ComputeAdapter):
    """Runs TRIBE v2 on Modal's serverless GPU infrastructure.

    This adapter calls the `predict_brain` function deployed on Modal
    via tribe_modal.py. The function runs on an A100-40GB GPU and
    returns vertex-level predictions + emotional profiles.

    The Modal function must be deployed before use:
        cd <path-to-tribe-tools>
        modal deploy tribe_modal.py
    """

    def __init__(
        self,
        app_name: str = "tribe-v2",
        function_name: str = "predict_brain",
        tribe_tools_path: Optional[str] = None,
    ):
        """Initialize Modal adapter.

        Args:
            app_name: Modal app name (as defined in tribe_modal.py)
            function_name: Modal function name to call
            tribe_tools_path: path to the directory containing tribe_modal.py
                (needed to import the Modal function reference)
        """
        self._app_name = app_name
        self._function_name = function_name
        self._tribe_tools_path = tribe_tools_path

    def predict(self, audio_bytes: bytes, filename: str) -> BrainPrediction:
        """Run TRIBE brain prediction on audio content via Modal GPU.

        Args:
            audio_bytes: raw audio file bytes (MP3, WAV, etc.)
            filename: original filename for format detection

        Returns:
            BrainPrediction with vertex-level time series
        """
        try:
            import modal
        except ImportError:
            raise ImportError(
                "modal package required. Install: pip install modal\n"
                "Then authenticate: modal setup"
            )

        # Look up the deployed function
        predict_fn = modal.Function.from_name(
            self._app_name, self._function_name
        )

        # Call Modal (sends audio to cloud GPU, waits for result)
        start = time.time()
        result = predict_fn.remote(audio_bytes, filename)
        runtime = time.time() - start

        # Extract predictions from Modal's response format
        # tribe_modal.py returns a dict with prediction_bytes, profiles, etc.
        if "prediction_bytes" in result:
            predictions = np.frombuffer(
                result["prediction_bytes"],
                dtype=np.float32,
            ).reshape(result.get("shape", (-1,)))

            # Handle shape — tribe_modal.py stores as flattened
            n_timesteps = result.get("n_timesteps", predictions.shape[0])
            if predictions.ndim == 1 and "n_vertices" in result:
                n_vertices = result["n_vertices"]
                predictions = predictions.reshape(n_timesteps, n_vertices)
            elif predictions.ndim == 1:
                predictions = predictions.reshape(1, -1)
        else:
            # Fallback: try to reconstruct from profiles
            profiles = result.get("profiles", [])
            if profiles:
                # Build a simple array from profile categories
                categories = list(profiles[0].get("categories", {}).keys())
                predictions = np.array([
                    [p["categories"].get(c, 0.0) for c in categories]
                    for p in profiles
                ])
                n_timesteps = len(profiles)
            else:
                raise RuntimeError("Modal returned no prediction data")

        n_timesteps, n_vertices = predictions.shape

        return BrainPrediction(
            predictions=predictions,
            n_timesteps=n_timesteps,
            n_vertices=n_vertices,
            duration_seconds=result.get("duration_seconds", 0.0),
            runtime_seconds=runtime,
            provider="modal",
            model_name="TRIBE v2",
            metadata={
                "profiles": result.get("profiles", []),
                "peak_timestep": result.get("peak_timestep"),
                "modal_app": self._app_name,
            },
        )

    def predict_from_file(self, audio_path: str | Path) -> BrainPrediction:
        """Convenience: predict from a local audio file path."""
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        audio_bytes = path.read_bytes()
        return self.predict(audio_bytes, path.name)

    @property
    def provider_name(self) -> str:
        return "Modal (A100-40GB)"

    @staticmethod
    def is_deployed() -> bool:
        """Check if the TRIBE function is deployed on Modal."""
        try:
            import modal
            fn = modal.Function.from_name("tribe-v2", "predict_brain")
            return True
        except Exception:
            return False
