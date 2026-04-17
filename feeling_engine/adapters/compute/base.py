"""Base interface for GPU compute adapters.

Wraps whatever service runs TRIBE v2 (or future brain models) on GPU.
The adapter's job is: audio bytes in, brain prediction array out.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BrainPrediction:
    """Raw brain prediction output from a brain model.

    Attributes:
        predictions: array of shape (n_timesteps, n_vertices)
            Each row is one timestep's brain activation across all vertices.
        n_timesteps: number of time windows in the prediction
        n_vertices: number of brain surface vertices (e.g., ~20k for fsaverage5)
        duration_seconds: duration of the input content
        runtime_seconds: how long the prediction took
        provider: which compute service was used
        model_name: which brain model was used
        metadata: any additional info from the provider
    """
    predictions: np.ndarray
    n_timesteps: int
    n_vertices: int
    duration_seconds: float
    runtime_seconds: float
    provider: str = "unknown"
    model_name: str = "unknown"
    metadata: dict = field(default_factory=dict)


class ComputeAdapter(ABC):
    """Interface for GPU compute providers running brain models."""

    @abstractmethod
    def predict(self, audio_bytes: bytes, filename: str) -> BrainPrediction:
        """Run brain prediction on audio content.

        Args:
            audio_bytes: raw audio file bytes
            filename: original filename (for format detection)

        Returns:
            BrainPrediction with full vertex-level time series
        """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable provider name (e.g., 'Modal A100')."""
