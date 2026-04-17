"""Base interface for brain model adapters.

Wraps the mapping from raw vertex-level predictions to interpretable
brain region categories. Currently supports TRIBE v2's fsaverage5 mesh
mapped to 7 emotional categories via HCP atlas.

Future models may use different parcellations or output formats. This
interface abstracts that.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BrainRegionProfile:
    """Brain activation aggregated by functional region.

    Each category represents a group of brain regions with a known
    functional role in emotional processing.
    """
    categories: dict[str, float]  # category_name → mean activation
    timestep: int
    dominant_category: str | None = None
    raw_regions: dict[str, dict] | None = None  # detailed per-region data


@dataclass
class BrainTimeSeries:
    """Full time series of brain region profiles.

    This is the standard input to the Translator — a sequence of
    per-timestep brain region activations.
    """
    profiles: list[BrainRegionProfile]
    n_timesteps: int
    category_names: list[str]
    metadata: dict = field(default_factory=dict)


class BrainModelAdapter(ABC):
    """Interface for brain model region mapping."""

    @abstractmethod
    def map_to_regions(
        self, predictions: np.ndarray
    ) -> BrainTimeSeries:
        """Map raw vertex predictions to brain region profiles.

        Args:
            predictions: array of shape (n_timesteps, n_vertices)

        Returns:
            BrainTimeSeries with per-timestep category activations
        """

    @property
    @abstractmethod
    def category_names(self) -> list[str]:
        """List of brain region category names this adapter produces."""
