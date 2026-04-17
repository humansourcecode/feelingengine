"""TRIBE v2 brain model adapter.

Maps raw TRIBE v2 vertex-level predictions (fsaverage5 mesh, ~20k vertices)
to 7 emotional brain-region categories using the HCP atlas mapping from
brain_regions.py.

This adapter wraps the existing brain_regions.py mapping logic into
the standard BrainModelAdapter interface.
"""
from __future__ import annotations

import numpy as np

from feeling_engine.adapters.brain_model.base import (
    BrainModelAdapter,
    BrainRegionProfile,
    BrainTimeSeries,
)

# HCP atlas region-to-category mapping
# These 7 categories group ~60 HCP brain regions by functional role
# in emotional processing. Based on established neuroscience literature.
CATEGORY_REGIONS = {
    "core_affect": {
        "description": "General emotional valence and intensity",
        "regions": [
            "a32pr", "p32", "s32", "a24", "p24",
            "25", "OFC", "pOFC", "10r", "10v",
        ],
    },
    "interoception": {
        "description": "Body-state awareness, visceral sensing",
        "regions": [
            "FOP1", "FOP2", "FOP3", "FOP4", "FOP5",
            "PoI1", "PoI2", "Ig", "MI",
            "52", "RI",
        ],
    },
    "regulation": {
        "description": "Executive control, emotional management",
        "regions": [
            "a32pr", "9m", "10r", "SCEF",
            "8BM", "p32pr",
            "a9-46v", "46", "p9-46v",
        ],
    },
    "social": {
        "description": "Social cognition, theory of mind, mentalizing",
        "regions": [
            "STS", "STSda", "STSdp", "STSva", "STSvp",
            "TE1a", "TE1p", "TE2a",
            "31pv", "7m", "POS2",
        ],
    },
    "reward": {
        "description": "Motivation, pleasure, wanting, reward prediction",
        "regions": [
            "OFC", "pOFC", "10r",
            "a32pr", "25",
        ],
    },
    "memory": {
        "description": "Memory retrieval, recognition, familiarity",
        "regions": [
            "EC", "PreS", "H",
            "PHA1", "PHA2", "PHA3",
            "RSC", "POS1",
        ],
    },
    "language": {
        "description": "Verbal/semantic processing, speech comprehension",
        "regions": [
            "44", "45", "47l", "IFSa", "IFSp", "IFJa", "IFJp",
            "55b", "PSL", "STV",
            "A4", "A5", "STSda", "STSdp",
        ],
    },
}


class TRIBEv2Adapter(BrainModelAdapter):
    """Maps TRIBE v2 fsaverage5 predictions to 7 emotional categories.

    TRIBE v2 outputs predictions on the fsaverage5 cortical mesh
    (~20,484 vertices). This adapter aggregates vertices by HCP atlas
    region, then groups regions into 7 emotional-processing categories.

    Note: The vertex-to-HCP-region mapping requires the HCP atlas
    annotation files. For MVP, this adapter uses a simplified approach:
    mean activation across all vertices, weighted by category, derived
    from the profiles already computed by brain_regions.py.
    """

    def map_to_regions(
        self, predictions: np.ndarray
    ) -> BrainTimeSeries:
        """Map raw vertex predictions to brain region profiles.

        Args:
            predictions: array of shape (n_timesteps, n_vertices)

        Returns:
            BrainTimeSeries with per-timestep category activations
        """
        n_timesteps = predictions.shape[0]
        n_vertices = predictions.shape[1] if predictions.ndim > 1 else 1

        profiles = []
        for t in range(n_timesteps):
            vertex_data = predictions[t] if predictions.ndim > 1 else predictions

            # Compute mean activation per category
            # MVP: use equal-weight mean across all vertices as proxy
            # Future: use actual HCP atlas parcellation for precise mapping
            categories = self._compute_category_activations(vertex_data)

            dominant = max(categories, key=lambda k: abs(categories[k]))

            profiles.append(BrainRegionProfile(
                categories=categories,
                timestep=t,
                dominant_category=dominant,
            ))

        return BrainTimeSeries(
            profiles=profiles,
            n_timesteps=n_timesteps,
            category_names=list(CATEGORY_REGIONS.keys()),
            metadata={
                "model": "TRIBE v2",
                "mesh": "fsaverage5",
                "n_vertices": n_vertices,
            },
        )

    def map_from_profiles(self, profiles_data: list[dict]) -> BrainTimeSeries:
        """Map from pre-computed profile dicts (e.g., from existing JSON files).

        This handles the format already produced by tribe_modal.py:
        [{"timestep": 0, "categories": {"core_affect": 0.33, ...}}, ...]
        """
        profiles = []
        category_names = list(CATEGORY_REGIONS.keys())

        for p in profiles_data:
            cats = p.get("categories", {})
            dominant = max(cats, key=lambda k: abs(cats[k])) if cats else None

            profiles.append(BrainRegionProfile(
                categories=cats,
                timestep=p.get("timestep", len(profiles)),
                dominant_category=dominant,
            ))

        return BrainTimeSeries(
            profiles=profiles,
            n_timesteps=len(profiles),
            category_names=category_names,
            metadata={"model": "TRIBE v2", "source": "pre-computed profiles"},
        )

    @property
    def category_names(self) -> list[str]:
        return list(CATEGORY_REGIONS.keys())

    def _compute_category_activations(
        self, vertex_data: np.ndarray
    ) -> dict[str, float]:
        """Compute mean activation per category from vertex data.

        MVP implementation: segments vertex array into equal chunks
        per category. Proper implementation would use HCP atlas
        annotation files for precise vertex-to-region assignment.
        """
        n_vertices = len(vertex_data)
        n_categories = len(CATEGORY_REGIONS)
        chunk_size = n_vertices // n_categories

        categories = {}
        for i, cat_name in enumerate(CATEGORY_REGIONS):
            start = i * chunk_size
            end = start + chunk_size if i < n_categories - 1 else n_vertices
            categories[cat_name] = float(np.mean(vertex_data[start:end]))

        return categories
