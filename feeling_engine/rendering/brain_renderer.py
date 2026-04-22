"""Canonical brain renderer — mechanism labels → brain images (no TRIBE).

Takes a mechanism label, looks up its canonical 7-category activation signature,
spreads each category's value across the HCP-MMP regions that category owns,
and renders a colored brain surface to PNG.

Public infrastructure: fsaverage mesh (nilearn), HCP-MMP atlas annotation files
(bundled in deploy/hcp_mmp_data/). Our IP: the per-label signature table.

No TRIBE at runtime. Fully commercial-safe.

Usage:
    from feeling_engine.rendering.brain_renderer import render_mechanism_brain

    render_mechanism_brain(
        label="body-turn",
        output_path="body_turn.png",
        view="lateral_both",
    )
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

from feeling_engine.rendering.signatures import (
    CATEGORIES, get_signature, signature_as_vector,
)


# Path to bundled HCP-MMP annotation files. They live next to tribe_modal.py
# because TRIBE uses them too, but they're public research assets — we can use
# them without the TRIBE license constraint.
_DEPLOY_HCP = Path(__file__).resolve().parents[2] / "deploy" / "hcp_mmp_data"
ANNOT_LH = _DEPLOY_HCP / "lh.HCPMMP1.annot"
ANNOT_RH = _DEPLOY_HCP / "rh.HCPMMP1.annot"


# ─── HCP region → emotional category mapping ─────────────────────────
# Lifted from deploy/brain_regions.py's EMOTIONAL_REGIONS dict (condensed).
# Each of the 7 categories maps to HCP region abbreviations. Names here match
# the HCP-MMP convention used in the annotation file: L_<name>_ROI / R_<name>_ROI.

CATEGORY_REGIONS: dict = {
    "interoception": [
        "Ig", "PoI1", "PoI2", "PI", "AVI", "MI",
        "FOP1", "FOP2", "FOP3", "FOP4", "FOP5",
    ],
    "regulation": [
        "a32pr", "d32", "p32pr", "9m", "9a", "9p",
        "10r", "10d", "24dd", "24dv", "SCEF", "SFL",
        "p10p", "a10p", "10pp",
    ],
    "core_affect": [
        "25", "33pr", "a24", "a24pr", "p24", "p24pr",
        "p32", "s32", "AAIC",
    ],
    "social": [
        "STSda", "STSdp", "STSva", "STSvp", "TPOJ1", "TPOJ2", "TPOJ3",
        "PGi", "PGs", "PGp", "TE1a", "TE1m", "TE1p", "TE2a", "TE2p",
        "PFm", "PFt", "PSL",
    ],
    "reward": [
        "OFC", "pOFC", "11l", "13l", "47l", "47m", "47s",
        "a47r", "p47r", "a10p", "6ma",  # 6ma borderline motivational
    ],
    "memory": [
        "EC", "PreS", "PeEc", "H", "ProS", "POS1", "POS2",
        "RSC", "v23ab", "d23ab", "31pv", "31pd", "31a",
        "7m", "7Pm",
    ],
    "language": [
        "44", "45", "IFSa", "IFSp", "IFJa", "IFJp",
        "A1", "LBelt", "MBelt", "PBelt", "A4", "A5",
        "STGa", "STV", "SFL", "PSL", "PHT",
    ],
}


# ─── Atlas loader ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_parcellation() -> dict:
    """Load HCP-MMP annotation for both hemispheres.

    Returns a dict with:
        "lh_labels": per-vertex label index (left hemi)
        "rh_labels": per-vertex label index (right hemi)
        "lh_names":  list of region names (index into annotation) e.g. 'L_V1_ROI'
        "rh_names":  same for right
    Cached to avoid re-reading the annotation files.
    """
    import nibabel as nib

    if not ANNOT_LH.exists() or not ANNOT_RH.exists():
        raise FileNotFoundError(
            f"HCP-MMP annotation files not found: {ANNOT_LH}, {ANNOT_RH}"
        )

    lh_labels, _, lh_names = nib.freesurfer.read_annot(str(ANNOT_LH))
    rh_labels, _, rh_names = nib.freesurfer.read_annot(str(ANNOT_RH))

    # Region names are bytes in the annot format; decode for comparison.
    lh_names = [n.decode() if isinstance(n, bytes) else n for n in lh_names]
    rh_names = [n.decode() if isinstance(n, bytes) else n for n in rh_names]
    return {
        "lh_labels": lh_labels,
        "rh_labels": rh_labels,
        "lh_names": lh_names,
        "rh_names": rh_names,
    }


def _hcp_name(region: str, hemi: str) -> str:
    """Normalize one of our region abbreviations to the HCP annotation convention.

    Region abbreviations come from CATEGORY_REGIONS. HCP annotation uses
    'L_<name>_ROI' (and 'R_...') or similar. Special cases: numeric regions
    (e.g. '25') are prefixed as 'L_25_ROI'.
    """
    prefix = "L" if hemi == "lh" else "R"
    return f"{prefix}_{region}_ROI"


def _vertex_activation_maps(label: str, intensity: float = 1.0) -> tuple:
    """Build a per-vertex activation vector for each hemisphere.

    Returns (lh_activations, rh_activations) — numpy arrays sized to the
    hemisphere vertex count, where each vertex gets the value of whichever
    category its region belongs to (scaled by the label's signature +
    user-provided intensity). Vertices not in any known emotional region
    get 0.0.
    """
    parc = _load_parcellation()
    sig = signature_as_vector(label)

    # Pre-compute region_name → category lookup for fast vertex fill
    region_to_value: dict = {}
    for category, regions in CATEGORY_REGIONS.items():
        value = sig.get(category, 0.0) * intensity
        for r in regions:
            region_to_value[r] = value

    def build_hemi(labels, names, hemi: str):
        activations = np.zeros(len(labels), dtype=np.float32)
        # For each annotation region index, check if that region maps to
        # a category in our signature.
        for idx, region_name in enumerate(names):
            # HCP names look like 'L_V1_ROI'; strip prefix + suffix for lookup.
            if region_name.startswith(("L_", "R_")) and region_name.endswith("_ROI"):
                short = region_name[2:-4]
            else:
                short = region_name
            value = region_to_value.get(short, 0.0)
            if value != 0.0:
                activations[labels == idx] = value
        return activations

    lh_act = build_hemi(parc["lh_labels"], parc["lh_names"], "lh")
    rh_act = build_hemi(parc["rh_labels"], parc["rh_names"], "rh")
    return lh_act, rh_act


# ─── Public rendering API ────────────────────────────────────────────

_VIEW_SPECS = {
    "lateral_left":  [("lh", "lateral")],
    "lateral_right": [("rh", "lateral")],
    "lateral_both":  [("lh", "lateral"), ("rh", "lateral")],
    "medial_both":   [("lh", "medial"), ("rh", "medial")],
    "all":           [("lh", "lateral"), ("rh", "lateral"),
                      ("lh", "medial"),  ("rh", "medial")],
}


def render_mechanism_brain(
    label: str,
    output_path,
    view: str = "lateral_both",
    intensity: float = 1.0,
    title: Optional[str] = None,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = +1.0,
    dpi: int = 150,
) -> Path:
    """Render the canonical brain pattern for one mechanism label.

    Args:
        label: mechanism label (e.g., "body-turn"). Must be in MECHANISM_SIGNATURE.
        output_path: where to write the PNG.
        view: one of 'lateral_left', 'lateral_right', 'lateral_both',
            'medial_both', 'all'. Default 'lateral_both'.
        intensity: scalar in [0, 1] scaling the signature magnitudes. Useful for
            tying the rendering to per-moment intensity. Default 1.0.
        title: optional title text above the figure.
        cmap: matplotlib colormap. Default 'RdBu_r' (diverging, red=positive).
        vmin, vmax: color range. Default -1.0 to +1.0 matches signature range.
        dpi: output DPI. Default 150.

    Returns:
        Path to the written PNG.
    """
    if view not in _VIEW_SPECS:
        raise ValueError(f"view must be one of {list(_VIEW_SPECS)}, got {view!r}")

    # Validate label up front — KeyError would otherwise surface mid-render
    get_signature(label)

    from nilearn import plotting, datasets
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lh_act, rh_act = _vertex_activation_maps(label, intensity=intensity)

    # Fetch fsaverage (standard) surface — first call downloads ~5MB, cached after
    fsaverage = datasets.fetch_surf_fsaverage("fsaverage")

    panels = _VIEW_SPECS[view]
    n = len(panels)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(
        rows, cols,
        figsize=(cols * 5, rows * 4),
        subplot_kw={"projection": "3d"},
    )
    if n == 1:
        axes = [axes]
    else:
        axes = np.array(axes).reshape(-1)

    for ax, (hemi, view_name) in zip(axes, panels):
        # Pial surface shows the actual anatomical brain shape (folds, sulci);
        # bg_on_data=True overlays the sulc shading onto colored regions for depth.
        side = "left" if hemi == "lh" else "right"
        mesh = fsaverage[f"pial_{side}"]
        bg_map = fsaverage[f"sulc_{side}"]
        activations = lh_act if hemi == "lh" else rh_act
        plotting.plot_surf(
            surf_mesh=mesh,
            surf_map=activations,
            bg_map=bg_map,
            bg_on_data=True,
            hemi=side,
            view=view_name,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            axes=ax,
            figure=fig,
        )
        ax.set_title(f"{hemi.upper()} {view_name}", fontsize=10)

    # Hide any unused axes
    for extra in axes[len(panels):]:
        extra.set_visible(False)

    if title:
        fig.suptitle(title, fontsize=14, y=0.98)
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    return output_path
