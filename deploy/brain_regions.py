"""Brain region mapping for emotional processing analysis.

Maps HCP Multi-Modal Parcellation (MMP) regions to functional categories
relevant to emotional processing. Used to interpret TRIBE v2 predictions.

The 20,484 vertices from TRIBE predictions are mapped to 181 HCP regions.
This module identifies which regions are involved in emotional processing
and provides tools to extract emotional activation profiles from predictions.

References:
- Glasser et al. (2016) "A multi-modal parcellation of human cerebral cortex"
- Lindquist et al. (2012) "The brain basis of emotion: A meta-analytic review"
- Barrett & Satpute (2013) "Large-scale brain networks in affective and social neuroscience"
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np

# ─── Emotional Region Mapping ─────────────────────────────────
# HCP region names → functional role in emotional processing
#
# Categories:
#   core_affect    — primary emotional experience and salience
#   regulation     — emotional regulation and control
#   social         — social emotions, empathy, theory of mind
#   reward         — reward, pleasure, motivational valence
#   interoception  — body state awareness, gut feelings
#   memory         — emotional memory, contextual associations
#   language       — emotional language processing, semantic emotion

EMOTIONAL_REGIONS = {
    # ─── Core Affect ──────────────────────────────────────────
    # Regions involved in basic emotional experience and salience detection
    "25": {"category": "core_affect", "name": "Area 25 (subgenual)", "role": "sadness, negative mood, autonomic emotional responses"},
    "33pr": {"category": "core_affect", "name": "Area 33 prime", "role": "emotional salience, visceral responses"},
    "a24": {"category": "core_affect", "name": "Anterior area 24", "role": "emotional processing, pain affect"},
    "a24pr": {"category": "core_affect", "name": "Anterior area 24 prime", "role": "emotional awareness, affective evaluation"},
    "p24": {"category": "core_affect", "name": "Posterior area 24", "role": "emotional monitoring"},
    "p24pr": {"category": "core_affect", "name": "Posterior area 24 prime", "role": "autonomic emotional regulation"},
    "p32": {"category": "core_affect", "name": "Posterior area 32", "role": "emotional conflict monitoring"},
    "s32": {"category": "core_affect", "name": "Area s32", "role": "emotional evaluation, self-referential emotion"},
    "AAIC": {"category": "core_affect", "name": "Anterior agranular insula", "role": "emotional awareness, interoceptive prediction"},

    # ─── Interoception ────────────────────────────────────────
    # Body state awareness — the physical feeling of emotion
    "Ig": {"category": "interoception", "name": "Insular granular", "role": "interoceptive awareness, body state monitoring"},
    "PoI1": {"category": "interoception", "name": "Posterior insular area 1", "role": "pain processing, visceral sensation"},
    "PoI2": {"category": "interoception", "name": "Posterior insular area 2", "role": "interoceptive integration"},
    "PI": {"category": "interoception", "name": "Para-insular", "role": "auditory-emotional integration"},
    "AVI": {"category": "interoception", "name": "Anterior ventral insula", "role": "emotional salience, empathy"},
    "MI": {"category": "interoception", "name": "Middle insula", "role": "emotional body state integration"},
    "FOP1": {"category": "interoception", "name": "Frontal opercular 1", "role": "taste, disgust, visceral awareness"},
    "FOP2": {"category": "interoception", "name": "Frontal opercular 2", "role": "emotional vocalization processing"},
    "FOP3": {"category": "interoception", "name": "Frontal opercular 3", "role": "emotional speech perception"},
    "FOP4": {"category": "interoception", "name": "Frontal opercular 4", "role": "sensorimotor emotion integration"},
    "FOP5": {"category": "interoception", "name": "Frontal opercular 5", "role": "emotional awareness"},

    # ─── Regulation ───────────────────────────────────────────
    # Prefrontal regions that regulate and modulate emotional responses
    "a32pr": {"category": "regulation", "name": "Anterior area 32 prime", "role": "emotional regulation, cognitive reappraisal"},
    "d32": {"category": "regulation", "name": "Dorsal area 32", "role": "conflict monitoring, emotional control"},
    "p32pr": {"category": "regulation", "name": "Posterior area 32 prime", "role": "emotional regulation"},
    "9m": {"category": "regulation", "name": "Area 9 medial", "role": "self-referential processing, emotional regulation"},
    "9a": {"category": "regulation", "name": "Area 9 anterior", "role": "emotional working memory"},
    "9p": {"category": "regulation", "name": "Area 9 posterior", "role": "metacognition, emotional awareness"},
    "10r": {"category": "regulation", "name": "Area 10 rostral", "role": "emotional decision-making, prospection"},
    "10d": {"category": "regulation", "name": "Area 10 dorsal", "role": "complex emotional evaluation"},
    "24dd": {"category": "regulation", "name": "Dorsal area 24d", "role": "emotional motor preparation"},
    "24dv": {"category": "regulation", "name": "Ventral area 24d", "role": "emotional response selection"},
    "SCEF": {"category": "regulation", "name": "Supplementary and cingulate eye field", "role": "emotional attention orientation"},
    "SFL": {"category": "regulation", "name": "Superior frontal language", "role": "emotional language regulation"},

    # ─── Social Emotion ───────────────────────────────────────
    # Theory of mind, empathy, social evaluation
    "STSda": {"category": "social", "name": "Superior temporal sulcus dorsal anterior", "role": "social perception, voice identity"},
    "STSdp": {"category": "social", "name": "Superior temporal sulcus dorsal posterior", "role": "social cognition, biological motion"},
    "STSva": {"category": "social", "name": "Superior temporal sulcus ventral anterior", "role": "social evaluation, emotional voice"},
    "STSvp": {"category": "social", "name": "Superior temporal sulcus ventral posterior", "role": "social meaning, narrative comprehension"},
    "STV": {"category": "social", "name": "Superior temporal visual", "role": "audiovisual emotional integration"},
    "TPOJ1": {"category": "social", "name": "Temporo-parieto-occipital junction 1", "role": "perspective-taking, empathy"},
    "TPOJ2": {"category": "social", "name": "Temporo-parieto-occipital junction 2", "role": "social attention"},
    "TGd": {"category": "social", "name": "Temporal pole dorsal", "role": "social concepts, emotional semantics"},
    "TGv": {"category": "social", "name": "Temporal pole ventral", "role": "social-emotional knowledge, person identity"},

    # ─── Reward / Valence ─────────────────────────────────────
    # Pleasure, motivation, positive/negative valence encoding
    "OFC": {"category": "reward", "name": "Orbitofrontal cortex", "role": "reward valuation, emotional outcome prediction"},
    "pOFC": {"category": "reward", "name": "Posterior orbitofrontal", "role": "emotional value computation"},
    "11l": {"category": "reward", "name": "Area 11 lateral", "role": "reward processing, hedonic evaluation"},
    "13l": {"category": "reward", "name": "Area 13 lateral", "role": "emotional valence encoding"},
    "a10p": {"category": "reward", "name": "Anterior area 10p", "role": "prospective reward, anticipation"},
    "a47r": {"category": "reward", "name": "Area 47r anterior", "role": "emotional evaluation of stimuli"},
    "p47r": {"category": "reward", "name": "Area 47r posterior", "role": "emotional reappraisal"},
    "47s": {"category": "reward", "name": "Area 47s", "role": "emotional language, semantic retrieval"},
    "10v": {"category": "reward", "name": "Area 10 ventral", "role": "value-based emotional decisions"},
    "10pp": {"category": "reward", "name": "Area 10 polar posterior", "role": "emotional prospection"},

    # ─── Emotional Memory ─────────────────────────────────────
    # Hippocampal/parahippocampal regions for emotional memory
    "EC": {"category": "memory", "name": "Entorhinal cortex", "role": "emotional memory encoding"},
    "H": {"category": "memory", "name": "Hippocampus", "role": "emotional memory consolidation, contextual fear"},
    "PeEc": {"category": "memory", "name": "Perirhinal/ectorhinal", "role": "emotional object recognition"},
    "PHA1": {"category": "memory", "name": "Parahippocampal area 1", "role": "contextual emotional associations"},
    "PHA2": {"category": "memory", "name": "Parahippocampal area 2", "role": "spatial-emotional context"},
    "PHA3": {"category": "memory", "name": "Parahippocampal area 3", "role": "scene-emotion binding"},
    "PreS": {"category": "memory", "name": "Presubiculum", "role": "spatial-emotional navigation"},
    "RSC": {"category": "memory", "name": "Retrosplenial cortex", "role": "emotional memory retrieval, nostalgia, self-continuity"},
    "TF": {"category": "memory", "name": "Area TF (fusiform)", "role": "emotional face/object processing"},
    "23c": {"category": "memory", "name": "Area 23c", "role": "emotional autobiographical memory"},
    "31a": {"category": "memory", "name": "Area 31a", "role": "self-referential emotional memory"},
    "d23ab": {"category": "memory", "name": "Dorsal area 23", "role": "emotional context retrieval"},
    "v23ab": {"category": "memory", "name": "Ventral area 23", "role": "emotional scene processing"},

    # ─── Emotional Language ───────────────────────────────────
    # Processing emotional content in language
    "44": {"category": "language", "name": "Area 44 (Broca's)", "role": "emotional speech production"},
    "45": {"category": "language", "name": "Area 45 (Broca's)", "role": "emotional language comprehension, semantic retrieval"},
    "47l": {"category": "language", "name": "Area 47 lateral", "role": "emotional semantic processing"},
    "STGa": {"category": "language", "name": "Superior temporal gyrus anterior", "role": "emotional prosody, vocal emotion"},
    "TA2": {"category": "language", "name": "Auditory association area", "role": "emotional sound processing"},
    "TE1a": {"category": "language", "name": "Temporal area 1 anterior", "role": "emotional narrative comprehension"},
    "TE1m": {"category": "language", "name": "Temporal area 1 middle", "role": "emotional concept processing"},
    "TE2a": {"category": "language", "name": "Temporal area 2 anterior", "role": "emotional semantic integration"},
    "A1": {"category": "language", "name": "Primary auditory cortex", "role": "emotional sound detection"},
    "LBelt": {"category": "language", "name": "Lateral belt", "role": "complex emotional sound processing"},
    "MBelt": {"category": "language", "name": "Medial belt", "role": "emotional voice processing"},
    "PBelt": {"category": "language", "name": "Parabelt", "role": "higher-order emotional audio"},
}

# Emotional region names grouped by category
CATEGORIES = {}
for region, info in EMOTIONAL_REGIONS.items():
    cat = info["category"]
    if cat not in CATEGORIES:
        CATEGORIES[cat] = []
    CATEGORIES[cat].append(region)


def get_emotional_profile(prediction: np.ndarray, mesh: str = "fsaverage5") -> dict:
    """Extract emotional activation profile from a TRIBE prediction.

    Args:
        prediction: 1D array of brain vertex activations (20,484 for fsaverage5)
        mesh: fsaverage mesh resolution

    Returns:
        Dict with per-category mean activation and per-region details
    """
    from tribev2.utils import get_hcp_roi_indices

    profile = {}
    for category, regions in CATEGORIES.items():
        region_activations = {}
        for region in regions:
            try:
                indices = get_hcp_roi_indices(region, mesh=mesh)
                activation = float(prediction[indices].mean())
                region_activations[region] = {
                    "activation": activation,
                    "name": EMOTIONAL_REGIONS[region]["name"],
                    "role": EMOTIONAL_REGIONS[region]["role"],
                }
            except (ValueError, IndexError):
                continue

        if region_activations:
            mean_activation = np.mean([r["activation"] for r in region_activations.values()])
            profile[category] = {
                "mean_activation": float(mean_activation),
                "regions": region_activations,
            }

    return profile


def get_top_emotional_regions(prediction: np.ndarray, k: int = 10,
                               mesh: str = "fsaverage5") -> list[dict]:
    """Get the k most activated emotional regions from a prediction.

    Returns list of dicts sorted by activation (descending).
    """
    from tribev2.utils import get_hcp_roi_indices

    all_regions = []
    for region, info in EMOTIONAL_REGIONS.items():
        try:
            indices = get_hcp_roi_indices(region, mesh=mesh)
            activation = float(prediction[indices].mean())
            all_regions.append({
                "region": region,
                "name": info["name"],
                "category": info["category"],
                "role": info["role"],
                "activation": activation,
            })
        except (ValueError, IndexError):
            continue

    all_regions.sort(key=lambda r: abs(r["activation"]), reverse=True)
    return all_regions[:k]


def compare_predictions(pred_a: np.ndarray, pred_b: np.ndarray,
                        mesh: str = "fsaverage5") -> dict:
    """Compare two TRIBE predictions for emotional similarity.

    Args:
        pred_a: First prediction (e.g., original content)
        pred_b: Second prediction (e.g., Fire's matched content)

    Returns:
        Dict with overall similarity, per-category comparison, and top divergences
    """
    profile_a = get_emotional_profile(pred_a, mesh)
    profile_b = get_emotional_profile(pred_b, mesh)

    # Per-category comparison
    category_comparison = {}
    activations_a = []
    activations_b = []

    for category in CATEGORIES:
        if category in profile_a and category in profile_b:
            mean_a = profile_a[category]["mean_activation"]
            mean_b = profile_b[category]["mean_activation"]
            activations_a.append(mean_a)
            activations_b.append(mean_b)
            category_comparison[category] = {
                "activation_a": mean_a,
                "activation_b": mean_b,
                "delta": mean_b - mean_a,
                "match": abs(mean_b - mean_a) < 0.05,  # within 5% = match
            }

    # Overall similarity (correlation of emotional region activations)
    if activations_a and activations_b:
        correlation = float(np.corrcoef(activations_a, activations_b)[0, 1])
        cosine = float(
            np.dot(activations_a, activations_b)
            / (np.linalg.norm(activations_a) * np.linalg.norm(activations_b) + 1e-10)
        )
    else:
        correlation = 0.0
        cosine = 0.0

    # Top divergences (regions that differ most)
    top_a = get_top_emotional_regions(pred_a, k=20, mesh=mesh)
    top_b = get_top_emotional_regions(pred_b, k=20, mesh=mesh)

    region_map_a = {r["region"]: r["activation"] for r in top_a}
    region_map_b = {r["region"]: r["activation"] for r in top_b}

    all_regions = set(region_map_a.keys()) | set(region_map_b.keys())
    divergences = []
    for region in all_regions:
        act_a = region_map_a.get(region, 0)
        act_b = region_map_b.get(region, 0)
        if abs(act_b - act_a) > 0.02:
            divergences.append({
                "region": region,
                "name": EMOTIONAL_REGIONS[region]["name"],
                "category": EMOTIONAL_REGIONS[region]["category"],
                "activation_a": act_a,
                "activation_b": act_b,
                "delta": act_b - act_a,
            })
    divergences.sort(key=lambda d: abs(d["delta"]), reverse=True)

    return {
        "overall_correlation": correlation,
        "overall_cosine": cosine,
        "verdict": "similar" if correlation > 0.7 else "different" if correlation < 0.3 else "partial",
        "category_comparison": category_comparison,
        "top_divergences": divergences[:10],
        "matching_categories": [c for c, v in category_comparison.items() if v["match"]],
        "diverging_categories": [c for c, v in category_comparison.items() if not v["match"]],
    }


def format_emotional_profile(profile: dict) -> str:
    """Format an emotional profile for display."""
    lines = []
    lines.append("  EMOTIONAL BRAIN PROFILE")
    lines.append("  " + "=" * 50)

    sorted_cats = sorted(profile.items(), key=lambda x: abs(x[1]["mean_activation"]), reverse=True)
    for category, data in sorted_cats:
        bar_len = int(abs(data["mean_activation"]) * 200)
        bar = "█" * min(bar_len, 30)
        sign = "+" if data["mean_activation"] > 0 else "-"
        lines.append(f"  {category.upper():20s} {sign}{abs(data['mean_activation']):.4f} {bar}")

        # Top 3 regions in this category
        sorted_regions = sorted(
            data["regions"].items(),
            key=lambda x: abs(x[1]["activation"]),
            reverse=True,
        )[:3]
        for region, info in sorted_regions:
            lines.append(f"    {info['name']:40s} {info['activation']:+.4f}  ({info['role']})")

    lines.append("  " + "=" * 50)
    return "\n".join(lines)


def format_comparison(comparison: dict) -> str:
    """Format a comparison result for display."""
    lines = []
    lines.append("  BRAIN COMPARISON")
    lines.append("  " + "=" * 50)
    lines.append(f"  Correlation: {comparison['overall_correlation']:.3f}")
    lines.append(f"  Cosine sim:  {comparison['overall_cosine']:.3f}")
    lines.append(f"  Verdict:     {comparison['verdict'].upper()}")
    lines.append("")

    lines.append("  Category breakdown:")
    for cat, data in comparison["category_comparison"].items():
        icon = "✓" if data["match"] else "✗"
        lines.append(f"    {icon} {cat:20s}  A:{data['activation_a']:+.4f}  B:{data['activation_b']:+.4f}  Δ:{data['delta']:+.4f}")

    if comparison["top_divergences"]:
        lines.append("")
        lines.append("  Top divergences:")
        for d in comparison["top_divergences"][:5]:
            lines.append(f"    {d['name']:40s}  A:{d['activation_a']:+.4f}  B:{d['activation_b']:+.4f}  ({d['category']})")

    lines.append("  " + "=" * 50)
    return "\n".join(lines)
