"""Tier 1 detectors — TRIBE-only signal processing.

Each detector takes a timeline of TRIBE category activations and returns
a list of LabelApplication objects for the label it detects.

Two threshold modes:

- ABSOLUTE (default): thresholds are raw magnitudes on TRIBE activations.
  Tuned against a speech-heavy baseline (the 60-second Steve Jobs
  Stanford death-pivot clip, which also ships in examples/arcs/). Biases
  toward speech-heavy content: quieter content types (music/visual) have
  narrower activation ranges and can under-fire.

- SIGMA: thresholds are σ-multipliers of per-video axis std-dev. Pass
  `axis_stats` (from `compute_axis_stats`) to enable. Makes detectors
  content-agnostic by construction — "relatively intense for this piece"
  rather than "above absolute X". `DEFAULT_SIGMA_THRESHOLDS` is a faithful
  port of `DEFAULT_THRESHOLDS` calibrated against the Jobs baseline, so
  detection on that clip is preserved within ~5%.

See docs/detector_validation.md for the absolute-threshold calibration
protocol.
"""
from __future__ import annotations

from typing import List, Optional
import statistics
from feeling_engine.mechanisms.vocabulary import LabelApplication

# ─── Default thresholds — ABSOLUTE mode (empirically tuned on Jobs) ──
DEFAULT_THRESHOLDS = {
    "body_turn_intero_deriv": 0.03,
    "body_turn_lang_deriv": -0.02,
    "body_surge_deriv": 0.10,
    "body_anchor_intero": 0.08,
    "body_anchor_stability": 0.03,
    "body_anchor_affect_max": 0.05,
    "body_anchor_lang_max": 0.10,
    "affect_rise_deriv": 0.05,
    "affect_fade_prev_peak": 0.15,
    "affect_fade_curr_max": 0.08,
    "restraint_regulation": 0.20,
    "restraint_stability": 0.04,
    "restraint_affect_floor": 0.05,
    "release_deriv": -0.05,
    "anticipation_reward_deriv": 0.03,
    "anticipation_reward_floor": 0.03,
    "anticipation_regulation_stability": 0.03,
    "satisfaction_peak_min_value": 0.05,
    "satisfaction_peak_prominence": 0.03,
    "recognition_memory_deriv": 0.03,
    "evocation_memory_deriv": 0.02,
    "evocation_affect_deriv": 0.02,
    "word_focus_language": 0.20,
    "word_focus_intero_max": 0.08,
    "word_recede_lang_deriv": -0.05,
    "word_recede_other_deriv": 0.02,
    "inward_pivot_lang_deriv": -0.02,
    "inward_pivot_intero_deriv": 0.05,
    "pattern_break_total_change": 0.30,
    "pattern_break_min_big_axes": 2,
    "pattern_break_axis_threshold": 0.07,
    "sensation_flood_intero": 0.15,
    "sensation_flood_affect": 0.10,
    "drift_max_value": 0.05,
    "drift_min_duration": 3,
    "vulnerability_social_deriv": 0.03,
    "vulnerability_intero_deriv": 0.03,
}

# ─── Default thresholds — SIGMA mode ─────────────────────────────────
# Faithful port of DEFAULT_THRESHOLDS: each σ value = absolute ÷ Jobs axis std.
# Computed against the Steve Jobs Stanford death-pivot clip (bundled at
# examples/arcs/steve_jobs_death_pivot.json) so detection on that clip
# reproduces under σ-mode within ~5%.
#
# Axis-specific thresholds use that axis's std on Jobs. Cross-axis thresholds
# (pattern_break_total_change, pattern_break_axis_threshold, drift_max_value)
# use the mean std across all 7 axes.
DEFAULT_SIGMA_THRESHOLDS = {
    "body_turn_intero_deriv": 0.76,       # 0.03 / 0.0395
    "body_turn_lang_deriv": -0.61,        # -0.02 / 0.0326
    "body_surge_deriv": 2.53,             # 0.10 / 0.0395
    "body_anchor_intero": 1.10,           # 0.08 / 0.0725
    "body_anchor_stability": 0.76,        # 0.03 / 0.0395
    "body_anchor_affect_max": 0.87,       # 0.05 / 0.0575
    "body_anchor_lang_max": 0.93,         # 0.10 / 0.1073
    "affect_rise_deriv": 1.45,            # 0.05 / 0.0345
    "affect_fade_prev_peak": 2.61,        # 0.15 / 0.0575
    "affect_fade_curr_max": 1.39,         # 0.08 / 0.0575
    "restraint_regulation": 3.88,         # 0.20 / 0.0516
    "restraint_stability": 1.30,          # 0.04 / 0.0308
    "restraint_affect_floor": 0.87,       # 0.05 / 0.0575
    "release_deriv": -1.63,               # -0.05 / 0.0308
    "anticipation_reward_deriv": 1.92,    # 0.03 / 0.0157
    "anticipation_reward_floor": 0.68,    # 0.03 / 0.0442
    "anticipation_regulation_stability": 0.98,  # 0.03 / 0.0308
    "satisfaction_peak_min_value": 1.13,  # 0.05 / 0.0442
    "satisfaction_peak_prominence": 0.68, # 0.03 / 0.0442
    "recognition_memory_deriv": 1.00,     # 0.03 / 0.0301
    "evocation_memory_deriv": 0.67,       # 0.02 / 0.0301
    "evocation_affect_deriv": 0.58,       # 0.02 / 0.0345
    "word_focus_language": 1.86,          # 0.20 / 0.1073
    "word_focus_intero_max": 1.10,        # 0.08 / 0.0725
    "word_recede_lang_deriv": -1.54,      # -0.05 / 0.0326
    "word_recede_other_deriv": 0.55,      # 0.02 / ~0.037 (mean of intero+affect deriv stds)
    "inward_pivot_lang_deriv": -0.61,     # -0.02 / 0.0326
    "inward_pivot_intero_deriv": 1.27,    # 0.05 / 0.0395
    "pattern_break_total_change": 9.36,   # 0.30 / 0.0320 (Jobs mean deriv std)
    "pattern_break_min_big_axes": 2,      # count — unchanged
    "pattern_break_axis_threshold": 2.18, # 0.07 / 0.0320
    "sensation_flood_intero": 2.07,       # 0.15 / 0.0725
    "sensation_flood_affect": 1.74,       # 0.10 / 0.0575
    "drift_max_value": 0.70,              # 0.05 / 0.0710 (Jobs mean val std)
    "drift_min_duration": 3,              # count — unchanged
    "vulnerability_social_deriv": 0.73,   # 0.03 / 0.0412
    "vulnerability_intero_deriv": 0.76,   # 0.03 / 0.0395
}

AXES = ("interoception", "core_affect", "regulation", "reward", "memory", "social", "language")


def compute_axis_stats(timeline: list) -> dict:
    """Compute per-axis mean+std for values and derivatives across a TRIBE timeline.

    Returns:
        {"val":   {axis: {"mean": float, "std": float}, ...},
         "deriv": {axis: {"mean": float, "std": float}, ...}}

    Used to drive σ-mode threshold interpretation in detect_tier1().
    std=0 (flat axis) is clamped downstream via max(std, 1e-9).
    """
    val_stats = {}
    deriv_stats = {}
    for a in AXES:
        vals = [p["categories"][a] for p in timeline]
        val_stats[a] = {
            "mean": statistics.mean(vals) if vals else 0.0,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
        derivs = [
            timeline[i]["categories"][a] - timeline[i - 1]["categories"][a]
            for i in range(1, len(timeline))
        ]
        deriv_stats[a] = {
            "mean": statistics.mean(derivs) if derivs else 0.0,
            "std": statistics.stdev(derivs) if len(derivs) > 1 else 0.0,
        }
    return {"val": val_stats, "deriv": deriv_stats}


def _deriv(timeline: list, attr: str, i: int, lookback: int = 1) -> float:
    """Discrete derivative of a single axis (raw)."""
    if i < lookback:
        return 0.0
    return timeline[i]["categories"][attr] - timeline[i - lookback]["categories"][attr]


def detect_tier1(
    timeline: list,
    thresholds: Optional[dict] = None,
    axis_stats: Optional[dict] = None,
) -> List[LabelApplication]:
    """Apply all Tier 1 detectors to a TRIBE timeline.

    Args:
        timeline: list of {timestep, categories:{interoception, core_affect, ...}}
        thresholds: optional dict of threshold overrides. Merged with the
            mode-appropriate default (DEFAULT_THRESHOLDS if axis_stats is None,
            DEFAULT_SIGMA_THRESHOLDS otherwise).
        axis_stats: output of compute_axis_stats(timeline). If provided,
            detectors normalize each value/derivative against per-axis
            mean+std before threshold comparison — thresholds interpreted
            as σ-multipliers. If None (default), absolute mode.

    Returns:
        List of LabelApplication — one per (timestep, label) detection.
    """
    if axis_stats is None:
        T = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    else:
        T = {**DEFAULT_SIGMA_THRESHOLDS, **(thresholds or {})}

    out: List[LabelApplication] = []

    def _v(axis: str, i: int, lookback: int = 0) -> float:
        """Value at step (i - lookback). In σ-mode: raw / axis_val_std.

        TRIBE activations have zero=brain-baseline meaning, so σ-mode
        scales by std without mean-centering. This preserves "above
        baseline by N × per-video-std" semantics and makes σ-thresholds
        a faithful port of absolute thresholds (abs = σ × std).
        """
        raw = timeline[i - lookback]["categories"][axis]
        if axis_stats is None:
            return raw
        return raw / max(axis_stats["val"][axis]["std"], 1e-9)

    def _dv(axis: str, i: int, lookback: int = 1) -> float:
        """Derivative at step i. In σ-mode: raw_deriv / axis_deriv_std."""
        if i < lookback:
            return 0.0
        raw = timeline[i]["categories"][axis] - timeline[i - lookback]["categories"][axis]
        if axis_stats is None:
            return raw
        return raw / max(axis_stats["deriv"][axis]["std"], 1e-9)

    for i, step in enumerate(timeline):
        t = step.get("timestep", i)

        def add(name: str, intensity: float, signals: dict):
            out.append(LabelApplication(
                label=name, tier=1,
                start_sec=float(t), end_sec=float(t + 1),
                intensity=max(0.0, min(1.0, intensity)),
                confidence=1.0,
                signals=signals))

        # body-turn
        dv_intero = _dv("interoception", i)
        dv_lang = _dv("language", i)
        if dv_intero > T["body_turn_intero_deriv"] and dv_lang < T["body_turn_lang_deriv"]:
            add("body-turn", _v("interoception", i) * 2,
                {"intero_deriv": dv_intero, "lang_deriv": dv_lang})

        # body-surge
        if dv_intero > T["body_surge_deriv"]:
            add("body-surge", dv_intero * 3,
                {"intero_deriv": dv_intero})

        # body-anchor
        v_intero = _v("interoception", i)
        v_affect = _v("core_affect", i)
        v_lang = _v("language", i)
        if v_intero > T["body_anchor_intero"] \
           and abs(dv_intero) < T["body_anchor_stability"] \
           and abs(v_affect) < T["body_anchor_affect_max"] \
           and abs(v_lang) < T["body_anchor_lang_max"]:
            add("body-anchor", v_intero * 2,
                {"intero": v_intero})

        # sensation-flood
        if v_intero > T["sensation_flood_intero"] \
           and abs(v_affect) > T["sensation_flood_affect"]:
            add("sensation-flood", (v_intero + abs(v_affect)) * 2,
                {"intero": v_intero, "affect": v_affect})

        # affect-rise
        dv_affect = _dv("core_affect", i)
        if dv_affect > T["affect_rise_deriv"]:
            add("affect-rise", abs(v_affect) * 3,
                {"affect_deriv": dv_affect})

        # affect-fade
        if i >= 2:
            prev_affect = _v("core_affect", i, lookback=2)
            if abs(prev_affect) > T["affect_fade_prev_peak"] \
               and abs(v_affect) < T["affect_fade_curr_max"]:
                add("affect-fade", 0.6, {"prev_affect": prev_affect})

        # restraint
        v_reg = _v("regulation", i)
        dv_reg = _dv("regulation", i)
        if v_reg > T["restraint_regulation"] \
           and abs(dv_reg) < T["restraint_stability"] \
           and abs(v_affect) > T["restraint_affect_floor"]:
            add("restraint", v_reg * 3,
                {"regulation": v_reg, "affect": v_affect})

        # release
        if dv_reg < T["release_deriv"]:
            add("release", abs(dv_reg) * 5, {"regulation_deriv": dv_reg})

        # anticipation
        v_reward = _v("reward", i)
        dv_reward = _dv("reward", i)
        if dv_reward > T["anticipation_reward_deriv"] \
           and abs(dv_reg) < T["anticipation_regulation_stability"] \
           and v_reward > T["anticipation_reward_floor"]:
            add("anticipation", v_reward * 8,
                {"reward_deriv": dv_reward, "reward": v_reward})

        # satisfaction-peak
        if i >= 2 and i < len(timeline) - 1:
            prev2 = _v("reward", i, lookback=2)
            prev = _v("reward", i, lookback=1)
            curr = v_reward
            if prev > prev2 and prev > curr \
               and prev > T["satisfaction_peak_min_value"] \
               and (prev - min(prev2, curr)) > T["satisfaction_peak_prominence"]:
                add("satisfaction-peak", prev * 6, {"peak_value": prev})

        # recognition
        dv_mem = _dv("memory", i)
        v_mem = _v("memory", i)
        if dv_mem > T["recognition_memory_deriv"]:
            add("recognition", abs(v_mem) * 8, {"memory_deriv": dv_mem})

        # evocation
        if dv_mem > T["evocation_memory_deriv"] and dv_affect > T["evocation_affect_deriv"]:
            add("evocation", (v_mem + abs(v_affect)) * 3,
                {"memory_deriv": dv_mem, "affect_deriv": dv_affect})

        # word-focus
        if v_lang > T["word_focus_language"] \
           and v_intero < T["word_focus_intero_max"]:
            add("word-focus", v_lang * 3,
                {"language": v_lang})

        # word-recede
        if dv_lang < T["word_recede_lang_deriv"] \
           and (dv_intero > T["word_recede_other_deriv"]
                or dv_affect > T["word_recede_other_deriv"]):
            add("word-recede", abs(dv_lang) * 4, {"lang_deriv": dv_lang})

        # inward-pivot
        if dv_lang < T["inward_pivot_lang_deriv"] \
           and dv_intero > T["inward_pivot_intero_deriv"]:
            add("inward-pivot", (abs(dv_lang) + dv_intero) * 2,
                {"lang_deriv": dv_lang, "intero_deriv": dv_intero})

        # pattern-break
        changes = [abs(_dv(a, i)) for a in AXES]
        total = sum(changes)
        big = sum(1 for c in changes if c > T["pattern_break_axis_threshold"])
        if total > T["pattern_break_total_change"] and big >= T["pattern_break_min_big_axes"]:
            add("pattern-break", min(total / 7.0, 1.0),
                {"total_change": total, "big_axes": big})

        # drift — requires N consecutive low-activation seconds
        all_vals = [abs(_v(a, i)) for a in AXES]
        if max(all_vals) < T["drift_max_value"]:
            dur = T["drift_min_duration"]
            qualifies = True
            for k in range(1, dur):
                if i - k < 0:
                    qualifies = False
                    break
                if max(abs(_v(a, i, lookback=k)) for a in AXES) >= T["drift_max_value"]:
                    qualifies = False
                    break
            if qualifies:
                add("drift", 0.5, {"max_axis_value": max(all_vals), "duration_met": True})

        # vulnerability-transfer
        dv_social = _dv("social", i)
        v_social = _v("social", i)
        if dv_social > T["vulnerability_social_deriv"] \
           and dv_intero > T["vulnerability_intero_deriv"]:
            add("vulnerability-transfer", (v_social + v_intero) * 2,
                {"social_deriv": dv_social, "intero_deriv": dv_intero})

    return out
