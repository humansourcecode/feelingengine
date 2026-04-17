"""Layer 2 — Change Detection.

Analyzes brain region activation time series to identify WHERE
significant changes occur. The emotional arc is defined by transitions,
not static states.

Detects:
- Change points (timesteps where activation shifts significantly)
- Direction (rising, falling, reversal)
- Magnitude (how large the shift)
- Rate (how fast — gradual vs. sudden)
- Transition type (onset, peak, plateau, decline, reversal)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from feeling_engine.adapters.brain_model.base import BrainTimeSeries


class TransitionType(str, Enum):
    ONSET = "onset"  # activation begins rising from baseline
    RAPID_SPIKE = "rapid_spike"  # sudden large increase (< 3 timesteps)
    GRADUAL_RISE = "gradual_rise"  # steady increase over 4+ timesteps
    PEAK = "peak"  # local maximum
    PLATEAU = "plateau"  # sustained high activation
    DECLINE = "decline"  # activation falling from peak
    RAPID_DROP = "rapid_drop"  # sudden large decrease
    REVERSAL = "reversal"  # direction change (rise → fall or fall → rise)
    BASELINE = "baseline"  # near-zero, stable


@dataclass
class ChangePoint:
    """A significant change in brain activation at a specific timestep."""
    timestep: int
    category: str  # which brain region changed
    activation: float  # current value
    delta: float  # change from previous timestep
    delta_magnitude: float  # absolute change
    direction: str  # "rising" or "falling"
    rate: float  # change per timestep (averaged over window)
    transition_type: TransitionType
    context: dict = field(default_factory=dict)  # surrounding timesteps for interpretation


@dataclass
class ChangeAnalysis:
    """Complete change analysis of a brain time series."""
    change_points: list[ChangePoint]
    n_timesteps: int
    categories_analyzed: list[str]
    threshold_used: float
    summary: dict = field(default_factory=dict)  # per-category summary stats


def detect_changes(
    brain_ts: BrainTimeSeries,
    threshold: float = 0.08,
    min_rate_for_rapid: float = 0.10,
    plateau_tolerance: float = 0.03,
    context_window: int = 3,
) -> ChangeAnalysis:
    """Detect significant change points in brain activation time series.

    Args:
        brain_ts: brain region activation time series (from BrainModelAdapter)
        threshold: minimum absolute delta to count as a change point
        min_rate_for_rapid: minimum rate to classify as rapid_spike/rapid_drop
        plateau_tolerance: max delta to still count as plateau
        context_window: how many surrounding timesteps to include in context

    Returns:
        ChangeAnalysis with all detected change points
    """
    change_points: list[ChangePoint] = []
    categories = brain_ts.category_names
    n = brain_ts.n_timesteps

    # Build per-category activation arrays
    cat_arrays: dict[str, np.ndarray] = {}
    for cat in categories:
        cat_arrays[cat] = np.array([
            p.categories.get(cat, 0.0) for p in brain_ts.profiles
        ])

    for cat in categories:
        values = cat_arrays[cat]
        if len(values) < 2:
            continue

        # Compute deltas
        deltas = np.diff(values)

        for t in range(1, n):
            delta = float(deltas[t - 1])
            abs_delta = abs(delta)

            if abs_delta < threshold:
                continue

            # Direction
            direction = "rising" if delta > 0 else "falling"

            # Rate (averaged over context window)
            window_start = max(0, t - context_window)
            window_end = min(n - 1, t + context_window)
            if window_end > window_start:
                window_vals = values[window_start:window_end + 1]
                rate = float((window_vals[-1] - window_vals[0]) / (window_end - window_start))
            else:
                rate = delta

            # Transition type classification
            transition = _classify_transition(
                values, t, delta, rate,
                min_rate_for_rapid=min_rate_for_rapid,
                plateau_tolerance=plateau_tolerance,
            )

            # Context: surrounding timestep values
            ctx_start = max(0, t - context_window)
            ctx_end = min(n, t + context_window + 1)
            context = {
                "window_values": values[ctx_start:ctx_end].tolist(),
                "window_timesteps": list(range(ctx_start, ctx_end)),
                "category_values_at_t": {
                    c: float(cat_arrays[c][t]) for c in categories
                },
            }

            change_points.append(ChangePoint(
                timestep=t,
                category=cat,
                activation=float(values[t]),
                delta=delta,
                delta_magnitude=abs_delta,
                direction=direction,
                rate=rate,
                transition_type=transition,
                context=context,
            ))

    # Sort by timestep, then by magnitude (most significant first within timestep)
    change_points.sort(key=lambda cp: (cp.timestep, -cp.delta_magnitude))

    # Summary stats per category
    summary = {}
    for cat in categories:
        cat_cps = [cp for cp in change_points if cp.category == cat]
        values = cat_arrays[cat]
        summary[cat] = {
            "n_changes": len(cat_cps),
            "max_activation": float(values.max()),
            "min_activation": float(values.min()),
            "mean_activation": float(values.mean()),
            "range": float(values.max() - values.min()),
            "peak_timestep": int(np.argmax(np.abs(values))),
        }

    return ChangeAnalysis(
        change_points=change_points,
        n_timesteps=n,
        categories_analyzed=categories,
        threshold_used=threshold,
        summary=summary,
    )


def _classify_transition(
    values: np.ndarray,
    t: int,
    delta: float,
    rate: float,
    min_rate_for_rapid: float,
    plateau_tolerance: float,
) -> TransitionType:
    """Classify the type of transition at timestep t."""
    n = len(values)
    current = values[t]
    prev = values[t - 1] if t > 0 else current

    # Check for peak (local maximum or minimum)
    if t < n - 1:
        next_val = values[t + 1]
        if current > prev and current > next_val:
            return TransitionType.PEAK
        if current < prev and current < next_val:
            return TransitionType.PEAK  # local minimum = trough peak

    # Check for rapid movement
    if abs(rate) >= min_rate_for_rapid:
        return TransitionType.RAPID_SPIKE if delta > 0 else TransitionType.RAPID_DROP

    # Check for plateau (high activation, low change)
    if abs(delta) <= plateau_tolerance and abs(current) > 0.15:
        return TransitionType.PLATEAU

    # Check for onset (rising from near-baseline)
    if delta > 0 and abs(prev) < 0.10:
        return TransitionType.ONSET

    # Check for reversal (direction changed from previous delta)
    if t >= 2:
        prev_delta = values[t - 1] - values[t - 2]
        if (prev_delta > 0 and delta < 0) or (prev_delta < 0 and delta > 0):
            return TransitionType.REVERSAL

    # Default: gradual rise or decline
    return TransitionType.GRADUAL_RISE if delta > 0 else TransitionType.DECLINE


def get_arc_segments(analysis: ChangeAnalysis) -> list[dict]:
    """Group change points into narrative arc segments.

    Returns a list of segments, each representing a coherent emotional
    phase (e.g., "rising tension," "peak intensity," "resolution").
    """
    if not analysis.change_points:
        return []

    segments = []
    current_segment: dict | None = None

    for cp in analysis.change_points:
        if current_segment is None:
            current_segment = {
                "start_timestep": cp.timestep,
                "end_timestep": cp.timestep,
                "dominant_category": cp.category,
                "transition_type": cp.transition_type.value,
                "peak_magnitude": cp.delta_magnitude,
                "change_points": [cp],
            }
        elif (
            cp.timestep - current_segment["end_timestep"] <= 2
            and cp.category == current_segment["dominant_category"]
        ):
            # Extend current segment
            current_segment["end_timestep"] = cp.timestep
            current_segment["peak_magnitude"] = max(
                current_segment["peak_magnitude"], cp.delta_magnitude
            )
            current_segment["change_points"].append(cp)
        else:
            # Close segment, start new one
            segments.append(current_segment)
            current_segment = {
                "start_timestep": cp.timestep,
                "end_timestep": cp.timestep,
                "dominant_category": cp.category,
                "transition_type": cp.transition_type.value,
                "peak_magnitude": cp.delta_magnitude,
                "change_points": [cp],
            }

    if current_segment:
        segments.append(current_segment)

    return segments
