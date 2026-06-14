"""Rep segmentation from cleaned squat trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .depth import DepthResult, analyze_depth, choose_visible_side, SIDES
from .constraints import ConstraintReport
from .temporal import CleanedPose


@dataclass(frozen=True)
class RepSegment:
    rep_index: int
    start_index: int
    bottom_index: int
    end_index: int


def segment_reps(
    cleaned: CleanedPose,
    min_prominence: float = 0.06,
    min_separation: int = 20,
) -> list[RepSegment]:
    """Segment repeated descent-bottom-ascent cycles from cleaned hip y."""

    side = choose_visible_side(cleaned)
    hip = SIDES[side]["hip"]
    hip_y = cleaned.landmarks[:, hip, 1]
    valid = ~np.isnan(hip_y)
    if valid.sum() < 3:
        return []

    with np.errstate(all="ignore"):
        standing_y = float(np.nanpercentile(hip_y, 20))
        deep_y = float(np.nanpercentile(hip_y, 90))
    if not np.isfinite(standing_y) or not np.isfinite(deep_y) or deep_y - standing_y < min_prominence:
        return []

    threshold = standing_y + 0.45 * (deep_y - standing_y)
    candidates = _local_maxima(hip_y, valid, min_prominence=min_prominence)
    candidates = _suppress_nearby_candidates(candidates, hip_y, min_separation=min_separation)

    segments: list[RepSegment] = []
    for bottom in candidates:
        start = bottom
        while start > 0 and valid[start - 1] and hip_y[start - 1] > threshold:
            start -= 1
        end = bottom
        while end < len(hip_y) - 1 and valid[end + 1] and hip_y[end + 1] > threshold:
            end += 1
        if end - start >= 2:
            segments.append(RepSegment(len(segments) + 1, start, bottom, end))
    return segments


def analyze_reps(
    cleaned: CleanedPose,
    constraints: ConstraintReport | None = None,
    segments: list[RepSegment] | None = None,
) -> tuple[list[RepSegment], list[DepthResult]]:
    """Return rep segments and one depth result per rep."""

    rep_segments = segments if segments is not None else segment_reps(cleaned)
    results = [
        analyze_depth(
            cleaned,
            constraints=constraints,
            analysis_window=(segment.start_index, segment.end_index),
        )
        for segment in rep_segments
    ]
    return rep_segments, results


def _local_maxima(hip_y: np.ndarray, valid: np.ndarray, min_prominence: float) -> list[int]:
    candidates = []
    for index in range(1, len(hip_y) - 1):
        if not valid[index - 1] or not valid[index] or not valid[index + 1]:
            continue
        if hip_y[index] >= hip_y[index - 1] and hip_y[index] >= hip_y[index + 1]:
            local_min = min(float(hip_y[index - 1]), float(hip_y[index + 1]))
            if float(hip_y[index]) - local_min >= min_prominence / 4:
                candidates.append(index)
    return candidates


def _suppress_nearby_candidates(candidates: list[int], hip_y: np.ndarray, min_separation: int) -> list[int]:
    selected: list[int] = []
    for candidate in sorted(candidates, key=lambda idx: hip_y[idx], reverse=True):
        if all(abs(candidate - existing) >= min_separation for existing in selected):
            selected.append(candidate)
    return sorted(selected)
