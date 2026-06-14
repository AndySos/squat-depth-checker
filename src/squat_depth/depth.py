"""Squat bottom-frame selection and depth classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constraints import ConstraintReport
from .pose import LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE
from .temporal import CleanedPose


SIDES = {
    "left": {"hip": LEFT_HIP, "knee": LEFT_KNEE, "ankle": LEFT_ANKLE},
    "right": {"hip": RIGHT_HIP, "knee": RIGHT_KNEE, "ankle": RIGHT_ANKLE},
}


@dataclass(frozen=True)
class DepthResult:
    label: str
    side: str
    bottom_frame_index: int
    bottom_timestamp_ms: int
    hip_knee_margin: float
    confidence: float
    low_confidence_frame_count: int
    suspicious_frame_count: int
    bottom_region_flagged: bool
    reliable_depth_frame_count: int
    max_reliable_depth_run: int
    depth_run_overlaps_bottom: bool
    reason: str


@dataclass(frozen=True)
class FrameDepthTrace:
    labels: list[str]
    margins: np.ndarray
    side: str
    flagged: np.ndarray


def analyze_depth(
    cleaned: CleanedPose,
    constraints: ConstraintReport | None = None,
    depth_margin: float = 0.0,
    bottom_window: int = 2,
    max_bottom_flagged_fraction: float = 0.4,
    min_reliable_depth_run: int = 2,
    analysis_window: tuple[int, int] | None = None,
) -> DepthResult:
    """Classify squat depth from sustained evidence near the bottom phase."""

    side = choose_visible_side(cleaned)
    joints = SIDES[side]
    hip = joints["hip"]
    knee = joints["knee"]

    hip_y = cleaned.landmarks[:, hip, 1]
    if analysis_window is None:
        window_start = 0
        window_end = cleaned.frame_count - 1
    else:
        window_start = max(0, analysis_window[0])
        window_end = min(cleaned.frame_count - 1, analysis_window[1])
    if window_end < window_start:
        return _uncertain(cleaned, side, "Invalid analysis window")

    window_slice = slice(window_start, window_end + 1)
    if np.all(np.isnan(hip_y[window_slice])):
        return _uncertain(cleaned, side, "No usable hip trajectory")

    bottom_pos = int(window_start + np.nanargmax(hip_y[window_slice]))
    start = max(0, bottom_pos - bottom_window)
    end = min(cleaned.frame_count, bottom_pos + bottom_window + 1)
    region = slice(start, end)

    bottom_low_conf = np.any(
        cleaned.low_confidence[region, [hip, knee]]
        | cleaned.implausible[region, [hip, knee]]
        | cleaned.long_occlusion[region, [hip, knee]],
        axis=1,
    )
    bottom_jumps = np.any(cleaned.jump_flags[region, [hip, knee]], axis=1)
    bottom_constraints = (
        constraints.frame_flags[region]
        if constraints is not None
        else np.zeros(end - start, dtype=bool)
    )
    bottom_flagged = bottom_low_conf | bottom_jumps | bottom_constraints
    flagged_fraction = float(np.mean(bottom_flagged)) if len(bottom_flagged) else 1.0

    margins = cleaned.landmarks[:, hip, 1] - cleaned.landmarks[:, knee, 1]
    margin = float(margins[bottom_pos])
    low_conf_count = int(np.any(cleaned.low_confidence[:, [hip, knee]], axis=1).sum())
    suspicious_count = int(np.any(cleaned.jump_flags[:, [hip, knee]], axis=1).sum())
    if constraints is not None:
        suspicious_count += int(constraints.frame_flags.sum())

    all_low_conf = np.any(
        cleaned.low_confidence[:, [hip, knee]]
        | cleaned.implausible[:, [hip, knee]]
        | cleaned.long_occlusion[:, [hip, knee]],
        axis=1,
    )
    all_jumps = np.any(cleaned.jump_flags[:, [hip, knee]], axis=1)
    all_constraints = (
        constraints.frame_flags
        if constraints is not None
        else np.zeros(cleaned.frame_count, dtype=bool)
    )
    all_flagged = all_low_conf | all_jumps | all_constraints | np.isnan(margins)
    reliable_depth = (margins > depth_margin) & ~all_flagged
    reliable_depth_in_window = reliable_depth[window_slice]
    reliable_depth_count = int(reliable_depth_in_window.sum())
    max_depth_run = _max_true_run(reliable_depth_in_window)
    bottom_mask = np.zeros(cleaned.frame_count, dtype=bool)
    bottom_mask[region] = True
    bottom_reliable_depth = reliable_depth & bottom_mask
    bottom_depth_run = _max_true_run(bottom_reliable_depth)
    depth_run_overlaps_bottom = bottom_depth_run >= min_reliable_depth_run
    analysis_mask = np.zeros(cleaned.frame_count, dtype=bool)
    analysis_mask[window_slice] = True
    any_depth_outside_bottom = bool(np.any(reliable_depth & analysis_mask & ~bottom_mask))

    if np.isnan(margin):
        label = "uncertain"
        reason = "Hip-knee margin is unavailable at the selected bottom frame"
    elif flagged_fraction > max_bottom_flagged_fraction:
        label = "uncertain"
        reason = "Too many low-confidence or physically suspicious frames near the bottom"
    elif depth_run_overlaps_bottom:
        label = "to_depth"
        reason = "Sustained reliable depth evidence overlaps the cleaned bottom phase"
    elif reliable_depth_count > 0 and any_depth_outside_bottom:
        label = "uncertain"
        reason = "Depth evidence appears away from the cleaned bottom phase"
    elif reliable_depth_count > 0:
        label = "uncertain"
        reason = "Depth evidence is too brief to trust"
    else:
        label = "not_to_depth"
        reason = "No sustained reliable depth evidence near the cleaned bottom phase"

    confidence = _confidence_from_evidence(
        flagged_fraction,
        low_conf_count,
        suspicious_count,
        cleaned.frame_count,
        bottom_depth_run,
        min_reliable_depth_run,
    )
    return DepthResult(
        label=label,
        side=side,
        bottom_frame_index=int(cleaned.frame_indices[bottom_pos]),
        bottom_timestamp_ms=int(cleaned.timestamps_ms[bottom_pos]),
        hip_knee_margin=margin,
        confidence=confidence,
        low_confidence_frame_count=low_conf_count,
        suspicious_frame_count=suspicious_count,
        bottom_region_flagged=bool(flagged_fraction > 0),
        reliable_depth_frame_count=reliable_depth_count,
        max_reliable_depth_run=max_depth_run,
        depth_run_overlaps_bottom=depth_run_overlaps_bottom,
        reason=reason,
    )


def frame_depth_trace(
    cleaned: CleanedPose,
    constraints: ConstraintReport | None = None,
    side: str | None = None,
    depth_margin: float = 0.0,
) -> FrameDepthTrace:
    """Classify every cleaned frame as to-depth, not-to-depth, or uncertain."""

    selected_side = side or choose_visible_side(cleaned)
    joints = SIDES[selected_side]
    hip = joints["hip"]
    knee = joints["knee"]

    margins = cleaned.landmarks[:, hip, 1] - cleaned.landmarks[:, knee, 1]
    low_conf = np.any(
        cleaned.low_confidence[:, [hip, knee]]
        | cleaned.implausible[:, [hip, knee]]
        | cleaned.long_occlusion[:, [hip, knee]],
        axis=1,
    )
    jumps = np.any(cleaned.jump_flags[:, [hip, knee]], axis=1)
    constraint_flags = constraints.frame_flags if constraints is not None else np.zeros(cleaned.frame_count, dtype=bool)
    flagged = low_conf | jumps | constraint_flags | np.isnan(margins)

    labels: list[str] = []
    for margin, is_flagged in zip(margins, flagged):
        if is_flagged:
            labels.append("uncertain")
        elif margin > depth_margin:
            labels.append("to_depth")
        else:
            labels.append("not_to_depth")

    return FrameDepthTrace(labels=labels, margins=margins, side=selected_side, flagged=flagged)


def choose_visible_side(cleaned: CleanedPose) -> str:
    """Choose the side with stronger average hip/knee/ankle visibility."""

    scores = {}
    for side, joints in SIDES.items():
        indices = [joints["hip"], joints["knee"], joints["ankle"]]
        with np.errstate(all="ignore"):
            scores[side] = float(np.nanmean(cleaned.visibility[:, indices]))
    return "left" if scores["left"] >= scores["right"] else "right"


def _confidence_from_evidence(
    bottom_flagged_fraction: float,
    low_conf_count: int,
    suspicious_count: int,
    frame_count: int,
    bottom_depth_run: int,
    min_reliable_depth_run: int,
) -> float:
    global_penalty = (low_conf_count + suspicious_count) / max(frame_count * 2, 1)
    run_bonus = min(bottom_depth_run / max(min_reliable_depth_run + 2, 1), 1.0)
    confidence = 0.35 + 0.35 * run_bonus - 0.4 * bottom_flagged_fraction - 0.25 * min(global_penalty, 1.0)
    return float(np.clip(confidence, 0.0, 1.0))


def _max_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _uncertain(cleaned: CleanedPose, side: str, reason: str) -> DepthResult:
    return DepthResult(
        label="uncertain",
        side=side,
        bottom_frame_index=int(cleaned.frame_indices[0]),
        bottom_timestamp_ms=int(cleaned.timestamps_ms[0]),
        hip_knee_margin=float("nan"),
        confidence=0.0,
        low_confidence_frame_count=int(np.any(cleaned.low_confidence, axis=1).sum()),
        suspicious_frame_count=int(np.any(cleaned.jump_flags, axis=1).sum()),
        bottom_region_flagged=True,
        reliable_depth_frame_count=0,
        max_reliable_depth_run=0,
        depth_run_overlaps_bottom=False,
        reason=reason,
    )
