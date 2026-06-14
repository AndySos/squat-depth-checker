"""Simple physical consistency checks for squat pose trajectories."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose import LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE
from .temporal import CleanedPose


SEGMENT_PAIRS = (
    (LEFT_HIP, LEFT_KNEE),
    (LEFT_KNEE, LEFT_ANKLE),
    (RIGHT_HIP, RIGHT_KNEE),
    (RIGHT_KNEE, RIGHT_ANKLE),
)


@dataclass(frozen=True)
class ConstraintReport:
    segment_lengths: np.ndarray
    segment_length_flags: np.ndarray
    frame_flags: np.ndarray
    median_lengths: np.ndarray


def evaluate_constraints(
    cleaned: CleanedPose,
    max_relative_change: float = 0.35,
    segment_pairs: tuple[tuple[int, int], ...] = SEGMENT_PAIRS,
) -> ConstraintReport:
    """Flag frames with unlikely lower-body segment-length changes."""

    if max_relative_change <= 0:
        raise ValueError("max_relative_change must be positive")

    lengths = compute_segment_lengths(cleaned.landmarks, segment_pairs)
    with np.errstate(all="ignore"):
        medians = np.nanmedian(lengths, axis=0)

    relative = np.abs(lengths - medians) / np.maximum(medians, 1e-6)
    flags = (relative > max_relative_change) & ~np.isnan(relative)
    frame_flags = np.any(flags, axis=1)
    return ConstraintReport(
        segment_lengths=lengths,
        segment_length_flags=flags,
        frame_flags=frame_flags,
        median_lengths=medians,
    )


def compute_segment_lengths(
    landmarks: np.ndarray,
    segment_pairs: tuple[tuple[int, int], ...] = SEGMENT_PAIRS,
) -> np.ndarray:
    lengths = np.full((landmarks.shape[0], len(segment_pairs)), np.nan, dtype=float)
    for index, (start, end) in enumerate(segment_pairs):
        deltas = landmarks[:, start, :2] - landmarks[:, end, :2]
        lengths[:, index] = np.linalg.norm(deltas, axis=1)
    return lengths
