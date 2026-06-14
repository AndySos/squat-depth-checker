"""Temporal cleaning for pose landmark trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import warnings

import numpy as np

from .pose import LandmarkFrame


@dataclass(frozen=True)
class CleanedPose:
    raw_landmarks: np.ndarray
    landmarks: np.ndarray
    visibility: np.ndarray
    timestamps_ms: np.ndarray
    frame_indices: np.ndarray
    low_confidence: np.ndarray
    interpolated: np.ndarray
    jump_flags: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.landmarks.shape[0])


def clean_trajectory(
    frames: Sequence[LandmarkFrame],
    min_visibility: float = 0.5,
    max_gap: int = 5,
    median_window: int = 5,
    jump_threshold: float = 0.12,
) -> CleanedPose:
    """Interpolate short gaps, smooth jitter, and flag large joint jumps."""

    if not frames:
        raise ValueError("At least one landmark frame is required")

    raw = np.stack([frame.landmarks for frame in frames]).astype(float)
    visibility = np.stack([frame.visibility for frame in frames]).astype(float)
    timestamps = np.array([frame.timestamp_ms for frame in frames], dtype=int)
    frame_indices = np.array([frame.frame_index for frame in frames], dtype=int)

    low_confidence = (visibility < min_visibility) | np.isnan(raw[:, :, 0])
    masked = raw.copy()
    masked[low_confidence] = np.nan

    interpolated_values, interpolated_flags = interpolate_short_gaps(masked, max_gap=max_gap)
    smoothed = rolling_nanmedian(interpolated_values, window=median_window)
    jump_flags = flag_large_jumps(smoothed, threshold=jump_threshold)

    return CleanedPose(
        raw_landmarks=raw,
        landmarks=smoothed,
        visibility=visibility,
        timestamps_ms=timestamps,
        frame_indices=frame_indices,
        low_confidence=low_confidence,
        interpolated=interpolated_flags,
        jump_flags=jump_flags,
    )


def interpolate_short_gaps(values: np.ndarray, max_gap: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Linearly fill NaN runs up to `max_gap` between valid observations."""

    if max_gap < 0:
        raise ValueError("max_gap must be >= 0")

    filled = values.copy()
    interpolated = np.zeros(values.shape[:2], dtype=bool)
    frame_count, joint_count, coord_count = values.shape

    for joint in range(joint_count):
        valid = ~np.isnan(values[:, joint, 0])
        valid_indices = np.flatnonzero(valid)
        if len(valid_indices) < 2:
            continue

        for start_valid, end_valid in zip(valid_indices[:-1], valid_indices[1:]):
            gap = end_valid - start_valid - 1
            if gap <= 0 or gap > max_gap:
                continue
            start = values[start_valid, joint]
            end = values[end_valid, joint]
            if np.isnan(start).any() or np.isnan(end).any():
                continue
            for offset in range(1, gap + 1):
                alpha = offset / (gap + 1)
                frame_index = start_valid + offset
                filled[frame_index, joint, :coord_count] = (1.0 - alpha) * start + alpha * end
                interpolated[frame_index, joint] = True

    # Preserve leading/trailing NaNs; they are not anchored by two good frames.
    _ = frame_count
    return filled, interpolated


def rolling_nanmedian(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply a centered rolling median, ignoring NaNs."""

    if window <= 1:
        return values.copy()
    if window % 2 == 0:
        raise ValueError("window must be odd so smoothing is centered")

    radius = window // 2
    smoothed = values.copy()
    frame_count = values.shape[0]
    for frame in range(frame_count):
        start = max(0, frame - radius)
        end = min(frame_count, frame + radius + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            candidate = np.nanmedian(values[start:end], axis=0)
        current_missing = np.isnan(smoothed[frame])
        smoothed[frame][~np.isnan(candidate)] = candidate[~np.isnan(candidate)]
        smoothed[frame][current_missing & np.isnan(candidate)] = np.nan
    return smoothed


def flag_large_jumps(values: np.ndarray, threshold: float = 0.12) -> np.ndarray:
    """Flag frames whose joint displacement from the prior frame is too large."""

    if threshold <= 0:
        raise ValueError("threshold must be positive")

    jumps = np.zeros(values.shape[:2], dtype=bool)
    if values.shape[0] < 2:
        return jumps

    deltas = np.linalg.norm(values[1:, :, :2] - values[:-1, :, :2], axis=2)
    invalid = np.isnan(deltas)
    jumps[1:] = (deltas > threshold) & ~invalid
    return jumps
