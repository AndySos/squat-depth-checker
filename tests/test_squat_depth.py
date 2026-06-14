import unittest

import numpy as np

from squat_depth.constraints import evaluate_constraints
from squat_depth.depth import analyze_depth, frame_depth_trace
from squat_depth.pose import LandmarkFrame, LEFT_ANKLE, LEFT_HIP, LEFT_KNEE, RIGHT_ANKLE, RIGHT_HIP, RIGHT_KNEE
from squat_depth.reps import segment_reps
from squat_depth.temporal import clean_trajectory


def make_frame(index, left_hip_y, left_knee_y=0.60, visibility=0.95):
    landmarks = np.full((33, 3), np.nan, dtype=float)
    vis = np.zeros(33, dtype=float)
    presence = np.zeros(33, dtype=float)

    landmarks[LEFT_HIP] = [0.45, left_hip_y, 0.0]
    landmarks[LEFT_KNEE] = [0.46, left_knee_y, 0.0]
    landmarks[LEFT_ANKLE] = [0.47, 0.85, 0.0]
    landmarks[RIGHT_HIP] = [0.55, left_hip_y, 0.0]
    landmarks[RIGHT_KNEE] = [0.56, left_knee_y, 0.0]
    landmarks[RIGHT_ANKLE] = [0.57, 0.85, 0.0]
    vis[[LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]] = visibility
    presence[:] = vis
    return LandmarkFrame(
        frame_index=index,
        timestamp_ms=index * 33,
        landmarks=landmarks,
        visibility=vis,
        presence=presence,
    )


class SquatDepthTests(unittest.TestCase):
    def test_smooth_to_depth_squat_classifies(self):
        frames = [make_frame(i, y) for i, y in enumerate([0.40, 0.50, 0.62, 0.70, 0.62, 0.50])]
        cleaned = clean_trajectory(frames, median_window=3)
        constraints = evaluate_constraints(cleaned)
        result = analyze_depth(cleaned, constraints)
        self.assertEqual(result.label, "to_depth")
        self.assertGreater(result.hip_knee_margin, 0)

    def test_smooth_high_squat_classifies_not_to_depth(self):
        frames = [make_frame(i, y) for i, y in enumerate([0.35, 0.45, 0.52, 0.58, 0.52, 0.45])]
        cleaned = clean_trajectory(frames, median_window=3)
        result = analyze_depth(cleaned, evaluate_constraints(cleaned))
        self.assertEqual(result.label, "not_to_depth")

    def test_short_missing_gap_is_interpolated(self):
        frames = [make_frame(i, y) for i, y in enumerate([0.40, 0.50, 0.60, 0.70])]
        frames[2].visibility[LEFT_HIP] = 0.0
        cleaned = clean_trajectory(frames, max_gap=2, median_window=1, plausibility_velocity_threshold=1.0)
        self.assertTrue(cleaned.interpolated[2, LEFT_HIP])
        self.assertFalse(np.isnan(cleaned.landmarks[2, LEFT_HIP, 1]))

    def test_large_jump_is_flagged(self):
        frames = [make_frame(i, y) for i, y in enumerate([0.40, 0.42, 0.95, 0.44])]
        cleaned = clean_trajectory(frames, median_window=1, jump_threshold=0.20, plausibility_velocity_threshold=1.0)
        self.assertTrue(cleaned.jump_flags[2, LEFT_HIP])

    def test_impossible_segment_change_is_flagged(self):
        frames = [make_frame(i, 0.45) for i in range(5)]
        frames[2].landmarks[LEFT_KNEE] = [0.90, 0.55, 0.0]
        cleaned = clean_trajectory(frames, median_window=1, plausibility_velocity_threshold=1.0)
        report = evaluate_constraints(cleaned, max_relative_change=0.30)
        self.assertTrue(report.frame_flags[2])

    def test_knee_spike_above_standing_bound_is_implausible(self):
        frames = [make_frame(i, 0.45, left_knee_y=0.55) for i in range(12)]
        frames[8].landmarks[LEFT_KNEE, 1] = 0.40
        cleaned = clean_trajectory(frames, median_window=1, plausibility_velocity_threshold=1.0)
        self.assertTrue(cleaned.implausible[8, LEFT_KNEE])
        self.assertTrue(cleaned.interpolated[8, LEFT_KNEE])

    def test_long_occlusion_remains_missing_and_flagged(self):
        frames = [make_frame(i, 0.45, left_knee_y=0.55) for i in range(8)]
        for frame in frames[2:7]:
            frame.visibility[LEFT_KNEE] = 0.0
        cleaned = clean_trajectory(frames, max_gap=2, median_window=1)
        self.assertTrue(cleaned.long_occlusion[2:7, LEFT_KNEE].all())
        self.assertTrue(np.isnan(cleaned.landmarks[4, LEFT_KNEE, 1]))

    def test_sustained_bottom_occlusion_returns_uncertain(self):
        frames = [make_frame(i, y) for i, y in enumerate([0.40, 0.55, 0.70, 0.72, 0.68])]
        for frame in frames[1:4]:
            frame.visibility[LEFT_HIP] = 0.0
            frame.visibility[LEFT_KNEE] = 0.0
            frame.visibility[RIGHT_HIP] = 0.0
            frame.visibility[RIGHT_KNEE] = 0.0
        cleaned = clean_trajectory(frames, max_gap=1, median_window=1)
        result = analyze_depth(cleaned, evaluate_constraints(cleaned))
        self.assertEqual(result.label, "uncertain")

    def test_frame_depth_trace_labels_each_frame(self):
        frames = [make_frame(i, y) for i, y in enumerate([0.50, 0.57, 0.62, 0.57, 0.50])]
        cleaned = clean_trajectory(frames, median_window=1)
        trace = frame_depth_trace(cleaned, side="left")
        self.assertEqual(trace.labels, ["not_to_depth", "not_to_depth", "to_depth", "not_to_depth", "not_to_depth"])
        self.assertEqual(len(trace.margins), 5)

    def test_sustained_depth_evidence_near_bottom_classifies_to_depth(self):
        frames = [make_frame(i, y) for i, y in enumerate([0.45, 0.55, 0.62, 0.67, 0.64, 0.55])]
        cleaned = clean_trajectory(frames, median_window=1, jump_threshold=1.0)
        result = analyze_depth(cleaned, min_reliable_depth_run=2)
        self.assertEqual(result.label, "to_depth")
        self.assertTrue(result.depth_run_overlaps_bottom)
        self.assertGreaterEqual(result.max_reliable_depth_run, 2)

    def test_single_depth_blip_away_from_bottom_is_uncertain(self):
        frames = [
            make_frame(0, 0.50, left_knee_y=0.60),
            make_frame(1, 0.62, left_knee_y=0.60),
            make_frame(2, 0.56, left_knee_y=0.60),
            make_frame(3, 0.75, left_knee_y=0.80),
            make_frame(4, 0.56, left_knee_y=0.60),
        ]
        cleaned = clean_trajectory(frames, median_window=1, jump_threshold=1.0)
        result = analyze_depth(cleaned, min_reliable_depth_run=2, bottom_window=1)
        self.assertEqual(result.label, "uncertain")
        self.assertFalse(result.depth_run_overlaps_bottom)
        self.assertEqual(result.reliable_depth_frame_count, 1)

    def test_multiple_smooth_reps_are_segmented(self):
        y_values = [
            0.42, 0.48, 0.58, 0.68, 0.58, 0.48, 0.42,
            0.43, 0.49, 0.59, 0.69, 0.59, 0.49, 0.43,
        ]
        frames = [make_frame(i, y, left_knee_y=0.60) for i, y in enumerate(y_values)]
        cleaned = clean_trajectory(frames, median_window=1, plausibility_velocity_threshold=1.0)
        segments = segment_reps(cleaned, min_separation=4)
        self.assertEqual(len(segments), 2)
        self.assertEqual([segment.bottom_index for segment in segments], [3, 10])


if __name__ == "__main__":
    unittest.main()
