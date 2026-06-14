# Future Work

The first implementation intentionally stays heuristic and inspectable. These are the richer directions to preserve for later iterations.

## Temporal Refinement

- Learn squat-specific joint velocity distributions and reject or downweight implausible jumps.
- Replace fixed jump thresholds with probability scores conditioned on joint, frame rate, and squat phase.
- Add Kalman filtering or Rauch-Tung-Striebel smoothing for trajectories where online estimates are useful but retrospective video analysis is allowed.
- Explore energy-minimization approaches for bridging occlusions, inspired by temporal smoothing work for 3D human pose estimation.
- Evaluate SmoothNet-style plug-and-play pose refinement if heuristic smoothing becomes the limiting factor.

## Physical And Biomechanical Constraints

- Track expected segment lengths per athlete/video and use them as soft constraints instead of simple relative-change flags.
- Add inverse-kinematics-style lower-body constraints for hip, knee, and ankle motion.
- Model foreshortening explicitly so apparent segment-length changes from hip abduction or camera angle are not over-penalized.
- Compare 2D image landmarks with MediaPipe world landmarks to see which better supports depth decisions.

## Squat-Specific Analysis

- Segment multiple reps and report each rep separately.
- Detect setup, descent, bottom, ascent, and lockout phases.
- Add camera-angle warnings when the view is not sufficiently side-on.
- Estimate hip crease and top-of-knee surface positions instead of using joint-center pose landmarks directly.
- Calibrate anatomy-specific offsets using segmentation masks, clothing/body contours, or manually labeled frames.
- Add richer feedback beyond depth, such as excessive forward knee travel or torso collapse, only after depth is reliable.

## Evaluation

- Build a small manually labeled validation set with clear high, clear depth, and borderline squats.
- Label bottom frame, visible side, camera angle quality, and depth outcome.
- Track false positives separately from false negatives because their costs differ for a judging-style tool.
- Preserve raw, cleaned, and flagged landmark traces for every validation video.

## References

- MediaPipe Pose Landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
- SmoothNet: https://arxiv.org/abs/2112.13715
- Temporal Smoothing for 3D Human Pose Estimation and Localization for Occluded People: https://arxiv.org/abs/2011.00250
- KinePose: https://arxiv.org/abs/2207.12841
