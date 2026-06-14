# Research Notes

This document will collect literature notes, prototype links, compute notes, and design decisions for the squat-depth checker.

## Core Direction

- Start with off-the-shelf pose estimation rather than fine-tuning.
- Assume side-view video for the first prototype.
- Use a geometric depth check at the bottom of the squat, with confidence warnings when landmarks are unreliable.
- Treat pose estimates as a trajectory, not independent snapshots. Smooth short-term jitter, interpolate short gaps, and flag implausible jumps.
- Use simple physical consistency checks, especially hip-knee and knee-ankle segment-length stability.
- Treat the overall depth decision as sustained evidence near the squat bottom, not a single-frame threshold crossing. A lone depth frame away from the cleaned bottom phase is suspicious; consecutive reliable depth frames around the bottom should increase confidence.
- For occluded multi-rep videos, reject implausible hip/knee values before smoothing, interpolate only short rejected/missing spans, and score each segmented rep separately.

## Hip Crease And Knee Caveat

The current MVP does not truly locate the hip crease or top surface of the knee. MediaPipe provides anatomical pose landmarks that behave more like joint-center estimates. For a side-view squat, the MVP uses the visible hip landmark's vertical position relative to the visible knee landmark as a first proxy for depth.

This proxy is useful for quick iteration because it is explainable and stable enough to inspect, but it is not identical to powerlifting judging language. Future versions should estimate offsets from joint centers to surface landmarks, account for camera angle and body segment thickness, and validate against manually labeled borderline squats.

## Papers And References To Review

- Pose Trainer: exercise-pose correction using pose estimation and geometric rules.
- Fitness-AQA: workout form assessment, including BackSquat examples and gym-video failure modes.
- MediaPipe Pose Landmarker: practical pose-landmark extraction for images and videos.
- MoveNet: lightweight pose-estimation baseline.
- OpenPose, RTMPose, and YOLO pose: alternatives if MediaPipe/MoveNet are insufficient.

## Open-Source Prototype Notes

Repos to inspect as references, not necessarily foundations:

- `stevenzchen/pose-trainer`
- `ParitoshParmar/Fitness-AQA`
- `NgoQuocBao1010/Exercise-Correction`
- `MichistaLin/mediapipe-Fitness-counter`
- `cavanaulton/Squat-AI`
- `rohanx01/Squat-Analysis-Model`
- `ChenMel27/powerlift-form-ai`

## Open Questions

- Which pose estimator gives the most stable hip and knee landmarks on common phone videos?
- Should the MVP use 2D image landmarks only, or also inspect MediaPipe world landmarks?
- How should confidence be reported when hip/knee landmarks are occluded or low confidence?
- What small validation set should be assembled for quick iteration?
