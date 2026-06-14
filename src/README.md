# Source

Reusable detector logic lives under `src/squat_depth/`.

Implemented first-run responsibilities:

- Video frame loading and sampling.
- Pose-estimation wrapper.
- Temporal interpolation, smoothing, and jump flags.
- Simple lower-body segment-length consistency checks.
- Bottom-frame detection.
- Squat-depth decision logic.
- Bottom-frame annotation helper.
