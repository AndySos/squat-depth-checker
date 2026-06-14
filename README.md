# Squat Depth Checker

This repository is the starting point for a squat-depth checker that takes a side-view video of a squat and determines whether the squat reached depth.

The first prototype uses off-the-shelf pose estimation, not model fine-tuning. It extracts body landmarks from video, cleans the landmark trajectory with lightweight temporal checks, identifies the bottom of the squat, and evaluates depth using sustained side-view evidence: the hip position relative to the knee position over the bottom phase, not just one threshold-crossing frame.

Important caveat: the current depth rule uses pose-estimated hip and knee landmarks as proxies for the competition concepts of hip crease and top of knee. These landmarks are closer to joint centers than surface anatomy, so the MVP should be interpreted as an explainable prototype rather than a rules-accurate judging system.

## Initial Assumptions

- Input videos are side-view recordings with one lifter in frame.
- The first version targets a hackathon-style MVP, not production coaching accuracy.
- No model fine-tuning is planned for the initial prototype.
- Private/raw lifting videos should not be committed unless they are explicitly selected for sharing.
- The future GitHub repository should be private by default.

## Planned Colab Workflow

The first notebook lives in `notebooks/squat_depth_mvp.ipynb` and is designed to run in Google Colab. The intended workflow is:

1. Open the notebook in Colab from the private GitHub repo.
2. Upload or mount a squat video.
3. Run pose estimation on the video frames.
4. Detect the bottom frame of the squat.
5. Classify the squat as to-depth or not-to-depth.
6. Export an annotated bottom frame and a per-frame annotated video with the decision and confidence warnings.

Because the planned GitHub repo is private, the Colab flow may require signing into GitHub or uploading the notebook manually until sharing details are finalized.

## Project Layout

```text
squat-depth-checker/
  README.md
  .gitignore
  docs/
    future-work.md
    research-notes.md
  examples/
    README.md
  notebooks/
    README.md
    squat_depth_mvp.ipynb
  src/
    README.md
    squat_depth/
      ...
  tests/
    test_squat_depth.py
```

## Current Status

Minimal first-run code is present. The reusable logic lives under `src/squat_depth/`, with synthetic unit tests covering trajectory smoothing, jump flags, segment-length checks, and depth classification. A real video smoke test still needs to be run in Colab or another environment with MediaPipe/OpenCV installed.
