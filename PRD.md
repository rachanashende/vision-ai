# Product Requirements Document (PRD)
## Real-Time ASL Recognition for Google Meet

**Version:** 1.0  
**Date:** November 13, 2026  
**Status:** Working prototype under development  
**Target:** December 2026 demonstration

### 1. Project Overview

We are developing a Chrome extension that recognizes a limited vocabulary of American Sign Language (ASL) gestures through webcam video and displays corresponding text captions during Google Meet sessions.

The system uses MediaPipe Holistic for landmark detection, a TensorFlow/Keras GRU model for temporal sign classification, TensorFlow.js for browser inference, and a Chrome Manifest V3 extension for Google Meet integration.

This is a college prototype, not a production-grade ASL translation system. It recognizes individual signs from a limited vocabulary rather than translating continuous sign language or full conversations.

### 2. Team Responsibilities

**Member 1 — ML Developer (project owner for this PRD)**

- Explore and prepare the WLASL dataset.
- Download and validate ASL videos.
- Extract MediaPipe Holistic landmarks.
- Define the ML input feature schema.
- Normalize and resample landmark sequences.
- Train and evaluate the GRU classifier.
- Export the model to TensorFlow.js.
- Deliver sample sequences, labels and documentation to Member 3.
- Coordinate preprocessing compatibility with Member 2.

**Member 2 — Chrome Extension Developer**

- Build the Manifest V3 extension.
- Implement webcam access and Google Meet integration.
- Run live MediaPipe landmark detection.
- Manage permissions, video frames and timestamps.
- Display recognition results as captions.
- Coordinate the JavaScript landmark representation with Member 1.

**Member 3 — Integration and Sequence Processing**

- Build the JavaScript sequence buffer and prediction interface.
- Validate real landmark sequences.
- Implement browser-side temporal resampling and preprocessing.
- Integrate TensorFlow.js inference.
- Verify compatibility between live sequences and the ML model.
- Coordinate output labels, confidence and idle-state handling.

Exact responsibility boundaries should be confirmed with the team.

### 3. Product Goals

1. Recognize a small set of isolated ASL signs.
2. Perform inference locally in the browser where feasible.
3. Display recognized words as captions during a Google Meet demonstration.
4. Keep inference sufficiently responsive for a live prototype.
5. Use a consistent preprocessing contract between Python training and JavaScript inference.

### 4. Initial Vocabulary

The initial ten candidate glosses are:

- book
- drink
- computer
- before
- chair
- go
- clothes
- who
- help
- yes

The final demonstration vocabulary may be reduced if data availability or model performance is insufficient.

### 5. Dataset

**Source:** WLASL dataset  
**Metadata:** `dataset/WLASL/start_kit/WLASL_v0.3.json`

Original metadata contains approximately:

- 2,000 glosses
- 21,083 instances
- 119 signers

The ten selected glosses have 277 metadata instances.

**Current local dataset as of November 13, 2026:**

| Split | Valid processed clips |
|---|---:|
| Training | 17 |
| Validation | 5 |
| Test | 6 |
| Total | 28 |

All 28 clips have corresponding extracted landmark archives.

The original WLASL train/validation/test assignments and signer identifiers must be preserved. Do not randomly reassign splits or introduce train/test leakage.

**Limitations:**

- Many remote WLASL video URLs are unavailable.
- Class coverage is sparse and uneven.
- Several classes lack validation or test examples.
- Sign boundaries and whether downloaded videos are already trimmed remain unverified.
- Current dataset size is insufficient to claim reliable real-world recognition accuracy.

### 6. Existing ML Pipeline

The existing implementation includes:

**Dataset exploration and downloading**
- `scripts/explore_dataset.py`
- `scripts/check_video_availability.py`
- `scripts/download_pilot_videos.py`
- `scripts/plan_dataset_expansion.py`
- `scripts/download_expansion_videos.py`

**Video inspection and landmark extraction**
- `scripts/inspect_videos.py`
- `scripts/test_landmark_extraction.py`
- `scripts/visualize_landmarks.py`
- `scripts/extract_landmarks_batch.py`
- `scripts/process_new_videos.py`

**Quality review**
- `scripts/analyze_landmark_quality.py`
- `scripts/review_flagged_clips.py`
- `scripts/validate_manual_review_decisions.py`
- `scripts/inspect_sign_boundaries.py`

### 7. MediaPipe and Landmark Data

The project uses:

- Python 3.11
- MediaPipe Tasks API
- MediaPipe Holistic landmarker
- Model file: `models/holistic_landmarker.task`
- OpenCV, NumPy and TensorFlow

The installed MediaPipe version uses the Tasks API. Do not assume legacy `mp.solutions` APIs are available.

Landmark archives are stored as:

`dataset/processed/<gloss>/<video_id>.npz`

Existing archives contain variable-length landmark sequences and associated metadata such as frame timestamps, detection masks, original split and signer information.

**The exact array names, shapes, coordinate conventions, normalization and landmark ordering must be read from the existing code and schema documentation. Do not guess them.**

### 8. ML-to-Extension Input Contract

This is the highest-priority unresolved integration requirement.

Member 3 needs:

1. Five to ten real landmark sequences in `.npy` format for testing.
2. An exact feature layout, including ordering and normalization.
3. A missing-landmark representation.
4. A deterministic conversion to 60 frames.
5. A model input/output specification and labels.

The team has discussed a potential `(60, 149)` input shape, but **149 features has not been verified or finalized**.

The ML developer must inspect the actual extracted landmark schema before confirming any feature dimension.

Python and JavaScript preprocessing must agree on:

- Landmark selection and ordering.
- Left/right hand identity.
- Coordinate dimensions and units.
- Normalization reference.
- Missing-landmark fill values and presence masks.
- Frame timestamps and sequence duration.
- Frame sampling/resampling.
- Handling of short clips and incomplete tracking.
- Mirroring and camera orientation.

The proposed 60-frame sequence length is provisional until the team agrees on the resampling algorithm.

Do not silently substitute a different feature format in either environment.

### 9. Model Requirements

**Planned architecture:** TensorFlow/Keras GRU sequence classifier.

Expected pipeline:

1. Read extracted NPZ landmark archives.
2. Apply the agreed feature contract.
3. Create fixed-length training sequences.
4. Train a baseline GRU model.
5. Evaluate using the original validation/test splits where coverage permits.
6. Inspect class-wise results and confusion matrix.
7. Convert the trained model to TensorFlow.js.
8. Verify Python and browser inference consistency.

The model has **not yet been trained or exported**.

The final output label order, number of classes, confidence thresholds and optional idle/no-sign behavior remain undecided.

A placeholder model, if used for integration testing, must be clearly labeled as a non-trained test artifact.

### 10. Chrome Extension Requirements

The Chrome extension is expected to:

- Run under Manifest V3.
- Obtain camera frames through an appropriate extension page or supported document context.
- Extract landmarks in the browser.
- Maintain a temporal sequence buffer.
- Apply preprocessing identical to Python.
- Run TensorFlow.js inference.
- Display predicted sign captions in the Google Meet experience.

The precise extension architecture and permissions are Member 2's responsibility.

Member 2 and Member 3 must agree on timestamps, frame rates, camera mirroring and UI behavior when no prediction is available.

### 11. Current Development Status

**Completed:**
- Python environment and dataset exploration.
- Selection of ten candidate signs.
- Pilot and expansion downloads.
- Validation of 28 local videos.
- MediaPipe landmark extraction for all 28.
- Landmark quality analysis and flagged-clip visual review for the original 17.
- Provisional review decisions for five flagged clips.
- Sign-boundary contact sheets for the original 17.
- Dataset expansion planning.
- Safe, resumable downloader.
- New-video validation and extraction workflows.

**Pending:**
- Exact ML feature contract.
- Export of integration-ready `.npy` samples.
- Agreement on 60-frame resampling.
- Expanded class and split coverage.
- Sign-boundary verification.
- GRU training and evaluation.
- TensorFlow.js model export.
- Browser integration and end-to-end testing.

### 12. Immediate Priority: Unblock Member 3

The next engineering task is to inspect the existing NPZ files and define a testable ML-to-JavaScript input contract.

Expected deliverables:

- `results/evaluation/ml_extension_contract.md`
- `results/evaluation/ml_extension_handoff.md`
- `scripts/export_integration_samples.py`
- Five to ten real `.npy` sample sequences.
- A manifest identifying the original gloss, video ID and split.
- Tests confirming shape, finite values, deterministic preprocessing and preservation of existing files.

Mark the contract **PROVISIONAL** until Member 2 and Member 3 confirm compatibility.

Do not begin GRU training based on an unconfirmed feature layout.

### 13. Safety and Engineering Rules

- Inspect existing code before editing.
- Reuse working scripts instead of duplicating functionality.
- Preserve original videos and NPZ files.
- Never overwrite review decisions or reports without explicit authorization.
- Keep original WLASL splits and signer metadata.
- Use deterministic preprocessing.
- Write tests for feature ordering and data integrity.
- Do not invent performance metrics.
- Do not treat sign-boundary detection as solved.
- Do not claim the model is trained until a real training run succeeds.
- Do not include downloaded videos, model bundles or large generated datasets in Git without checking repository policy.
- Keep commands compatible with Windows PowerShell.

### 14. Development Environment and Git

Original working directory:

`C:\Users\Luzain Sara\ASL_Recognition_Project`

GitHub clone:

`C:\Users\Luzain Sara\ASL_Recognition_Git`

Feature branch:

`feature/ml-development`

The original working directory and GitHub clone are separate. Changes made in the original directory are not automatically reflected in the GitHub clone.

Before committing, synchronize only intended source code and documentation, review `.gitignore`, inspect `git status`, and avoid overwriting teammates' work.

### 15. Definition of Done for ML Integration Handoff

Member 1's integration handoff is ready when:

1. Real sample sequences are exported and validated.
2. The feature layout is documented precisely.
3. Missing-data behavior is deterministic.
4. Python and JavaScript agree on temporal preprocessing.
5. A machine-readable label mapping is available.
6. Member 3 successfully validates real sequences.
7. The eventual trained model matches the agreed input and output contract.

The first five items can be completed before model training; the remaining items require team integration and training work.