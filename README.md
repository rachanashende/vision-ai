# Vision AI

Real-time American Sign Language (ASL) captioning for video conferencing, using MediaPipe and deep learning.

Vision AI is a Chrome extension that recognizes ASL signs from the user's webcam during a Google Meet call and shows them as live captions. Landmark extraction and model inference are designed to run on the user's own device, so no video or landmark data needs to be sent to a server.

> **Status: work in progress.** This is a college mini project (BMS Institute of Technology and Management, CSE, 2026-27). See [Project status](#project-status) for what exists today. Performance figures below are targets, not measured results.

## Why

Video platforms caption speech automatically, but none of them recognize or caption sign language. A Deaf ASL user currently has to book an interpreter or type in chat. Public ASL recognition projects mostly stop at a webcam demo, so this project focuses on carrying the pipeline through to a working integration inside a live call.

## How it works

```
Webcam -> MediaPipe landmarks -> Validation + rolling buffer -> Sequence prep (60 x 149)
       -> GRU model (TensorFlow.js) -> Prediction filtering -> Caption overlay in Google Meet
```

1. **Video capture**: the extension reads the webcam feed in the call tab.
2. **Landmark extraction**: MediaPipe extracts hand, body and face landmarks per frame.
3. **Validation and buffering**: frames are checked for wrong dimensions, missing values, NaNs and infinities, then added to a rolling buffer of about four seconds.
4. **Sequence preparation**: the buffer is resampled to a fixed sequence, planned as 60 frames x 149 features. The Python (training) and JavaScript (inference) code must use identical logic.
5. **Model inference**: a GRU classifier, converted to TensorFlow.js, returns a prediction message `{label, confidence, timestamp}`.
6. **Prediction filtering**: a confidence threshold, a stability check (provisionally 3 consecutive matching predictions) and duplicate suppression decide what is shown as a caption.

### Planned differentiators

- On-device inference (MediaPipe and TensorFlow.js in the browser, no server).
- A finished live Google Meet integration, not just a webcam demo.
- Fingerspelling fallback for signs outside the trained vocabulary (to be confirmed with the team).

### Targets

- Inference roughly every 350 ms.
- About 15 FPS frame processing.

These are goals to measure during testing. Actual results will be recorded here once available.

## Project status

| Stage | Status |
|---|---|
| Repository and environment setup | Done |
| Sequence shape/dtype check (`validate_sequence.py`) | Done, on dummy data only |
| Real landmark validation | Not started (needs real data and the agreed feature layout) |
| Rolling buffer and resampling | Not started |
| Prediction filtering | Not started |
| Model training and TensorFlow.js conversion | Not in this repository |
| Chrome extension and caption overlay | Not in this repository |
| Google Meet integration and testing | Not started |

## Team

| Name | Area |
|---|---|
| Rachana Shende | Backend: validation, sequence preparation, prediction filtering, integration |
| Luzain Sara Mansoor | Frontend: Chrome extension, UI, caption overlay |
| Vaishnavi H | Testing and datasets |

Guide: Dr. Swetha M S.

## Open decisions

These need to be confirmed across the team before building against them:

- Final model input shape (provisionally `(1, 60, 149)`) and feature ordering.
- Landmark normalization and coordinate conventions.
- How a missing or undetected landmark is represented.
- How the four-second buffer is resampled to 60 frames.
- Model output format and label mapping.
- Where TensorFlow.js inference runs in the extension (for example an offscreen document).
- Confidence threshold and stability-filter settings.

## Getting started

Requires Python 3.11 or later.

```bash
git clone https://github.com/rachanashende/vision-ai.git
cd vision-ai

python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
python validate_sequence.py
```

Expected output ends with `Basic validation passed!`. At the moment this checks only that a dummy `(60, 149)` float32 array has the right shape, type and finite values.

## Repository layout

```
vision-ai/
├── validate_sequence.py   # sequence shape/dtype/finite-value check (dummy data)
├── requirements.txt       # Python dependencies (NumPy so far)
├── .gitignore
└── README.md
```

## Related work

- *Real Time Captioning of Sign Language Gestures in Video Meetings* (arXiv, 2025): a near-identical browser-extension approach.
- VESIT (Mumbai) 2022 patent: a Chrome extension for Google Meet that translates signs to captions.
- Open-source MediaPipe landmark-to-sequence models (BiLSTM, GRU, TFLite) for webcam ASL recognition.
- Sign-Speak: a commercial ASL-English translation product, used as a quality reference.

## Privacy

The design keeps all processing on the user's device. This will be verified during testing, including checks that no video or landmark data is stored or transmitted, and the results documented here.
