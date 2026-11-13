# Provisional manual review decisions

All five flagged clips are recorded as **provisional_keep**, as instructed by
Member 1 (ML Developer), using the supplied observations from six sampled frames
per clip. These decisions are stored separately in
`dataset/metadata/manual_review_decisions.csv`; the existing
`flagged_visual_review.csv` is unchanged.

| Gloss / video ID | Six sampled source frame indices | Supplied observation |
| --- | --- | --- |
| clothes/11310 | 0, 18, 35, 53, 70, 88 | Both hands tracked during the visible gesture around sampled frames 53–70; resting frames lack hand detections. |
| computer/12311 | 0, 14, 28, 43, 57, 71 | Both hands tracked during the visible gesture around frames 28–43; hands lowered in other sampled frames. |
| drink/17710 | 0, 14, 28, 41, 55, 69 | Signing hand tracked near face around frame 28 and lower around frame 41. |
| drink/17711 | 0, 16, 32, 48, 64, 80 | Signing hand tracked around frames 32–48; opposite hand not visibly active in sampled gesture. |
| go/24941 | 0, 13, 26, 39, 52, 65 | Both hands tracked during the gesture around frames 26–39. |

The decisions are provisional. Six sampled frames do not establish tracking
quality across every frame, verify sign correctness, or fully establish sign
onset and offset. **Sign-boundary verification is pending for all five clips.**
Full temporal review and sign correctness verification remain necessary before
treating these as fully verified training examples. No clips are automatically
rejected or deleted by these decisions.

Conventions follow `results/evaluation/landmark_batch_schema.md`
(`asl-holistic-xyz-v1`): frame indices are zero-based source frame indices;
video IDs are text identifiers; left/right refer to MediaPipe anatomical labels.
Saved WLASL frame boundaries have not been applied or verified, and trimming
status remains uncertain. Split and signer metadata in the original reports and
NPZs are unchanged; no split reassignment is implied by provisional retention.

`review_status=provisional_sampled_frame_review` describes the limited evidence.
`sign_boundary_verification=pending` explicitly records the remaining boundary
check. The notes record the supplied observations; no new contact sheets,
MediaPipe inference, or full-video review were performed for this record.

Validate from the project root in PowerShell:

```powershell
python -B scripts/validate_manual_review_decisions.py
```

The validator uses only Python's standard library and writes no files.
