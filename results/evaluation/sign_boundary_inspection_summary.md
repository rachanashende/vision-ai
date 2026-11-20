# Sign-boundary inspection

Inspected 17 local videos using `C:\Users\Luzain Sara\ASL_Recognition_Project\dataset\WLASL\start_kit\WLASL_v0.3.json`.
All clips require manual boundary review. Proposed start/end values remain blank.

WLASL frame_start/frame_end are 1-based source annotations; -1 means source EOF.
Local review proposals must use zero-based inclusive decoded frame indices.
Metadata FPS can differ from local FPS; source frame numbers cannot be copied blindly.
No clip is labelled already_trimmed or untrimmed without verified provenance.
unknown means insufficient evidence; needs_review flags missing data or a mismatch.

Landmark presence alone cannot identify sign boundaries: hands may be detected at rest,
absent during signing, or unobserved because inference failed. Rest/transition frames
can coexist with a pre-cut clip. Short duration or an annotation outside local duration
is ambiguous evidence, not proof of prior trimming. Sampled frames cannot verify sign
correctness or every-frame tracking. Full temporal review is still required.

Saved masks are aligned by source frame index, decoded length, identity, and FPS.
No content checksum exists in NPZs, so unchanged video content cannot be proven.
Timestamps use OpenCV source positions, with labelled FPS estimates when unavailable.
Twelve evenly spaced samples include endpoints; very short videos repeat labelled frames.

The five provisional_keep decisions are retained as context, not boundary verification.
Original videos, NPZs, metadata, and manual decisions are unchanged. No inference,
cropping, automatic proposals, trimming, or training was performed.

| Clip | Decoded frames | Local FPS | WLASL start/end | Trim status | Inspection |
| --- | ---: | ---: | --- | --- | --- |
| before/05728 | 26 | 23.976146807791224 | 1/-1 | unknown | inspected |
| before/05729 | 52 | 29.97002997002997 | 1/-1 | unknown | inspected |
| book/07068 | 68 | 29.97 | 1/-1 | unknown | inspected |
| book/07069 | 30 | 23.976023976023978 | 1/-1 | unknown | inspected |
| book/69241 | 75 | 29.97002997002997 | 1/-1 | unknown | inspected |
| chair/09848 | 96 | 25.0 | 1/-1 | unknown | inspected |
| chair/09849 | 81 | 29.97002997002997 | 1/-1 | unknown | inspected |
| clothes/11310 | 89 | 29.97002997002997 | 1/-1 | unknown | inspected |
| computer/12311 | 72 | 29.97 | 1/-1 | unknown | inspected |
| drink/17710 | 70 | 29.97 | 1/-1 | unknown | inspected |
| drink/17711 | 81 | 29.97002997002997 | 1/-1 | unknown | inspected |
| go/24941 | 66 | 29.97 | 1/-1 | unknown | inspected |
| help/27206 | 71 | 29.97 | 1/-1 | unknown | inspected |
| help/27207 | 58 | 29.97002997002997 | 1/-1 | unknown | inspected |
| who/63227 | 36 | 29.97002997002997 | 1/-1 | unknown | inspected |
| who/69534 | 63 | 29.97002997002997 | 1/-1 | unknown | inspected |
| yes/64284 | 56 | 29.97002997002997 | 1/-1 | unknown | inspected |

## Ambiguities and errors

- before/05728: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- before/05729: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- book/07068: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- book/07069: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- book/69241: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- chair/09848: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- chair/09849: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- clothes/11310: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- computer/12311: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- drink/17710: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- drink/17711: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- go/24941: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- help/27206: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- help/27207: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- who/63227: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- who/69534: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 
- yes/64284: WLASL marks source frame 1 through EOF; this does not prove local trimming status or sign onset/offset. 

Manual workflow: inspect the sheet, then watch the entire source clip around candidate
onset/offset frames. Record only manually verified local boundaries and the reviewer’s
evidence in the CSV. Check 0 <= start <= end < decoded_frame_count using
validate_proposed_boundaries(..., manually_verified=True). Do not run --overwrite
after recording decisions unless intentional replacement of generated reviews is desired.
