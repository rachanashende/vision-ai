# WLASL dataset expansion plan

Original metadata: `C:\Users\Luzain Sara\ASL_Recognition_Project\dataset\WLASL\start_kit\WLASL_v0.3.json`.
The original JSON and instances CSV were cross-checked for identity, split, signer, and URL consistency.
All source train/val/test assignments and signer IDs are preserved; IDs are read as text.

Initial planning target: 15 valid training clips per class, with 20 as a stretch target, plus
available original validation/test clips. These counts are neither guaranteed nor sufficient
evidence of a reliable classifier. Some classes have only 15-19 training metadata instances,
so a target of 20 cannot be reached without changing scope. Splits must not be reassigned.

Local validity was checked by full OpenCV decoding using the existing inspector. A local
validity result does not verify sign correctness, sign boundaries, trimming, or training quality.
No boundary reviews, provisional retention decisions, videos, NPZs, or existing reports were changed.

No network requests were made. Supported URL syntax is not proof of availability. Previous
pilot success/failure is historical evidence for the exact same URL, never a live availability claim.
Restricted/unsupported URLs, failed prior requests, incomplete identity metadata, and existing
invalid/misplaced files require manual review and are not recommended automatically.
The existing pilot downloader supports public direct MP4s only; YouTube/SWF require a separately
agreed supported workflow. No access controls are bypassed and no dependencies are installed.

Ranked candidates prioritize actionable original validation/test instances, then training gaps.
Within those phases, empty split coverage, metadata completeness, and new signer coverage
are prioritized, followed by historical URL success. Signer novelty is tracked separately
within each original class/split; this does not establish signer-disjoint splits.
All 277 selected metadata instances are retained in the plan when present in this dataset;
existing valid videos have blank download priority. Lower numbers mean earlier candidates.
Candidate counts below are actionable but untested remote requests; missing/invalid counts
also include unsupported and previously failed sources. Recommendations are a queue, not a promise.

| Gloss | Split | Metadata | Valid local | Missing/invalid | Untested candidates | Recommended | Gap to 15 train | Gap to 20 train |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| book | train | 30 | 3 | 27 | 4 | 4 | 12 | 17 |
| book | val | 6 | 0 | 6 | 1 | 1 |  |  |
| book | test | 4 | 0 | 4 | 1 | 1 |  |  |
| drink | train | 25 | 2 | 23 | 10 | 10 | 13 | 18 |
| drink | val | 6 | 0 | 6 | 3 | 3 |  |  |
| drink | test | 4 | 0 | 4 | 3 | 3 |  |  |
| computer | train | 20 | 1 | 19 | 11 | 11 | 14 | 19 |
| computer | val | 5 | 0 | 5 | 3 | 3 |  |  |
| computer | test | 5 | 0 | 5 | 1 | 1 |  |  |
| before | train | 18 | 2 | 16 | 8 | 8 | 13 | 18 |
| before | val | 4 | 0 | 4 | 3 | 3 |  |  |
| before | test | 4 | 0 | 4 | 4 | 4 |  |  |
| chair | train | 19 | 2 | 17 | 2 | 2 | 13 | 18 |
| chair | val | 4 | 0 | 4 | 2 | 2 |  |  |
| chair | test | 3 | 0 | 3 | 0 | 0 |  |  |
| go | train | 19 | 1 | 18 | 9 | 9 | 14 | 19 |
| go | val | 4 | 0 | 4 | 4 | 4 |  |  |
| go | test | 3 | 0 | 3 | 1 | 1 |  |  |
| clothes | train | 18 | 1 | 17 | 2 | 2 | 14 | 19 |
| clothes | val | 4 | 0 | 4 | 0 | 0 |  |  |
| clothes | test | 3 | 0 | 3 | 1 | 1 |  |  |
| who | train | 18 | 2 | 16 | 5 | 5 | 13 | 18 |
| who | val | 4 | 0 | 4 | 4 | 4 |  |  |
| who | test | 3 | 0 | 3 | 2 | 2 |  |  |
| help | train | 15 | 2 | 13 | 8 | 8 | 13 | 18 |
| help | val | 4 | 0 | 4 | 1 | 1 |  |  |
| help | test | 3 | 0 | 3 | 2 | 2 |  |  |
| yes | train | 15 | 1 | 14 | 7 | 7 | 14 | 19 |
| yes | val | 4 | 0 | 4 | 2 | 2 |  |  |
| yes | test | 3 | 0 | 3 | 1 | 1 |  |  |

A separate explicit download request is required before implementing or running expansion
downloads. Do not use the old pilot command as an expansion downloader: it selects at most
three videos per class and does not consume this plan. This planner has no download mode.
