# PROVISIONAL GRU dataset preparation

Dataset: `gru-dataset-PROVISIONAL_v1`; contract: `asl-extension-149-v0.1-PROVISIONAL`.
Feature packing and timestamp resampling call export_integration_samples.convert_sequence directly.
Arrays were recomputed twice for determinism and compared exactly with all ten real integration fixtures.
No model was trained or exported. Member 2/3 JavaScript parity and final contract confirmation remain pending.

| Class index | Gloss | Train | Val | Test |
| ---: | --- | ---: | ---: | ---: |
| 0 | book | 3 | 0 | 0 |
| 1 | drink | 2 | 0 | 1 |
| 2 | computer | 1 | 1 | 1 |
| 3 | before | 2 | 0 | 1 |
| 4 | chair | 2 | 1 | 0 |
| 5 | go | 1 | 1 | 0 |
| 6 | clothes | 1 | 0 | 1 |
| 7 | who | 2 | 1 | 1 |
| 8 | help | 2 | 0 | 1 |
| 9 | yes | 1 | 1 | 0 |

## Saved array schema

`X_<split>.npy`: (N,60,149), float32; `y_<split>.npy`: (N,), int64 sparse class indices.
Use labels.json for the one shared PROVISIONAL class mapping, including classes absent in a split.
Manifest samples map each split row to original gloss, video ID, signer, source checksum and sampling slots.
All original splits and signer IDs are preserved. There is no split mixing, reassignment or shuffling.
Mask columns 144-148 retain left/right/pose/inference/real-frame flags. Zero coordinates alone do not mean missing.
Failed inference, absent hands and right-padding remain distinguishable. No frame is dropped for lacking hands.
Long clips use timestamp-nearest 60-frame selection; short clips retain their frames then right-pad zeros.
No extra normalization, cropping, WLASL frame offsets, or 60-frame interpolation was introduced.

## Training limitations and leakage risks

- Missing val classes: book, drink, before, clothes, help.
- Missing test classes: book, chair, go, yes.
- Cross-split signer overlap train_val: 31, 38, 4.
- Cross-split signer overlap train_test: 36, 38.
- Cross-split signer overlap val_test: 38, 59.
- Shared signers can inflate apparent generalization. Original WLASL splits are not claimed signer-disjoint; do not silently reassign them.
- Only 28 current clips: sparse training coverage creates severe overfitting and unstable evaluation risks, not measured model results.
- Validation/test class gaps prevent complete ten-class evaluation; samples are not independent evidence of reliable real-world recognition.
- Sign boundaries, source trimming and sign correctness remain unverified. Provisional_keep is not boundary verification; reviews were not changed.
- Presence masks do not guarantee landmark visibility or accuracy. Coordinate z references differ between hands and pose.
- Timestamp estimates assume constant reported FPS. Long-clip compression and short-clip padding yield different duration distributions.
- No idle/background class, confidence threshold, or browser preprocessing parity has been finalized.
- Integration fixtures include held-out examples: use them for engineering parity, not classifier tuning or training leakage.

Generation preserves outputs unless --overwrite is explicitly supplied. That flag replaces only this dataset's eight files and this summary; unrelated files are kept.
Verification uses --verify and writes nothing. Sources, reviews, raw videos, integration samples and the feature contract are never edited.
