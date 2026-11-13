# EXPERIMENTAL PROVISIONAL GRU baseline

TensorFlow 2.21.0; seed 42; parameters 8186.
Epochs run: 30; restored validation-selected epoch: 30.
Test data was excluded from fitting, early stopping, and hyperparameter selection.
Test metrics/confusion/predictions reuse one post-training forward pass. All metrics are exploratory.

| Split | Clips | Correct | Accuracy | Cross-entropy |
| --- | ---: | ---: | ---: | ---: |
| train | 17 | 5 | 0.294118 | 1.885291 |
| val | 5 | 1 | 0.200000 | 2.113671 |
| test | 6 | 1 | 0.166667 | 2.174140 |

| Gloss | Train | Val | Test |
| --- | ---: | ---: | ---: |
| book | 3 | 0 | 0 |
| drink | 2 | 0 | 1 |
| computer | 1 | 1 | 1 |
| before | 2 | 0 | 1 |
| chair | 2 | 1 | 0 |
| go | 1 | 1 | 0 |
| clothes | 1 | 0 | 1 |
| who | 2 | 1 | 1 |
| help | 2 | 0 | 1 |
| yes | 1 | 1 | 0 |

## Limitations

Only 17 training clips (1-3/class), five validation clips, and six test clips.
Validation accuracy changes by 20 percentage points per example; early stopping is a noisy heuristic.
Missing val classes: book, drink, before, clothes, help.
Missing test classes: book, chair, go, yes.
Observed signer overlaps: {'train_val': ['31', '38', '4'], 'train_test': ['36', '38'], 'val_test': ['38', '59']}.
Shared signers can inflate apparent generalization. No original split or signer assignments were changed.
Severe overfitting risk; these metrics cannot establish robust ten-class recognition.
Boundaries, trimming and sign correctness remain unverified. Input schema, live cadence and browser parity remain provisional.
Masking ignores all-zero padding only; real failed-inference frames retain their real-frame flag and are not silently dropped.
Softmax confidence is uncalibrated; there is no idle/background class or webcam demonstration yet.
No webcam application, TensorFlow.js export, dependency installation, or source-data edits occurred.
Reproducibility is limited to seeded deterministic operations where supported; cross-platform bitwise equality is not guaranteed.
