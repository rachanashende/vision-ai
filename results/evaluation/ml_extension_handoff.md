# PROVISIONAL handoff to Member 3 (with Member 2)

No trained or fake TF.js model is included. These are real saved landmark fixtures
converted under `asl-extension-149-v0.1-PROVISIONAL`, pending Member 2/3 confirmation.

Project root: `C:\Users\Luzain Sara\ASL_Recognition_Project`.

Send these paths together, preserving filenames:

- `results/evaluation/ml_extension_contract.md` — exact provisional feature/mask,
  coordinate, resampling and label contract, plus a minimal JavaScript packer.
- `results/integration_samples/PROVISIONAL_v1/manifest.json` — metadata, source hashes,
  selected indices/timestamps, padding and original splits for all ten samples.
- All ten `.npy` files in `results/integration_samples/PROVISIONAL_v1/`:
  `PROVISIONAL_book_07068.npy`, `PROVISIONAL_drink_17713.npy`,
  `PROVISIONAL_computer_12312.npy`, `PROVISIONAL_before_05728.npy`,
  `PROVISIONAL_chair_09851.npy`, `PROVISIONAL_go_24946.npy`,
  `PROVISIONAL_clothes_11311.npy`, `PROVISIONAL_who_63229.npy`,
  `PROVISIONAL_help_27209.npy`, `PROVISIONAL_yes_64287.npy`.
- `scripts/export_integration_samples.py` and `scripts/test_export_integration_samples.py`
  — reproducible Python reference and tests (run inside this project).

Each array is `(60,149)` little-endian float32, time-major: 144 xyz features and
five binary flags. `.npy` includes a header: use an NPY parser, not a raw headerless
Float32Array view. No final trained model input or verified sign segmentation is implied.
The manifest's `source_npz` paths refer to Member 1's local project; source NPZs
are needed only for full reproducibility verification, not for loading fixtures in JS.

From the project root, generate once (existing directories are preserved):

```powershell
.\.venv\Scripts\python.exe -B scripts/export_integration_samples.py
```

Minimal read-only verification against the original NPZs:

```powershell
.\.venv\Scripts\python.exe -B scripts/export_integration_samples.py --verify
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -B scripts/test_export_integration_samples.py
```

For a fresh regenerated export, supply `--output-dir results/integration_samples/PROVISIONAL_v2`
and use the same argument with `--verify`. This preserves v1 rather than overwriting it.

Member 3: check buffer shape/flattening, mask columns, missing-hand behavior,
right-padding and exact nearest-time selection (including ties) against the
Python fixtures. Member 2: confirm browser result adapter, image orientation,
anatomical handedness and shared timestamp/cadence behavior. Treat padding as
padding, not a detected resting pose. Do not drop frames based on the diagnostic
frame_valid flag. Record parity results before declaring this contract final.

The ten real fixture sources include train/val/test metadata unchanged. No source
NPZ/video is modified, no inference runs, and no model is trained/exported. All
sign boundaries remain unverified; the six-landmark pose subset, 60-step policy,
normalization and proposed label order are still team decisions.
