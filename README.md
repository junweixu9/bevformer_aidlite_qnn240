# BEVFormer AidLite QNN2.40

BEVFormer multi-camera temporal 3D detection on Qualcomm QCS8550 HTP. The repository follows the AidLite example style: one direct Python inference entry, explicit model and asset paths, small deployment tools, and reproducible output checks.

## Supported pipeline

```text
six-camera preprocessed tensors
→ QNN2.40 Backbone on HTP
→ NumPy in-memory handoff
→ QNN2.40 Temporal Encoder on HTP
→ live prev_bev rotation and recursion
→ QNN2.40 Snapshot Decoder on HTP
→ NumPy NMSFreeCoder on board CPU
→ boxes (300,9), scores (300,), labels (300,)
```

The logical `images` input is little-endian float32 with shape `(6,3,450,800)`. JPEG/PNG decode, resize, normalize, zero-copy, and official full nuScenes validation are outside this release.

## Validated environment

- QCS8550, HTP V73
- AidLite SDK 2.4.0.265
- QNN runtime 2.40
- `FrameworkType.TYPE_QNN240`
- Python 3.10
- NumPy 1.26.4
- SciPy 1.11.4

## Project layout

```text
README.md
VERSION
requirements-host.txt
python/
  run_test.py                    # direct QCS8550 inference entry
  bevformer.py                   # pipeline facade and explicit runner command
  acceptance.py                  # runtime gates and corrected accuracy contract
  temporal.py                    # reusable prev_bev state and rotation API
  utils.py                       # Manifest and SHA helpers
  runtime_api.py                 # focused AidLite runtime facade
  postprocess_api.py             # focused postprocess facade
  bevformer_aidlite_qnn240_e2e_performance_v1.py  # frozen numerical runner
  functional_mother.py
  portable_numpy_nmsfreecoder.py
  verify_contract.py
  frame009_numpy_native_reference.npz
configs/
  nms_runtime_contract.json
tools/
  validate_host.sh               # board-independent repository validation
  run_remote.sh                  # public Container-B remote entry
  run_board.sh                   # deploy, execute, and pull results
  preflight_host.sh
  preflight_board.sh
  copy_models.sh
  copy_demo_assets.py
  board.env.example
models/
  README.md
  EXPECTED_SHA256.txt
assets/
  README.md
tests/
docs/
  ARCHITECTURE.md
audit/
  NEXT_VALIDATION.txt
outputs/                          # generated and ignored by Git
```

See `docs/ARCHITECTURE.md` for the call graph and per-file responsibilities.

## 1. Host setup and static validation

Run in Container B or another Linux Host environment:

```bash
python3 -m pip install -r requirements-host.txt
bash tools/validate_host.sh
```

Expected final marker:

```text
HOST_VALIDATION_GATE=PASS
```

## 2. Private board configuration

```bash
cp tools/board.env.example tools/board.env
```

Edit `tools/board.env`. It is ignored by Git. Host project paths are derived from script locations and are not hardcoded in this file.

## 3. Install validated models

Run on the machine that holds the three Context files:

```bash
bash tools/copy_models.sh \
  /path/to/backbone_context.bin \
  /path/to/encoder_context.bin \
  /path/to/decoder_context.bin \
  ./models/QCS8550/QNN240
```

The script validates the frozen SHA256 identities before and after copying.

## 4. Install the ten-frame demonstration assets

Run where the source Manifest paths are accessible:

```bash
python3 tools/copy_demo_assets.py \
  --source-manifest /path/to/validated/asset_manifest.json \
  --destination ./assets/unseen10
```

The tool validates source files, copies them into per-frame directories, rewrites record paths, and writes the installed Manifest.

## 5. Direct board inference

Run on QCS8550:

```bash
python3 python/run_test.py \
  --backbone-model ./models/QCS8550/QNN240/backbone_context.bin \
  --encoder-model ./models/QCS8550/QNN240/encoder_context.bin \
  --decoder-model ./models/QCS8550/QNN240/decoder_context.bin \
  --asset-manifest ./assets/unseen10/asset_manifest.json \
  --nms-contract ./configs/nms_runtime_contract.json \
  --reference ./python/frame009_numpy_native_reference.npz \
  --output-dir ./outputs/direct_board_run
```

Execution path:

```text
run_test.py
→ BevFormerPipeline
→ frozen numerical Runner
→ acceptance.py
```

The frozen Runner remains the numerical source of truth. The public entry returns zero only when runtime completion and the corrected float32 coordinate contract pass.

## 6. Container-B remote execution

```bash
bash tools/preflight_host.sh
bash tools/preflight_board.sh
bash tools/run_remote.sh
```

The remote helper deploys the same board entry, runs it on QCS8550, pulls the output directory, and repeats coordinate verification on the Host.

Expected final marker:

```text
FINAL_DELIVERY_ACCEPTANCE_GATE=PASS
```

## 7. Verify existing coordinates without inference

```bash
python3 python/verify_contract.py \
  --reference python/frame009_numpy_native_reference.npz \
  --candidate outputs/run_YYYYMMDD_HHMMSS/frame009_final_coordinates.npz \
  --report-json outputs/run_YYYYMMDD_HHMMSS/corrected_report.json \
  --report-txt outputs/run_YYYYMMDD_HHMMSS/corrected_report.txt
```

Acceptance contract:

- shapes exactly `(300,9)`, `(300,)`, and `(300,)`;
- all floating values finite;
- labels exactly equal and ordered;
- score max absolute error at most `2 × float32 epsilon`;
- box max absolute error at most `8 × float32 epsilon`.

The original strict `1e-7` score failure is preserved. It is accepted only when warmup, ten measured frames, Interpreter identity, cleanup, no runtime exception, and the corrected contract all pass.

## Performance baseline

Protocol: 3 warmup frames and 10 measured frames. Context loading, Python startup, source-asset disk I/O, and image decode are excluded.

| Metric | Baseline |
|---|---:|
| Backbone invoke mean | 22.468505 ms |
| Encoder invoke mean | 355.982209 ms |
| Decoder invoke mean | 37.689356 ms |
| Three-model invoke mean | 416.140070 ms |
| NMSFreeCoder mean | 0.683763 ms |
| Total wall mean | 474.943284 ms |
| Total wall P95 | 481.733953 ms |

## Current validation status

Repository refactoring is complete. Host validation and the post-refactor QCS8550 regression must still be executed from Container B before declaring final teacher delivery `PASS`. The exact checklist is in `audit/NEXT_VALIDATION.txt`.

## Current limitations

- raw image preprocessing is not part of the frozen path;
- model files and ten-frame assets are installed separately;
- NMSFreeCoder runs on board CPU;
- this is not official full nuScenes validation.
