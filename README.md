# BEVFormer AidLite QNN2.40

BEVFormer multi-camera temporal 3D detection on Qualcomm QCS8550 HTP.

## Supported pipeline

```text
six-camera preprocessed tensors
→ QNN2.40 Backbone on HTP
→ NumPy in-memory handoff
→ QNN2.40 Temporal Encoder on HTP
→ live prev_bev rotation and recursion
→ QNN2.40 Snapshot Decoder on HTP
→ NumPy NMSFreeCoder on board CPU
→ boxes (N,9), scores (N,), labels (N,)
```

The logical `images` shape is `(6,3,450,800)`. Assets may be stored as FP16 or FP32 according to the manifest; the AidLite API receives contiguous float32 arrays. JPEG/PNG decode, resize, normalize, zero-copy, and full official nuScenes validation are outside this release.

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
  run_test.py
  bevformer.py
  temporal.py
  utils.py
  runtime_api.py
  postprocess_api.py
  bevformer_aidlite_qnn240_e2e_performance_v1.py
  functional_mother.py
  portable_numpy_nmsfreecoder.py
  verify_contract.py
  frame009_numpy_native_reference.npz
configs/
  nms_runtime_contract.json
tools/
  run_remote.sh
  run_board.sh
  copy_models.sh
  copy_demo_assets.py
  preflight_host.sh
  preflight_board.sh
  board.env.example
models/
  README.md
  EXPECTED_SHA256.txt
assets/
  README.md
tests/
outputs/
```

## Setup

Install the host-only dependencies and create the private board configuration:

```bash
python3 -m pip install -r requirements-host.txt
cp tools/board.env.example tools/board.env
```

Edit `tools/board.env`. It is ignored by Git.

## Install validated models

Copy the three existing QNN2.40 Context files into the standard project directory:

```bash
bash tools/copy_models.sh \
  /path/to/backbone_context.bin \
  /path/to/encoder_context.bin \
  /path/to/decoder_context.bin \
  ./models/QCS8550/QNN240
```

The script validates the frozen SHA256 values before and after copying.

## Install the ten-frame demonstration assets

```bash
python3 tools/copy_demo_assets.py \
  --source-manifest /path/to/original/asset_manifest.json \
  --destination ./assets/unseen10
```

The installer checks `status=PASS`, verifies `frame_indices=[0..9]`, validates every source SHA when provided, copies assets into per-frame directories, and rewrites the installed Manifest paths.

## Direct board inference

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

`python/run_test.py` is the actual inference entry. It validates all paths, calls the frozen numerically validated runner, checks warmup/measured execution, and applies the corrected float32 contract before returning its final exit code.

## Container-B remote execution

```bash
bash tools/preflight_host.sh
bash tools/preflight_board.sh
bash tools/run_remote.sh
```

`tools/run_remote.sh` is the public remote entry and forwards to `tools/run_board.sh`. The remote helper deploys the same Python entry and static contract, executes it on QCS8550, pulls results, and performs an additional host-side acceptance check.

Expected final marker:

```text
FINAL_DELIVERY_ACCEPTANCE_GATE=PASS
```

## Unit tests

The host-only tests do not require AidLite or a development board:

```bash
python3 -m pytest -q
```

They cover corrected float32 acceptance, manifest structure, deterministic NumPy NMSFreeCoder behavior, and temporal scene-start state.

## Verify existing coordinates

```bash
python3 python/verify_contract.py \
  --reference python/frame009_numpy_native_reference.npz \
  --candidate outputs/run_YYYYMMDD_HHMMSS/frame009_final_coordinates.npz \
  --report-json outputs/run_YYYYMMDD_HHMMSS/corrected_report.json \
  --report-txt outputs/run_YYYYMMDD_HHMMSS/corrected_report.txt
```

Contract:

- shapes identical;
- all values finite;
- labels exact and ordered;
- score max absolute error at most `2 × float32 epsilon`;
- box max absolute error at most `8 × float32 epsilon`.

A nonzero historical strict exit is accepted only when warmup, ten measured frames, interpreter identity, cleanup, result pull, and the independent corrected contract all pass.

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

## Current limitations

- raw image preprocessing is not part of the frozen path;
- model files and ten-frame assets are installed separately;
- NMSFreeCoder runs on board CPU;
- this is not official full nuScenes validation.
