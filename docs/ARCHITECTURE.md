# Architecture and file guide

## 1. Two supported entries

### Direct board execution

```text
python/run_test.py
  → python/bevformer.py
  → frozen performance runner
  → python/acceptance.py
```

This is the primary AidLite-style entry. It runs on QCS8550 and returns success only after runtime completion and corrected coordinate verification.

### Container-B remote execution

```text
tools/run_remote.sh
  → tools/run_board.sh
  → Host preflight
  → Board preflight
  → deploy runtime files
  → invoke python/run_test.py on QCS8550
  → pull outputs
  → independent Host coordinate verification
```

The remote path does not maintain separate model inference logic.

## 2. Runtime data flow

```text
images: float32 [6,3,450,800]
  → Backbone QNN2.40 Interpreter
  → img_feat [6,256,15,25]
  → FP16 native handoff semantics, exposed to Encoder as float32
  → Encoder inputs:
       can_bus [1,18]
       img_feat [1,6,256,15,25]
       lidar2img [1,6,4,4]
       shift [1,2]
       prev_bev [1,2500,256]
  → bev_embed [1,2500,256]
  → Decoder
  → cls_scores [1,900,10] and bbox_preds [1,900,10]
  → NumPy NMSFreeCoder
  → boxes [300,9], scores [300], labels [300]
```

Frame 000 uses an all-zero `prev_bev`. Frames 001–009 use the previous live Encoder output after the validated SciPy rotation and FP16 round trip.

## 3. Python files

| File | Responsibility |
|---|---|
| `python/run_test.py` | User-facing board CLI; validates paths, runs the pipeline, and requests final acceptance. |
| `python/bevformer.py` | `BevFormerConfig` and `BevFormerPipeline`; constructs the explicit frozen-runner command. |
| `python/acceptance.py` | Runtime gates and corrected float32 coordinate contract. |
| `python/bevformer_aidlite_qnn240_e2e_performance_v1.py` | Frozen numerical implementation: three persistent AidLite Interpreters, 3 warmup frames, 10 measured frames, temporal recursion, performance reports. |
| `python/functional_mother.py` | Validated AidLite runtime helpers, tensor contracts, model SHA checks, input/output handling, and temporal rotation support used by the frozen runner. |
| `python/portable_numpy_nmsfreecoder.py` | Portable deterministic NumPy NMSFreeCoder implementation. |
| `python/verify_contract.py` | Standalone Host verifier for an existing coordinate NPZ. |
| `python/temporal.py` | Small reusable temporal-state and rotation API for tests and future refactoring. |
| `python/utils.py` | Manifest and SHA utility API. |
| `python/runtime_api.py` | Focused facade over validated AidLite runtime helpers. |
| `python/postprocess_api.py` | Focused facade over the validated NumPy postprocessor. |
| `python/run_e2e.py` | Compatibility wrapper; delegates to `run_test.py`, remote execution, or standalone verification. |

## 4. Tool files

| File | Responsibility |
|---|---|
| `tools/validate_host.sh` | Board-independent Python compile, Shell syntax, JSON contract, and unit-test validation. |
| `tools/preflight_host.sh` | Container-B files, configuration, JSON contract, compilation, and SSH checks. |
| `tools/preflight_board.sh` | Board AidLite/QNN2.40 enums, Context SHA256, and ten-frame Manifest checks. |
| `tools/run_remote.sh` | Public remote entry. |
| `tools/run_board.sh` | Deployment, board execution, result pull, and Host-side verification. |
| `tools/copy_models.sh` | Installs the three Context files only after SHA256 verification. |
| `tools/copy_demo_assets.py` | Copies validated ten-frame assets and rewrites their Manifest paths. |
| `tools/board.env.example` | Public board configuration template. The private `tools/board.env` is ignored. |

## 5. Configuration and immutable identities

- `configs/nms_runtime_contract.json`: NMSFreeCoder dimensions, range, threshold, and sigmoid mode.
- `models/EXPECTED_SHA256.txt`: validated QNN2.40 Context identities.
- `python/frame009_numpy_native_reference.npz`: ordered Frame009 coordinate reference.

## 6. Output contract

A successful run produces:

```text
performance_result.json
performance_report.txt
frame009_final_coordinates.npz
corrected_float32_tolerance_report.json
corrected_float32_tolerance_report.txt
```

Acceptance requires:

- 3 warmup frames;
- 10 measured frames;
- stable Interpreter identities;
- no runtime exception;
- successful cleanup;
- coordinate shapes exactly `(300,9)`, `(300,)`, `(300,)`;
- labels exactly equal and ordered;
- scores within `2 × float32 epsilon`;
- boxes within `8 × float32 epsilon`.

## 7. Scope boundary

This repository currently starts from preprocessed six-camera tensors. Raw image read/decode, resize, normalize, zero-copy, and official full nuScenes validation are not part of the frozen performance path.
