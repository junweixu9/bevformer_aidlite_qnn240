# BEVFormer AidLite QNN2.40 refactor status

## Static repository status

- `python/run_test.py` is the direct QCS8550 inference entry.
- `tools/run_remote.sh` is the public Container-B remote entry.
- `tools/run_board.sh` deploys and invokes the same direct Python entry.
- `configs/nms_runtime_contract.json` is tracked.
- `tools/board.env.example` contains no host `PROJECT_ROOT` hardcoding.
- private `tools/board.env`, generated outputs, model binaries, and demo assets are ignored.
- host and board preflight scripts validate required source files, AidLite enums, Context SHA256 values, and the ten-frame Manifest contract.
- model and demo-asset copy tools are tracked.
- host-only contract, Manifest, NMSFreeCoder, and temporal tests are tracked.
- GitHub Actions host test workflow is tracked.

## Required runtime validation

The following items require execution in Container B and on the QCS8550 board. They are not declared PASS by repository editing alone:

```text
HOST_PREFLIGHT_GATE
BOARD_PREFLIGHT_GATE
MODEL_COPY_GATE
ASSET_INSTALL_GATE
PYTHON_COMPILE_GATE
SHELL_SYNTAX_GATE
UNIT_TEST_GATE
THREE_CONTEXT_LOAD_GATE
WARMUP_COUNT_GATE
MEASURED_FRAME_COUNT_GATE
TEMPORAL_RECURSION_GATE
CORRECTED_FLOAT32_CONTRACT_GATE
PERFORMANCE_REGRESSION_GATE
FINAL_DELIVERY_ACCEPTANCE_GATE
```

## Validation commands

Host-only checks:

```bash
python3 -m pip install -r requirements-host.txt
python3 -m py_compile python/run_test.py python/bevformer.py python/temporal.py python/utils.py python/runtime_api.py python/postprocess_api.py python/verify_contract.py
bash -n tools/run_remote.sh tools/run_board.sh tools/preflight_host.sh tools/preflight_board.sh tools/copy_models.sh
python3 -m pytest -q
```

Remote board validation:

```bash
cp tools/board.env.example tools/board.env
# edit tools/board.env
bash tools/preflight_host.sh
bash tools/preflight_board.sh
bash tools/run_remote.sh
```

Expected final marker after a successful board regression:

```text
FINAL_DELIVERY_ACCEPTANCE_GATE=PASS
```

## Scope boundary

The validated performance scope remains six-camera preprocessed tensors to ordered 3D coordinate arrays. Raw JPEG/PNG decoding, resize, normalize, zero-copy, and official full nuScenes validation are not included.
