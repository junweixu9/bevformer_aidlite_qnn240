# Environment responsibilities

This project uses three separate environments. The word `Host` must not be used ambiguously.

## Container A: algorithm and ONNX/ORT reference environment

Container A is responsible for:

- PyTorch model execution;
- ONNX export and graph inspection;
- ONNX Runtime FP32 reference inference;
- tensor-by-tensor numerical comparison;
- local visualization;
- task-level evaluation and accuracy analysis.

Container A is the primary environment for ONNX/ORT work.

## Container B: repository and deployment-control environment

Container B is responsible for:

- Git clone, pull, and source management;
- Python syntax compilation;
- Shell syntax checks;
- JSON contract validation;
- board-independent unit tests;
- SSH/SCP orchestration;
- QCS8550 preflight;
- remote execution and result pull;
- report collection and corrected coordinate verification.

Container B does not execute the formal ONNX/ORT reference pipeline. `tools/preflight_host.sh` means the Container-B control-side preflight, not Container-A ONNX inference.

## QCS8550 board: AidLite QNN2.40 runtime environment

The board is responsible for:

- loading the three validated QNN2.40 Context files;
- Backbone, Encoder, and Decoder execution through AidLite;
- live `prev_bev` temporal recursion;
- board-CPU NumPy NMSFreeCoder;
- generation of ordered 3D coordinate outputs.

## Delivery boundary

The compact delivery repository supports Container-B orchestration and direct QCS8550 execution. Container-A ONNX/ORT validation remains an external reference workflow and is not required for the one-command board delivery path.
