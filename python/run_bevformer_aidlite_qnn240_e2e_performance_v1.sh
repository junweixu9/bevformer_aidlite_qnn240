#!/bin/bash

set -euo pipefail

: "${BACKBONE_MODEL:?BACKBONE_MODEL is required}"
: "${ENCODER_MODEL:?ENCODER_MODEL is required}"
: "${DECODER_MODEL:?DECODER_MODEL is required}"
: "${ASSET_MANIFEST:?ASSET_MANIFEST is required}"
: "${OUTPUT_DIRECTORY:?OUTPUT_DIRECTORY is required}"

PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULT_JSON="${RESULT_JSON:-${OUTPUT_DIRECTORY}/performance_result.json}"

mkdir -p "${OUTPUT_DIRECTORY}"

/usr/bin/python3 \
  "${PACKAGE_DIR}/bevformer_aidlite_qnn240_e2e_performance_v1.py" \
  --backbone-model "${BACKBONE_MODEL}" \
  --encoder-model "${ENCODER_MODEL}" \
  --decoder-model "${DECODER_MODEL}" \
  --asset-manifest "${ASSET_MANIFEST}" \
  --nms-contract "${PACKAGE_DIR}/nms_runtime_contract.json" \
  --nms-reference "${PACKAGE_DIR}/frame009_numpy_native_reference.npz" \
  --output-directory "${OUTPUT_DIRECTORY}" \
  --result-json "${RESULT_JSON}"
