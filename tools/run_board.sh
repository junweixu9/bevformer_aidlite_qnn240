#!/usr/bin/env bash
# Container-B remote helper. Actual inference and acceptance run in python/run_test.py.
set -Eeuo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SELF_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/tools/board.env"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "BOARD_ENV_GATE=FAIL PATH=$ENV_FILE"
  echo "Create it with: cp tools/board.env.example tools/board.env"
  exit 1
fi
# shellcheck disable=SC1090
source "$ENV_FILE"

"$SELF_DIR/preflight_host.sh"
"$SELF_DIR/preflight_board.sh"

STAMP=$(date +%Y%m%d_%H%M%S)
REMOTE_PY="$BOARD_PROJECT_ROOT/python"
REMOTE_CONFIG="$BOARD_PROJECT_ROOT/configs"
REMOTE_OUTPUT="$BOARD_PROJECT_ROOT/outputs/run_$STAMP"
LOCAL_OUTPUT="$PROJECT_ROOT/outputs/run_$STAMP"
mkdir -p "$LOCAL_OUTPUT"

ssh -o BatchMode=yes "$BOARD_HOST" \
  "mkdir -p '$REMOTE_PY' '$REMOTE_CONFIG' '$REMOTE_OUTPUT'"

scp -q -o BatchMode=yes \
  "$PROJECT_ROOT/python/run_test.py" \
  "$PROJECT_ROOT/python/bevformer.py" \
  "$PROJECT_ROOT/python/acceptance.py" \
  "$PROJECT_ROOT/python/bevformer_aidlite_qnn240_e2e_performance_v1.py" \
  "$PROJECT_ROOT/python/functional_mother.py" \
  "$PROJECT_ROOT/python/portable_numpy_nmsfreecoder.py" \
  "$PROJECT_ROOT/python/frame009_numpy_native_reference.npz" \
  "$BOARD_HOST:$REMOTE_PY/"

scp -q -o BatchMode=yes \
  "$PROJECT_ROOT/configs/nms_runtime_contract.json" \
  "$BOARD_HOST:$REMOTE_CONFIG/"

echo "BOARD_DEPLOY_GATE=PASS REMOTE_ROOT=$BOARD_PROJECT_ROOT"

set +e
ssh -o BatchMode=yes "$BOARD_HOST" /bin/bash -s -- \
  "$REMOTE_PY" \
  "$REMOTE_CONFIG/nms_runtime_contract.json" \
  "$BACKBONE_MODEL_REMOTE" \
  "$ENCODER_MODEL_REMOTE" \
  "$DECODER_MODEL_REMOTE" \
  "$ASSET_MANIFEST_REMOTE" \
  "$REMOTE_OUTPUT" <<'REMOTE_RUN' 2>&1 | tee "$LOCAL_OUTPUT/remote_execution.log"
set -Eeuo pipefail
PY_DIR="$1"
NMS_CONTRACT="$2"
BACKBONE_MODEL="$3"
ENCODER_MODEL="$4"
DECODER_MODEL="$5"
ASSET_MANIFEST="$6"
OUTPUT_DIRECTORY="$7"

exec /usr/bin/python3 "$PY_DIR/run_test.py" \
  --backbone-model "$BACKBONE_MODEL" \
  --encoder-model "$ENCODER_MODEL" \
  --decoder-model "$DECODER_MODEL" \
  --asset-manifest "$ASSET_MANIFEST" \
  --nms-contract "$NMS_CONTRACT" \
  --reference "$PY_DIR/frame009_numpy_native_reference.npz" \
  --output-dir "$OUTPUT_DIRECTORY"
REMOTE_RUN
remote_rc=${PIPESTATUS[0]}
set -e
echo "$remote_rc" > "$LOCAL_OUTPUT/remote_execution.exit_code"

set +e
ssh -o BatchMode=yes "$BOARD_HOST" "tar -C '$REMOTE_OUTPUT' -czf - ." \
  | tar -C "$LOCAL_OUTPUT" -xzf -
pull_rc=$?
set -e

echo "REMOTE_EXECUTION_EXIT=$remote_rc"
echo "RESULT_PULL_EXIT=$pull_rc"
echo "LOCAL_OUTPUT=$LOCAL_OUTPUT"

CANDIDATE="$LOCAL_OUTPUT/frame009_final_coordinates.npz"
REFERENCE="$PROJECT_ROOT/python/frame009_numpy_native_reference.npz"

set +e
if [[ -s "$CANDIDATE" ]]; then
  python3 "$PROJECT_ROOT/python/verify_contract.py" \
    --reference "$REFERENCE" \
    --candidate "$CANDIDATE" \
    --report-json "$LOCAL_OUTPUT/host_corrected_contract.json" \
    --report-txt "$LOCAL_OUTPUT/host_corrected_contract.txt"
  host_verify_rc=$?
else
  host_verify_rc=1
fi
set -e

echo "HOST_CORRECTED_VERIFICATION_EXIT=$host_verify_rc"

if [[ $remote_rc -eq 0 \
   && $pull_rc -eq 0 \
   && $host_verify_rc -eq 0 \
   && -s "$LOCAL_OUTPUT/performance_result.json" \
   && -s "$CANDIDATE" ]] \
   && grep -Fq 'FINAL_DELIVERY_ACCEPTANCE_GATE=PASS' "$LOCAL_OUTPUT/remote_execution.log"; then
  echo "OUTPUT_COORDINATES_GATE=PASS PATH=$CANDIDATE"
  echo "FINAL_DELIVERY_ACCEPTANCE_GATE=PASS"
  exit 0
fi

echo "OUTPUT_COORDINATES_GATE=FAIL"
echo "FINAL_DELIVERY_ACCEPTANCE_GATE=FAIL"
exit 1
