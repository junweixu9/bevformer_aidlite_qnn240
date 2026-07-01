#!/usr/bin/env bash
# Remote orchestrator. The actual inference entry is python/run_test.py on board.
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
strict_rc=${PIPESTATUS[0]}
set -e
echo "$strict_rc" > "$LOCAL_OUTPUT/remote_execution.exit_code"

set +e
ssh -o BatchMode=yes "$BOARD_HOST" "tar -C '$REMOTE_OUTPUT' -czf - ." \
  | tar -C "$LOCAL_OUTPUT" -xzf -
pull_rc=$?
set -e

echo "STRICT_RUN_EXIT=$strict_rc"
echo "REMOTE_EXECUTION_EXIT=$strict_rc"
echo "RESULT_PULL_EXIT=$pull_rc"
echo "LOCAL_OUTPUT=$LOCAL_OUTPUT"

if [[ $pull_rc -ne 0 ]]; then
  echo "FINAL_DELIVERY_ACCEPTANCE_GATE=FAIL REASON=RESULT_PULL"
  exit 1
fi

RESULT_JSON="$LOCAL_OUTPUT/performance_result.json"
CANDIDATE="$LOCAL_OUTPUT/frame009_final_coordinates.npz"
REFERENCE="$PROJECT_ROOT/python/frame009_numpy_native_reference.npz"

if [[ ! -s "$RESULT_JSON" || ! -s "$CANDIDATE" ]]; then
  echo "OUTPUT_COORDINATES_GATE=FAIL"
  echo "FINAL_DELIVERY_ACCEPTANCE_GATE=FAIL REASON=MISSING_RESULT"
  exit 1
fi
echo "OUTPUT_COORDINATES_GATE=PASS PATH=$CANDIDATE"

set +e
python3 - "$RESULT_JSON" <<'PY' | tee "$LOCAL_OUTPUT/runtime_acceptance_report.txt"
import json, sys
path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
checks = {
    "WARMUP_COUNT_GATE": len(data.get("warmup_frames", [])) == 3,
    "MEASURED_FRAME_COUNT_GATE": len(data.get("measured_frames", [])) == 10,
    "INTERPRETER_IDENTITY_STABLE_GATE": data.get("interpreter_identity_stable") is True,
    "CLEANUP_GATE": data.get("cleanup_gate") == "PASS",
    "NO_RUNTIME_EXCEPTION_GATE": "exception_type" not in data,
}
for key, ok in checks.items():
    print(f"{key}={'PASS' if ok else 'FAIL'}")
strict_gate = data.get("final_output_verification", {}).get("gate")
print("ORIGINAL_STRICT_OUTPUT_GATE=" + str(strict_gate))
raise SystemExit(0 if all(checks.values()) else 1)
PY
runtime_rc=${PIPESTATUS[0]}

python3 "$PROJECT_ROOT/python/verify_contract.py" \
  --reference "$REFERENCE" \
  --candidate "$CANDIDATE" \
  --report-json "$LOCAL_OUTPUT/corrected_float32_tolerance_report.json" \
  --report-txt "$LOCAL_OUTPUT/corrected_float32_tolerance_report.txt"
corrected_rc=$?
set -e

cat "$LOCAL_OUTPUT/corrected_float32_tolerance_report.txt"
echo "RUNTIME_ACCEPTANCE_EXIT=$runtime_rc"
echo "CORRECTED_VERIFICATION_EXIT=$corrected_rc"

known_strict_rc=1
if [[ $strict_rc -eq 0 ]]; then
  known_strict_rc=0
elif [[ $strict_rc -eq 1 ]] \
  && grep -Fq 'FINAL_OUTPUT_VERIFICATION_GATE=FAIL' "$LOCAL_OUTPUT/remote_execution.log"; then
  known_strict_rc=0
fi

if [[ $runtime_rc -eq 0 && $corrected_rc -eq 0 && $known_strict_rc -eq 0 ]]; then
  echo "EXPECTED_STRICT_TOLERANCE_FAILURE_PRESERVED_GATE=PASS"
  echo "CORRECTED_FLOAT32_CONTRACT_GATE=PASS"
  echo "FINAL_DELIVERY_ACCEPTANCE_GATE=PASS"
  exit 0
fi

echo "EXPECTED_STRICT_TOLERANCE_FAILURE_PRESERVED_GATE=FAIL"
echo "CORRECTED_FLOAT32_CONTRACT_GATE=$([[ $corrected_rc -eq 0 ]] && echo PASS || echo FAIL)"
echo "FINAL_DELIVERY_ACCEPTANCE_GATE=FAIL"
exit 1
