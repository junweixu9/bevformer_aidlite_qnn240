#!/usr/bin/env bash
# Independent Phase 5.3 ten-frame export entry.
# This script preserves tools/run_board.sh and the frozen performance runner.
set -Eeuo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SELF_DIR/.." && pwd)"
ENV_FILE="$PROJECT_ROOT/tools/board.env"

fail() {
  local reason="$1"
  echo "TENFRAME_BOARD_ENTRY_FAILURE_REASON=$reason"
  echo "FINAL_TENFRAME_BOARD_EXECUTION_GATE=FAIL"
  exit 1
}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "BOARD_ENV_GATE=FAIL PATH=$ENV_FILE"
  exit 1
fi

# shellcheck disable=SC1090
source "$ENV_FILE"

bash "$SELF_DIR/preflight_host.sh"
bash "$SELF_DIR/preflight_board.sh"

STAMP="$(date +%Y%m%d_%H%M%S)"
REMOTE_PY="$BOARD_PROJECT_ROOT/python"
REMOTE_CONFIG="$BOARD_PROJECT_ROOT/configs"
REMOTE_OUTPUT="$BOARD_PROJECT_ROOT/outputs/tenframe_run_$STAMP"
LOCAL_OUTPUT="$PROJECT_ROOT/outputs/tenframe_run_$STAMP"

mkdir -p "$LOCAL_OUTPUT"

LOCAL_FILES=(
  "$PROJECT_ROOT/python/bevformer_aidlite_qnn240_e2e_tenframe_export_v1.py"
  "$PROJECT_ROOT/python/verify_tenframe_coordinates.py"
  "$PROJECT_ROOT/python/bevformer_aidlite_qnn240_e2e_performance_v1.py"
  "$PROJECT_ROOT/python/functional_mother.py"
  "$PROJECT_ROOT/python/portable_numpy_nmsfreecoder.py"
  "$PROJECT_ROOT/python/acceptance.py"
  "$PROJECT_ROOT/python/verify_contract.py"
  "$PROJECT_ROOT/python/frame009_numpy_native_reference.npz"
  "$PROJECT_ROOT/configs/nms_runtime_contract.json"
)

for file in "${LOCAL_FILES[@]}"; do
  [[ -s "$file" ]] || fail "LOCAL_DEPLOYMENT_ASSET_MISSING:$file"
done

{
  echo "AUDIT_TYPE=BEVFORMER_PHASE5_TENFRAME_BOARD_EXECUTION"
  echo "TIMESTAMP=$STAMP"
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "BOARD_HOST=$BOARD_HOST"
  echo "BOARD_PROJECT_ROOT=$BOARD_PROJECT_ROOT"
  echo "REMOTE_OUTPUT=$REMOTE_OUTPUT"
  echo "LOCAL_OUTPUT=$LOCAL_OUTPUT"
  echo "GIT_HEAD=$(git -C "$PROJECT_ROOT" rev-parse HEAD 2>/dev/null || echo UNAVAILABLE)"
  echo "GIT_BRANCH=$(git -C "$PROJECT_ROOT" branch --show-current 2>/dev/null || echo UNAVAILABLE)"
  for file in "${LOCAL_FILES[@]}"; do
    sha256sum "$file"
  done
} | tee "$LOCAL_OUTPUT/deployment_manifest.txt"

ssh -o BatchMode=yes "$BOARD_HOST" \
  "mkdir -p '$REMOTE_PY' '$REMOTE_CONFIG' '$REMOTE_OUTPUT'"

scp -q -o BatchMode=yes \
  "$PROJECT_ROOT/python/bevformer_aidlite_qnn240_e2e_tenframe_export_v1.py" \
  "$PROJECT_ROOT/python/verify_tenframe_coordinates.py" \
  "$PROJECT_ROOT/python/bevformer_aidlite_qnn240_e2e_performance_v1.py" \
  "$PROJECT_ROOT/python/functional_mother.py" \
  "$PROJECT_ROOT/python/portable_numpy_nmsfreecoder.py" \
  "$PROJECT_ROOT/python/acceptance.py" \
  "$PROJECT_ROOT/python/frame009_numpy_native_reference.npz" \
  "$BOARD_HOST:$REMOTE_PY/"

scp -q -o BatchMode=yes \
  "$PROJECT_ROOT/configs/nms_runtime_contract.json" \
  "$BOARD_HOST:$REMOTE_CONFIG/"

echo "TENFRAME_BOARD_DEPLOY_GATE=PASS REMOTE_ROOT=$BOARD_PROJECT_ROOT"

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

REFERENCE="$PY_DIR/frame009_numpy_native_reference.npz"
RESULT_JSON="$OUTPUT_DIRECTORY/performance_result.json"
FRAME009="$OUTPUT_DIRECTORY/frame009_final_coordinates.npz"
TENFRAME="$OUTPUT_DIRECTORY/tenframe_final_coordinates.npz"
TENFRAME_REPORT_JSON="$OUTPUT_DIRECTORY/tenframe_coordinate_export_report.json"
TENFRAME_REPORT_TXT="$OUTPUT_DIRECTORY/tenframe_coordinate_export_report.txt"

echo "============================================================"
echo "AUDIT_TYPE=BEVFORMER_PHASE5_TENFRAME_REMOTE_EXECUTION"
echo "REMOTE_HOSTNAME=$(hostname)"
echo "REMOTE_USER=$(whoami)"
echo "PY_DIR=$PY_DIR"
echo "OUTPUT_DIRECTORY=$OUTPUT_DIRECTORY"
echo "============================================================"

for file in \
  "$PY_DIR/bevformer_aidlite_qnn240_e2e_tenframe_export_v1.py" \
  "$PY_DIR/verify_tenframe_coordinates.py" \
  "$PY_DIR/bevformer_aidlite_qnn240_e2e_performance_v1.py" \
  "$PY_DIR/functional_mother.py" \
  "$PY_DIR/portable_numpy_nmsfreecoder.py" \
  "$PY_DIR/acceptance.py" \
  "$REFERENCE" \
  "$NMS_CONTRACT" \
  "$BACKBONE_MODEL" \
  "$ENCODER_MODEL" \
  "$DECODER_MODEL" \
  "$ASSET_MANIFEST"
do
  if [[ ! -s "$file" ]]; then
    echo "REMOTE_REQUIRED_ASSET_GATE=FAIL FILE=$file"
    exit 20
  fi
  echo "REMOTE_REQUIRED_ASSET_GATE=PASS FILE=$file"
done

/usr/bin/python3 -m py_compile \
  "$PY_DIR/bevformer_aidlite_qnn240_e2e_tenframe_export_v1.py" \
  "$PY_DIR/verify_tenframe_coordinates.py" \
  "$PY_DIR/bevformer_aidlite_qnn240_e2e_performance_v1.py" \
  "$PY_DIR/functional_mother.py" \
  "$PY_DIR/portable_numpy_nmsfreecoder.py" \
  "$PY_DIR/acceptance.py"

echo "REMOTE_PYTHON_COMPILE_GATE=PASS"

set +e
/usr/bin/python3 \
  "$PY_DIR/bevformer_aidlite_qnn240_e2e_tenframe_export_v1.py" \
  --backbone-model "$BACKBONE_MODEL" \
  --encoder-model "$ENCODER_MODEL" \
  --decoder-model "$DECODER_MODEL" \
  --asset-manifest "$ASSET_MANIFEST" \
  --nms-contract "$NMS_CONTRACT" \
  --nms-reference "$REFERENCE" \
  --output-directory "$OUTPUT_DIRECTORY" \
  --result-json "$RESULT_JSON"
sidecar_rc=$?
set -e

echo "TENFRAME_SIDECAR_EXIT=$sidecar_rc"

set +e
if [[ -s "$TENFRAME" && -s "$FRAME009" ]]; then
  /usr/bin/python3 "$PY_DIR/verify_tenframe_coordinates.py" \
    --tenframe "$TENFRAME" \
    --frame009 "$FRAME009" \
    --report-json "$OUTPUT_DIRECTORY/independent_tenframe_contract.json" \
    --report-txt "$OUTPUT_DIRECTORY/independent_tenframe_contract.txt"
  independent_tenframe_rc=$?
else
  independent_tenframe_rc=1
fi
set -e

echo "INDEPENDENT_TENFRAME_VERIFICATION_EXIT=$independent_tenframe_rc"

set +e
/usr/bin/python3 - \
  "$PY_DIR" \
  "$OUTPUT_DIRECTORY" \
  "$REFERENCE" \
  "$TENFRAME_REPORT_JSON" <<'PY_ACCEPT'
from pathlib import Path
import json
import sys

py_dir = Path(sys.argv[1])
output_directory = Path(sys.argv[2])
reference = Path(sys.argv[3])
tenframe_report_path = Path(sys.argv[4])

sys.path.insert(0, str(py_dir))
from acceptance import final_acceptance

if not tenframe_report_path.is_file():
    print("TENFRAME_REPORT_GATE=FAIL")
    raise SystemExit(1)

report = json.loads(
    tenframe_report_path.read_text(encoding="utf-8")
)
strict_exit = int(report.get("base_runner_exit", 255))

print("TENFRAME_REPORT_GATE=PASS")
print("RECOVERED_BASE_RUNNER_EXIT={}".format(strict_exit))

raise SystemExit(
    final_acceptance(
        strict_exit=strict_exit,
        result_path=output_directory / "performance_result.json",
        candidate_path=output_directory / "frame009_final_coordinates.npz",
        reference_path=reference,
        output_directory=output_directory,
    )
)
PY_ACCEPT
acceptance_rc=$?
set -e

echo "INDEPENDENT_DELIVERY_ACCEPTANCE_EXIT=$acceptance_rc"

required_outputs=(
  "$RESULT_JSON"
  "$OUTPUT_DIRECTORY/performance_report.txt"
  "$FRAME009"
  "$TENFRAME"
  "$TENFRAME_REPORT_JSON"
  "$TENFRAME_REPORT_TXT"
  "$OUTPUT_DIRECTORY/independent_tenframe_contract.json"
  "$OUTPUT_DIRECTORY/independent_tenframe_contract.txt"
  "$OUTPUT_DIRECTORY/corrected_float32_tolerance_report.json"
  "$OUTPUT_DIRECTORY/corrected_float32_tolerance_report.txt"
)

output_gate=PASS
for file in "${required_outputs[@]}"; do
  if [[ -s "$file" ]]; then
    echo "REMOTE_OUTPUT_ASSET_GATE=PASS FILE=$file"
  else
    echo "REMOTE_OUTPUT_ASSET_GATE=FAIL FILE=$file"
    output_gate=FAIL
  fi
done

if [[ "$sidecar_rc" -eq 0 \
   && "$independent_tenframe_rc" -eq 0 \
   && "$acceptance_rc" -eq 0 \
   && "$output_gate" == PASS ]]; then
  echo "TENFRAME_COORDINATE_EXPORT_GATE=PASS"
  echo "TENFRAME_COORDINATE_CONTRACT_GATE=PASS"
  echo "FINAL_DELIVERY_ACCEPTANCE_GATE=PASS"
  echo "FINAL_TENFRAME_BOARD_EXECUTION_GATE=PASS"
  exit 0
fi

echo "FINAL_TENFRAME_BOARD_EXECUTION_GATE=FAIL"
exit 1
REMOTE_RUN
remote_rc=${PIPESTATUS[0]}
set -e

echo "$remote_rc" > "$LOCAL_OUTPUT/remote_execution.exit_code"

set +e
ssh -o BatchMode=yes "$BOARD_HOST" \
  "tar -C '$REMOTE_OUTPUT' -czf - ." \
  | tar -C "$LOCAL_OUTPUT" -xzf -
pull_rc=$?
set -e

echo "REMOTE_EXECUTION_EXIT=$remote_rc"
echo "RESULT_PULL_EXIT=$pull_rc"
echo "LOCAL_OUTPUT=$LOCAL_OUTPUT"

set +e
python3 "$PROJECT_ROOT/python/verify_contract.py" \
  --reference "$PROJECT_ROOT/python/frame009_numpy_native_reference.npz" \
  --candidate "$LOCAL_OUTPUT/frame009_final_coordinates.npz" \
  --report-json "$LOCAL_OUTPUT/host_frame009_contract.json" \
  --report-txt "$LOCAL_OUTPUT/host_frame009_contract.txt"
host_frame009_rc=$?

python3 "$PROJECT_ROOT/python/verify_tenframe_coordinates.py" \
  --tenframe "$LOCAL_OUTPUT/tenframe_final_coordinates.npz" \
  --frame009 "$LOCAL_OUTPUT/frame009_final_coordinates.npz" \
  --report-json "$LOCAL_OUTPUT/host_tenframe_contract.json" \
  --report-txt "$LOCAL_OUTPUT/host_tenframe_contract.txt"
host_tenframe_rc=$?

python3 - \
  "$LOCAL_OUTPUT/performance_result.json" \
  "$LOCAL_OUTPUT/tenframe_coordinate_export_report.json" \
  "$LOCAL_OUTPUT/host_runtime_contract.json" \
  "$LOCAL_OUTPUT/host_runtime_contract.txt" <<'PY_HOST'
from pathlib import Path
import json
import sys

performance_path = Path(sys.argv[1])
tenframe_report_path = Path(sys.argv[2])
json_out = Path(sys.argv[3])
text_out = Path(sys.argv[4])

report = {"gate": "FAIL"}

try:
    performance = json.loads(
        performance_path.read_text(encoding="utf-8")
    )
    tenframe = json.loads(
        tenframe_report_path.read_text(encoding="utf-8")
    )

    checks = {
        "warmup_count_gate":
            len(performance.get("warmup_frames", [])) == 3,
        "measured_frame_count_gate":
            len(performance.get("measured_frames", [])) == 10,
        "interpreter_identity_stable_gate":
            performance.get("interpreter_identity_stable") is True,
        "cleanup_gate":
            performance.get("cleanup_gate") == "PASS",
        "no_runtime_exception_gate":
            "exception_type" not in performance,
        "measured_loop_file_write_gate":
            performance.get("measured_loop_file_write") is False,
        "execute_frame_call_count_gate":
            tenframe.get("execute_frame_call_count") == 13,
        "sidecar_memory_collection_gate":
            tenframe.get("sidecar_memory_collection_enabled") is True,
        "base_runner_runtime_gate":
            tenframe.get("base_runner_runtime_gate") == "PASS",
        "base_runner_known_strict_outcome_gate":
            tenframe.get("base_runner_known_strict_outcome_gate")
            == "PASS",
        "base_runner_outcome_gate":
            tenframe.get("base_runner_outcome_gate") == "PASS",
        "tenframe_report_gate":
            tenframe.get("gate") == "PASS",
    }

    report.update({
        key: "PASS" if passed else "FAIL"
        for key, passed in checks.items()
    })
    report["gate"] = (
        "PASS" if all(checks.values()) else "FAIL"
    )
except Exception as exc:
    report["exception_type"] = type(exc).__name__
    report["exception_message"] = str(exc)

json_out.write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
lines = [
    "{}={}".format(key.upper(), value)
    for key, value in report.items()
]
text_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
raise SystemExit(0 if report["gate"] == "PASS" else 1)
PY_HOST
host_runtime_rc=$?
set -e

echo "HOST_FRAME009_VERIFICATION_EXIT=$host_frame009_rc"
echo "HOST_TENFRAME_VERIFICATION_EXIT=$host_tenframe_rc"
echo "HOST_RUNTIME_CONTRACT_EXIT=$host_runtime_rc"

required_local=(
  "$LOCAL_OUTPUT/performance_result.json"
  "$LOCAL_OUTPUT/performance_report.txt"
  "$LOCAL_OUTPUT/frame009_final_coordinates.npz"
  "$LOCAL_OUTPUT/tenframe_final_coordinates.npz"
  "$LOCAL_OUTPUT/tenframe_coordinate_export_report.json"
  "$LOCAL_OUTPUT/tenframe_coordinate_export_report.txt"
)

local_asset_gate=PASS
for file in "${required_local[@]}"; do
  if [[ -s "$file" ]]; then
    echo "LOCAL_OUTPUT_ASSET_GATE=PASS FILE=$file"
  else
    echo "LOCAL_OUTPUT_ASSET_GATE=FAIL FILE=$file"
    local_asset_gate=FAIL
  fi
done

if [[ "$remote_rc" -eq 0 \
   && "$pull_rc" -eq 0 \
   && "$host_frame009_rc" -eq 0 \
   && "$host_tenframe_rc" -eq 0 \
   && "$host_runtime_rc" -eq 0 \
   && "$local_asset_gate" == PASS ]] \
   && grep -Fq 'FINAL_TENFRAME_BOARD_EXECUTION_GATE=PASS' \
      "$LOCAL_OUTPUT/remote_execution.log" \
   && grep -Fq 'FINAL_DELIVERY_ACCEPTANCE_GATE=PASS' \
      "$LOCAL_OUTPUT/remote_execution.log"; then
  echo "HOST_FRAME009_CONTRACT_GATE=PASS"
  echo "HOST_TENFRAME_COORDINATE_CONTRACT_GATE=PASS"
  echo "HOST_RUNTIME_CONTRACT_GATE=PASS"
  echo "FINAL_DELIVERY_ACCEPTANCE_GATE=PASS"
  echo "FINAL_TENFRAME_BOARD_EXECUTION_GATE=PASS"
  exit 0
fi

echo "FINAL_TENFRAME_BOARD_EXECUTION_GATE=FAIL"
exit 1
