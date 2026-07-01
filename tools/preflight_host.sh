#!/usr/bin/env bash
# Container-B host-side checks. This script does not require AidLite locally.
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

fail=0
check(){
  local key="$1" path="$2"
  if [[ -s "$path" ]]; then
    echo "${key}_GATE=PASS PATH=$path"
  else
    echo "${key}_GATE=FAIL PATH=$path"
    fail=1
  fi
}

echo "AUDIT_TYPE=BEVFORMER_HOST_PREFLIGHT"
echo "PROJECT_ROOT=$PROJECT_ROOT"
check RUN_TEST       "$PROJECT_ROOT/python/run_test.py"
check PIPELINE       "$PROJECT_ROOT/python/bevformer.py"
check ACCEPTANCE     "$PROJECT_ROOT/python/acceptance.py"
check RUNNER         "$PROJECT_ROOT/python/bevformer_aidlite_qnn240_e2e_performance_v1.py"
check NMS            "$PROJECT_ROOT/python/portable_numpy_nmsfreecoder.py"
check MOTHER         "$PROJECT_ROOT/python/functional_mother.py"
check VERIFY         "$PROJECT_ROOT/python/verify_contract.py"
check REFERENCE      "$PROJECT_ROOT/python/frame009_numpy_native_reference.npz"
check NMS_CONTRACT   "$PROJECT_ROOT/configs/nms_runtime_contract.json"

python3 -m py_compile \
  "$PROJECT_ROOT/python/run_test.py" \
  "$PROJECT_ROOT/python/bevformer.py" \
  "$PROJECT_ROOT/python/acceptance.py" \
  "$PROJECT_ROOT/python/bevformer_aidlite_qnn240_e2e_performance_v1.py" \
  "$PROJECT_ROOT/python/portable_numpy_nmsfreecoder.py" \
  "$PROJECT_ROOT/python/functional_mother.py" \
  "$PROJECT_ROOT/python/verify_contract.py" \
  && echo "PYTHON_COMPILE_GATE=PASS" \
  || { echo "PYTHON_COMPILE_GATE=FAIL"; fail=1; }

python3 - "$PROJECT_ROOT/configs/nms_runtime_contract.json" <<'PY' \
  && echo "NMS_CONTRACT_JSON_GATE=PASS" \
  || { echo "NMS_CONTRACT_JSON_GATE=FAIL"; fail=1; }
import json, sys
path = sys.argv[1]
data = json.load(open(path, encoding="utf-8"))
required = {"num_classes", "max_num", "num_query", "code_size", "post_center_range", "score_threshold", "selected_sigmoid_mode"}
missing = required.difference(data)
if missing:
    raise SystemExit("missing keys: " + ",".join(sorted(missing)))
if data["num_classes"] != 10 or data["max_num"] != 300 or data["num_query"] != 900 or data["code_size"] != 10:
    raise SystemExit("unexpected NMS contract dimensions")
if len(data["post_center_range"]) != 6:
    raise SystemExit("post_center_range must contain 6 values")
PY

ssh -o BatchMode=yes -o ConnectTimeout=10 "$BOARD_HOST" true \
  && echo "BOARD_SSH_GATE=PASS" \
  || { echo "BOARD_SSH_GATE=FAIL"; fail=1; }

[[ $fail -eq 0 ]] \
  && echo "HOST_PREFLIGHT_GATE=PASS" \
  || { echo "HOST_PREFLIGHT_GATE=FAIL"; exit 1; }
