#!/usr/bin/env bash
# Container B host-side asset check.
# Verifies: Python files present, syntax clean, SSH reachable.
set -Eeuo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SELF_DIR/.." && pwd)"
source "$PROJECT_ROOT/tools/board.env"

fail=0
check(){ [[ -s "$2" ]] && echo "$1_GATE=PASS PATH=$2" || { echo "$1_GATE=FAIL PATH=$2"; fail=1; }; }

echo "AUDIT_TYPE=BEVFORMER_HOST_PREFLIGHT"

check RUNNER         "$PROJECT_ROOT/python/bevformer_aidlite_qnn240_e2e_performance_v1.py"
check LAUNCHER       "$PROJECT_ROOT/python/run_bevformer_aidlite_qnn240_e2e_performance_v1.sh"
check NMS            "$PROJECT_ROOT/python/portable_numpy_nmsfreecoder.py"
check MOTHER         "$PROJECT_ROOT/python/functional_mother.py"
check VERIFY         "$PROJECT_ROOT/python/verify_contract.py"
check REFERENCE      "$PROJECT_ROOT/python/frame009_numpy_native_reference.npz"

bash -n "$PROJECT_ROOT/python/run_bevformer_aidlite_qnn240_e2e_performance_v1.sh" \
  && echo "LAUNCHER_SYNTAX_GATE=PASS" \
  || { echo "LAUNCHER_SYNTAX_GATE=FAIL"; fail=1; }

python3 -m py_compile \
  "$PROJECT_ROOT/python/bevformer_aidlite_qnn240_e2e_performance_v1.py" \
  "$PROJECT_ROOT/python/portable_numpy_nmsfreecoder.py" \
  "$PROJECT_ROOT/python/functional_mother.py" \
  "$PROJECT_ROOT/python/verify_contract.py" \
  && echo "PYTHON_COMPILE_GATE=PASS" \
  || { echo "PYTHON_COMPILE_GATE=FAIL"; fail=1; }

ssh -o BatchMode=yes -o ConnectTimeout=10 "$BOARD_HOST" true \
  && echo "BOARD_SSH_GATE=PASS" \
  || { echo "BOARD_SSH_GATE=FAIL"; fail=1; }

[[ $fail -eq 0 ]] && echo "HOST_PREFLIGHT_GATE=PASS" || { echo "HOST_PREFLIGHT_GATE=FAIL"; exit 1; }
