#!/usr/bin/env bash
# Deploy application code to QCS8550 board without executing inference.
# This is the same deployment step used by run_board.sh, exposed as a
# standalone tool for inspection and debugging.
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

REMOTE_PY="$BOARD_PROJECT_ROOT/python"
REMOTE_CONFIG="$BOARD_PROJECT_ROOT/configs"

ssh -o BatchMode=yes "$BOARD_HOST" "mkdir -p '$REMOTE_PY' '$REMOTE_CONFIG'"

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
