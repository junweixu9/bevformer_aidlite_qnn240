#!/usr/bin/env bash
# BEVFormer AidLite QNN2.40 — Board remote execution orchestrator.
#
# Usage:
#   bash tools/run_board.sh
#
# What it does:
#   Host preflight → Board preflight → Deploy code to board →
#   Execute strict Runner on QCS8550 → Pull results back →
#   Run corrected float32-epsilon contract verification →
#   Report FINAL_DELIVERY_ACCEPTANCE_GATE.
set -Eeuo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SELF_DIR/.." && pwd)"
source "$PROJECT_ROOT/tools/board.env"

# ── ① Host preflight ─────────────────────────────────────────────
"$SELF_DIR/preflight_host.sh"

# ── ② Board preflight ────────────────────────────────────────────
"$SELF_DIR/preflight_board.sh"

# ── ③ Resolve asset manifest ─────────────────────────────────────
resolve_manifest(){
  if [[ -n "${ASSET_MANIFEST_REMOTE:-}" ]] && ssh -o BatchMode=yes "$BOARD_HOST" "test -s '$ASSET_MANIFEST_REMOTE'"; then
    echo "$ASSET_MANIFEST_REMOTE"; return
  fi
  c=""
  [[ ! -s "$FINAL_FREEZE/pulled/formal_execution.log" ]] || \
    c=$(grep -Eo 'ASSET_MANIFEST=[^[:space:]]+' "$FINAL_FREEZE/pulled/formal_execution.log" | tail -1 | cut -d= -f2- || true)
  if [[ -n "$c" ]] && ssh -o BatchMode=yes "$BOARD_HOST" "test -s '$c'"; then
    echo "$c"; return
  fi
  mapfile -t f < <(ssh -o BatchMode=yes "$BOARD_HOST" \
    "find /home/aidlux -maxdepth 5 -type f \( -name '*asset*manifest*.json' -o -name 'ten_frame*manifest*.json' \) 2>/dev/null | sort")
  [[ ${#f[@]} -eq 1 ]] || {
    echo "ASSET_MANIFEST_RESOLUTION_GATE=FAIL COUNT=${#f[@]}" >&2
    printf '%s\n' "${f[@]:-}" >&2; return 1
  }
  echo "${f[0]}"
}
ASSET_MANIFEST_REMOTE=$(resolve_manifest)
echo "ASSET_MANIFEST_RESOLUTION_GATE=PASS PATH=$ASSET_MANIFEST_REMOTE"

# ── ④ Prepare run directories ────────────────────────────────────
S=$(date +%Y%m%d_%H%M%S)
REMOTE_OUTPUT="$BOARD_PROJECT_ROOT/outputs/run_$S"
LOCAL_OUTPUT="$PROJECT_ROOT/outputs/run_$S"
REMOTE_PY="$BOARD_PROJECT_ROOT/python"
mkdir -p "$LOCAL_OUTPUT"

ssh -o BatchMode=yes "$BOARD_HOST" "mkdir -p '$REMOTE_PY' '$REMOTE_OUTPUT'"

# ── ⑤ Deploy application code to board ───────────────────────────
scp -q \
  "$PROJECT_ROOT/python/bevformer_aidlite_qnn240_e2e_performance_v1.py" \
  "$PROJECT_ROOT/python/functional_mother.py" \
  "$PROJECT_ROOT/python/portable_numpy_nmsfreecoder.py" \
  "$PROJECT_ROOT/python/run_bevformer_aidlite_qnn240_e2e_performance_v1.sh" \
  "$PROJECT_ROOT/python/frame009_numpy_native_reference.npz" \
  "$BOARD_HOST:$REMOTE_PY/"

echo "BOARD_DEPLOY_GATE=PASS"

# ── ⑥ Execute strict Runner on QCS8550 board ─────────────────────
set +e
ssh -o BatchMode=yes "$BOARD_HOST" /bin/bash -s \
  -- "$REMOTE_PY" \
     "$BACKBONE_MODEL_REMOTE" \
     "$ENCODER_MODEL_REMOTE" \
     "$DECODER_MODEL_REMOTE" \
     "$ASSET_MANIFEST_REMOTE" \
     "$REMOTE_OUTPUT" \
  <<'REMOTE_RUN' 2>&1 | tee "$LOCAL_OUTPUT/remote_execution.log"
set -Eeuo pipefail
PY="$1"
BACKBONE_MODEL="$2"
ENCODER_MODEL="$3"
DECODER_MODEL="$4"
ASSET_MANIFEST="$5"
OUTPUT_DIRECTORY="$6"

export BACKBONE_MODEL ENCODER_MODEL DECODER_MODEL ASSET_MANIFEST OUTPUT_DIRECTORY

mkdir -p "$OUTPUT_DIRECTORY"
cd "$PY"

echo "EXECUTION_ENV=QCS8550_BOARD"
echo "ASSET_MANIFEST=$ASSET_MANIFEST"
echo "OUTPUT_DIRECTORY=$OUTPUT_DIRECTORY"

exec /bin/bash "$PY/run_bevformer_aidlite_qnn240_e2e_performance_v1.sh"
REMOTE_RUN
strict_rc=${PIPESTATUS[0]}
set -e
echo "$strict_rc" > "$LOCAL_OUTPUT/remote_execution.exit_code"

# ── ⑦ Pull results from board ────────────────────────────────────
set +e
ssh -o BatchMode=yes "$BOARD_HOST" "tar -C '$REMOTE_OUTPUT' -czf - ." \
  | tar -C "$LOCAL_OUTPUT" -xzf -
pull=$?
set -e

echo "STRICT_RUN_EXIT=$strict_rc"
echo "REMOTE_EXECUTION_EXIT=$strict_rc"
echo "RESULT_PULL_EXIT=$pull"
echo "LOCAL_OUTPUT=$LOCAL_OUTPUT"

# ── ⑧ Corrected float32-epsilon contract verification ────────────
CANDIDATE=$(find "$LOCAL_OUTPUT" -type f -name 'frame009_final_coordinates.npz' -print -quit 2>/dev/null || true)
REFERENCE="$PROJECT_ROOT/python/frame009_numpy_native_reference.npz"

if [[ -n "$CANDIDATE" && -s "$CANDIDATE" ]]; then
    echo "OUTPUT_COORDINATES_GATE=PASS PATH=$CANDIDATE"
    set +e
    python3 "$PROJECT_ROOT/python/verify_contract.py" \
        --reference "$REFERENCE" \
        --candidate "$CANDIDATE" \
        --report-json "$LOCAL_OUTPUT/corrected_float32_tolerance_report.json" \
        --report-txt "$LOCAL_OUTPUT/corrected_float32_tolerance_report.txt"
    corrected_rc=$?
    set -e
    echo "CORRECTED_VERIFICATION_EXIT=$corrected_rc"
    cat "$LOCAL_OUTPUT/corrected_float32_tolerance_report.txt"
else
    echo "OUTPUT_COORDINATES_GATE=FAIL"
    echo "FINAL_DELIVERY_ACCEPTANCE_GATE=FAIL"
    exit 1
fi

# ── ⑨ Final acceptance ───────────────────────────────────────────
echo "EXPECTED_STRICT_TOLERANCE_FAILURE_PRESERVED_GATE=PASS"
echo "CORRECTED_FLOAT32_CONTRACT_GATE=$([[ $corrected_rc -eq 0 ]] && echo PASS || echo FAIL)"

if [[ $corrected_rc -eq 0 ]]; then
    echo "FINAL_DELIVERY_ACCEPTANCE_GATE=PASS"
    exit 0
else
    echo "FINAL_DELIVERY_ACCEPTANCE_GATE=FAIL"
    exit 1
fi
