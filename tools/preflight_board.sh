#!/usr/bin/env bash
# QCS8550 board-side environment check.
# Verifies: AidLite, TYPE_QNN240, NumPy, SciPy, three Context SHA256.
set -Eeuo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SELF_DIR/.." && pwd)"
source "$PROJECT_ROOT/tools/board.env"

ssh -o BatchMode=yes -o ConnectTimeout=10 "$BOARD_HOST" /bin/bash -s \
  -- "$BACKBONE_MODEL_REMOTE" "$BACKBONE_SHA" \
     "$ENCODER_MODEL_REMOTE"  "$ENCODER_SHA" \
     "$DECODER_MODEL_REMOTE"  "$DECODER_SHA" <<'REMOTE'
set -Eeuo pipefail
fail=0

check_sha(){
  local k=$1 p=$2 e=$3
  [[ -s "$p" ]] || { echo "${k}_ASSET_GATE=FAIL PATH=$p"; fail=1; return; }
  a=$(/usr/bin/sha256sum "$p" | /usr/bin/awk '{print $1}')
  [[ "$a" == "$e" ]] \
    && echo "${k}_SHA_GATE=PASS SHA256=$a" \
    || { echo "${k}_SHA_GATE=FAIL EXPECTED=$e ACTUAL=$a"; fail=1; }
}

/usr/bin/python3 - <<'PY' || fail=1
import aidlite, numpy, scipy
assert int(aidlite.FrameworkType.TYPE_QNN240) == 109
assert int(aidlite.ImplementType.TYPE_LOCAL) == 3
assert int(aidlite.AccelerateType.TYPE_DSP) == 3
print('AIDLITE_IMPORT_GATE=PASS')
print('TYPE_QNN240_GATE=PASS VALUE=' + str(aidlite.FrameworkType.TYPE_QNN240))
print('NUMPY_VERSION=' + numpy.__version__)
print('SCIPY_VERSION=' + scipy.__version__)
PY

check_sha BACKBONE "$1" "$2"
check_sha ENCODER  "$3" "$4"
check_sha DECODER  "$5" "$6"

[[ $fail -eq 0 ]] && echo "BOARD_PREFLIGHT_GATE=PASS" || { echo "BOARD_PREFLIGHT_GATE=FAIL"; exit 1; }
REMOTE
