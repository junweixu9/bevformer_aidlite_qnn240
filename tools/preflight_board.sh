#!/usr/bin/env bash
# QCS8550 board-side environment and frozen Context identity checks.
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

ssh -o BatchMode=yes -o ConnectTimeout=10 "$BOARD_HOST" /bin/bash -s \
  -- "$BACKBONE_MODEL_REMOTE" "$BACKBONE_SHA" \
     "$ENCODER_MODEL_REMOTE"  "$ENCODER_SHA" \
     "$DECODER_MODEL_REMOTE"  "$DECODER_SHA" \
     "$ASSET_MANIFEST_REMOTE" <<'REMOTE'
set -Eeuo pipefail
fail=0

check_sha(){
  local key="$1" path="$2" expected="$3" actual
  if [[ ! -s "$path" ]]; then
    echo "${key}_ASSET_GATE=FAIL PATH=$path"
    fail=1
    return
  fi
  actual=$(/usr/bin/sha256sum "$path" | /usr/bin/awk '{print $1}')
  if [[ "$actual" == "$expected" ]]; then
    echo "${key}_SHA_GATE=PASS SHA256=$actual"
  else
    echo "${key}_SHA_GATE=FAIL EXPECTED=$expected ACTUAL=$actual"
    fail=1
  fi
}

/usr/bin/python3 - <<'PY' || fail=1
import aidlite, numpy, scipy
assert int(aidlite.FrameworkType.TYPE_QNN240) == 109
assert int(aidlite.ImplementType.TYPE_LOCAL) == 3
assert int(aidlite.AccelerateType.TYPE_DSP) == 3
print("AIDLITE_IMPORT_GATE=PASS")
print("TYPE_QNN240_GATE=PASS VALUE=" + str(aidlite.FrameworkType.TYPE_QNN240))
print("NUMPY_VERSION=" + numpy.__version__)
print("SCIPY_VERSION=" + scipy.__version__)
PY

check_sha BACKBONE "$1" "$2"
check_sha ENCODER  "$3" "$4"
check_sha DECODER  "$5" "$6"

/usr/bin/python3 - "$7" <<'PY' || fail=1
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if not path.is_file() or path.stat().st_size == 0:
    raise SystemExit("asset manifest missing")
data = json.loads(path.read_text(encoding="utf-8"))
if data.get("status") != "PASS":
    raise SystemExit("asset manifest status is not PASS")
if data.get("frame_indices") != list(range(10)):
    raise SystemExit("asset manifest frame_indices must be 0..9")
print("ASSET_MANIFEST_CONTRACT_GATE=PASS PATH=" + str(path))
PY

[[ $fail -eq 0 ]] \
  && echo "BOARD_PREFLIGHT_GATE=PASS" \
  || { echo "BOARD_PREFLIGHT_GATE=FAIL"; exit 1; }
REMOTE
