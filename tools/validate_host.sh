#!/usr/bin/env bash
# Board-independent repository validation for Container B or CI.
set -Eeuo pipefail
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SELF_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "AUDIT_TYPE=BEVFORMER_REPOSITORY_HOST_VALIDATION"
echo "PROJECT_ROOT=$PROJECT_ROOT"

python3 -m py_compile \
  python/run_test.py \
  python/run_e2e.py \
  python/bevformer.py \
  python/temporal.py \
  python/utils.py \
  python/runtime_api.py \
  python/postprocess_api.py \
  python/bevformer_aidlite_qnn240_e2e_performance_v1.py \
  python/functional_mother.py \
  python/portable_numpy_nmsfreecoder.py \
  python/verify_contract.py \
  tools/copy_demo_assets.py

echo "PYTHON_COMPILE_GATE=PASS"

bash -n \
  tools/validate_host.sh \
  tools/run_remote.sh \
  tools/run_board.sh \
  tools/preflight_host.sh \
  tools/preflight_board.sh \
  tools/copy_models.sh

echo "SHELL_SYNTAX_GATE=PASS"

python3 -m json.tool configs/nms_runtime_contract.json >/dev/null
python3 - <<'PY'
import json
from pathlib import Path

contract = json.loads(Path("configs/nms_runtime_contract.json").read_text(encoding="utf-8"))
required = {
    "num_classes",
    "max_num",
    "num_query",
    "code_size",
    "post_center_range",
    "score_threshold",
    "selected_sigmoid_mode",
}
missing = required.difference(contract)
if missing:
    raise SystemExit("missing NMS contract keys: " + ",".join(sorted(missing)))
assert contract["num_classes"] == 10
assert contract["max_num"] == 300
assert contract["num_query"] == 900
assert contract["code_size"] == 10
assert len(contract["post_center_range"]) == 6
print("NMS_CONTRACT_GATE=PASS")
PY

python3 -m pytest -q
echo "UNIT_TEST_GATE=PASS"
echo "HOST_VALIDATION_GATE=PASS"
