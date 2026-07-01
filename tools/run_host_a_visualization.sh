#!/usr/bin/env bash
set -Eeuo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SELF_DIR/.." && pwd)"
ENV_FILE="${HOST_VISUALIZATION_ENV_FILE:-$SELF_DIR/host_visualization.env}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "HOST_VISUALIZATION_ENV_GATE=FAIL PATH=$ENV_FILE"
    exit 1
fi

source "$ENV_FILE"

for v in HOST_A_CONDA_SH HOST_A_CONDA_ENV HOST_A_PYTHON BEVFORMER_SOURCE_ROOT BEVFORMER_RESULT_PKL HOST_A_VISUALIZATION_OUTPUT_ROOT; do
    if [[ -z "${!v:-}" ]]; then
        echo "HOST_VISUALIZATION_VARIABLE_GATE=FAIL NAME=$v"
        exit 1
    fi
done

source "$HOST_A_CONDA_SH"
conda activate "$HOST_A_CONDA_ENV"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$HOST_A_CONDA_ENV" ]]; then
    echo "HOST_A_CONDA_GATE=FAIL"
    exit 1
fi

STAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="$HOST_A_VISUALIZATION_OUTPUT_ROOT/run_$STAMP"
mkdir -p "$HOST_A_VISUALIZATION_OUTPUT_ROOT"

exec "$HOST_A_PYTHON" \
    "$PROJECT_ROOT/visualization/render_host_a.py" \
    --source-root "$BEVFORMER_SOURCE_ROOT" \
    --result-pkl "$BEVFORMER_RESULT_PKL" \
    --output-dir "$OUTPUT_DIR"
