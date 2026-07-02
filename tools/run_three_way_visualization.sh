#!/usr/bin/env bash
set -Eeuo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SELF_DIR/.." && pwd)"
ENV_FILE="${THREE_WAY_VISUALIZATION_ENV_FILE:-$SELF_DIR/three_way_visualization.env}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "THREE_WAY_VISUALIZATION_ENV_GATE=FAIL PATH=$ENV_FILE"
    exit 1
fi

source "$ENV_FILE"

for v in \
    HOST_A_CONDA_SH \
    HOST_A_CONDA_ENV \
    HOST_A_PYTHON \
    BEVFORMER_SOURCE_ROOT \
    LOCAL_PYTORCH_VISUALIZATION_DIR \
    QNN240_VISUALIZATION_DIR \
    THREE_WAY_VISUALIZATION_OUTPUT_ROOT
do
    if [[ -z "${!v:-}" ]]; then
        echo "THREE_WAY_VISUALIZATION_VARIABLE_GATE=FAIL NAME=$v"
        exit 1
    fi
done

source "$HOST_A_CONDA_SH"
conda activate "$HOST_A_CONDA_ENV"

if [[ "${CONDA_DEFAULT_ENV:-}" != "$HOST_A_CONDA_ENV" ]]; then
    echo "HOST_A_CONDA_GATE=FAIL ACTUAL=${CONDA_DEFAULT_ENV:-UNSET}"
    exit 1
fi

test -x "$HOST_A_PYTHON" || {
    echo "HOST_A_PYTHON_GATE=FAIL PATH=$HOST_A_PYTHON"
    exit 1
}

STAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="$THREE_WAY_VISUALIZATION_OUTPUT_ROOT/run_$STAMP"
mkdir -p "$THREE_WAY_VISUALIZATION_OUTPUT_ROOT"

"$HOST_A_PYTHON" \
    "$PROJECT_ROOT/visualization/render_three_way_comparison_v2.py" \
    --source-root "$BEVFORMER_SOURCE_ROOT" \
    --local-visualization-dir "$LOCAL_PYTORCH_VISUALIZATION_DIR" \
    --qnn-visualization-dir "$QNN240_VISUALIZATION_DIR" \
    --output-dir "$OUTPUT_DIR"

HTML="$OUTPUT_DIR/visualization/index.html"
if [[ ! -f "$HTML" ]]; then
    echo "THREE_WAY_HTML_GATE=FAIL PATH=$HTML"
    exit 1
fi

echo "THREE_WAY_HTML_GATE=PASS PATH=$HTML"
echo "THREE_WAY_OUTPUT_DIR=$OUTPUT_DIR"
echo "VIEW_COMMAND=bash tools/serve_latest_three_way_visualization.sh"
