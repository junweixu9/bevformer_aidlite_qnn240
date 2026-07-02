#!/usr/bin/env bash
set -Eeuo pipefail

SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${THREE_WAY_VISUALIZATION_ENV_FILE:-$SELF_DIR/three_way_visualization.env}"
PORT="${THREE_WAY_VISUALIZATION_PORT:-8000}"

if [[ ! -f "$ENV_FILE" ]]; then
    echo "THREE_WAY_VISUALIZATION_ENV_GATE=FAIL PATH=$ENV_FILE"
    exit 1
fi

source "$ENV_FILE"

for v in HOST_A_PYTHON THREE_WAY_VISUALIZATION_OUTPUT_ROOT; do
    if [[ -z "${!v:-}" ]]; then
        echo "THREE_WAY_VISUALIZATION_VARIABLE_GATE=FAIL NAME=$v"
        exit 1
    fi
done

LATEST="$(
    find "$THREE_WAY_VISUALIZATION_OUTPUT_ROOT" \
        -mindepth 2 \
        -maxdepth 3 \
        -type f \
        -path '*/run_*/visualization/index.html' \
        -printf '%h\n' \
    | sort \
    | tail -1
)"

if [[ -z "$LATEST" || ! -f "$LATEST/index.html" ]]; then
    echo "LATEST_THREE_WAY_VISUALIZATION_GATE=FAIL ROOT=$THREE_WAY_VISUALIZATION_OUTPUT_ROOT"
    exit 1
fi

echo "LATEST_THREE_WAY_VISUALIZATION_GATE=PASS PATH=$LATEST/index.html"
echo "URL=http://127.0.0.1:$PORT/index.html"

exec "$HOST_A_PYTHON" -m http.server "$PORT" \
    --bind 0.0.0.0 \
    --directory "$LATEST"
