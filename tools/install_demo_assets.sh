#!/usr/bin/env bash
# Install validated ten-frame demonstration assets.
# Usage: bash tools/install_demo_assets.sh <source_manifest.json> [dst_dir]
set -Eeuo pipefail
SRC_MANIFEST="${1:?source manifest path required}"
DST="${2:-./assets/unseen10}"
python3 "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/copy_demo_assets.py" \
  --source-manifest "$SRC_MANIFEST" --destination "$DST"
