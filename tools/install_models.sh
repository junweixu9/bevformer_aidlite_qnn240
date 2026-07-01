#!/usr/bin/env bash
# Install validated QNN2.40 Context files into the standard model directory.
# Usage: bash tools/install_models.sh <backbone_src> <encoder_src> <decoder_src> [dst_dir]
set -Eeuo pipefail
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/copy_models.sh" "$@"
