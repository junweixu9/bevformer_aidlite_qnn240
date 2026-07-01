#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 BACKBONE ENCODER DECODER DESTINATION"
  exit 2
fi

BACKBONE_SOURCE="$1"
ENCODER_SOURCE="$2"
DECODER_SOURCE="$3"
DESTINATION="$4"

BACKBONE_SHA="ec709ae41f0dc6b25e2f8950abc8e15a5c85301ac31eea1fa1750927e364c945"
ENCODER_SHA="2868cb971a6e291ff9d3d874b2dfe2238c4b545ffdb8af1b01b7b56697ce71a1"
DECODER_SHA="dacebf6428168bbe0e29410f05285a7fde454860a607c73f983149e607b78d7c"

check_file(){
  local name="$1" path="$2" expected="$3" actual
  [[ -s "$path" ]] || { echo "${name}_ASSET_GATE=FAIL PATH=$path"; exit 1; }
  actual=$(sha256sum "$path" | awk '{print $1}')
  [[ "$actual" == "$expected" ]] || {
    echo "${name}_SHA_GATE=FAIL EXPECTED=$expected ACTUAL=$actual"
    exit 1
  }
  echo "${name}_SHA_GATE=PASS SHA256=$actual"
}

check_file BACKBONE "$BACKBONE_SOURCE" "$BACKBONE_SHA"
check_file ENCODER "$ENCODER_SOURCE" "$ENCODER_SHA"
check_file DECODER "$DECODER_SOURCE" "$DECODER_SHA"

mkdir -p "$DESTINATION"
cp "$BACKBONE_SOURCE" "$DESTINATION/backbone_context.bin"
cp "$ENCODER_SOURCE" "$DESTINATION/encoder_context.bin"
cp "$DECODER_SOURCE" "$DESTINATION/decoder_context.bin"

check_file BACKBONE "$DESTINATION/backbone_context.bin" "$BACKBONE_SHA"
check_file ENCODER "$DESTINATION/encoder_context.bin" "$ENCODER_SHA"
check_file DECODER "$DESTINATION/decoder_context.bin" "$DECODER_SHA"
echo "MODEL_COPY_GATE=PASS DESTINATION=$DESTINATION"
