#!/usr/bin/env python3
"""Direct QCS8550 board entry for BEVFormer AidLite QNN2.40 inference."""
from __future__ import annotations

import argparse
from pathlib import Path

from acceptance import final_acceptance
from bevformer import BevFormerConfig, BevFormerPipeline

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
RUNNER = PACKAGE_DIR / "bevformer_aidlite_qnn240_e2e_performance_v1.py"
DEFAULT_CONTRACT = PROJECT_ROOT / "configs" / "nms_runtime_contract.json"
DEFAULT_REFERENCE = PACKAGE_DIR / "frame009_numpy_native_reference.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BEVFormer AidLite QNN2.40: six-camera tensors to 3D coordinates"
    )
    parser.add_argument("--backbone-model", required=True)
    parser.add_argument("--encoder-model", required=True)
    parser.add_argument("--decoder-model", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--nms-contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def require_file(label: str, value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"{label} is missing or empty: {path}")
    return path


def main() -> int:
    args = parse_args()
    config = BevFormerConfig(
        backbone_model=require_file("backbone model", args.backbone_model),
        encoder_model=require_file("encoder model", args.encoder_model),
        decoder_model=require_file("decoder model", args.decoder_model),
        asset_manifest=require_file("asset manifest", args.asset_manifest),
        nms_contract=require_file("NMS contract", args.nms_contract),
        reference=require_file("Frame009 reference", args.reference),
        output_dir=Path(args.output_dir).expanduser().resolve(),
    )
    runner = require_file("frozen runner", str(RUNNER))
    pipeline = BevFormerPipeline(config=config, runner_path=runner)

    print("EXECUTION_ENV=QCS8550_BOARD")
    print(f"ASSET_MANIFEST={config.asset_manifest}")
    print(f"OUTPUT_DIRECTORY={config.output_dir}")

    strict_exit = pipeline.run()
    print(f"STRICT_RUN_EXIT={strict_exit}")

    return final_acceptance(
        strict_exit=strict_exit,
        result_path=config.output_dir / "performance_result.json",
        candidate_path=config.output_dir / "frame009_final_coordinates.npz",
        reference_path=config.reference,
        output_directory=config.output_dir,
    )


if __name__ == "__main__":
    raise SystemExit(main())
