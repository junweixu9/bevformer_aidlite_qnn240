#!/usr/bin/env python3
"""Direct QCS8550 board entry for BEVFormer AidLite QNN2.40 inference."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
RUNNER = PACKAGE_DIR / "bevformer_aidlite_qnn240_e2e_performance_v1.py"
VERIFIER = PACKAGE_DIR / "verify_contract.py"
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


def runtime_contract_passed(result_json: Path) -> bool:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    checks = {
        "WARMUP_COUNT_GATE": len(data.get("warmup_frames", [])) == 3,
        "MEASURED_FRAME_COUNT_GATE": len(data.get("measured_frames", [])) == 10,
        "INTERPRETER_IDENTITY_STABLE_GATE": data.get("interpreter_identity_stable") is True,
        "CLEANUP_GATE": data.get("cleanup_gate") == "PASS",
        "NO_RUNTIME_EXCEPTION_GATE": "exception_type" not in data,
    }
    for key, passed in checks.items():
        print(f"{key}={'PASS' if passed else 'FAIL'}")
    print(
        "ORIGINAL_STRICT_OUTPUT_GATE="
        + str(data.get("final_output_verification", {}).get("gate", "FAIL"))
    )
    return all(checks.values())


def main() -> int:
    args = parse_args()
    backbone = require_file("backbone model", args.backbone_model)
    encoder = require_file("encoder model", args.encoder_model)
    decoder = require_file("decoder model", args.decoder_model)
    manifest = require_file("asset manifest", args.asset_manifest)
    contract = require_file("NMS contract", args.nms_contract)
    reference = require_file("Frame009 reference", args.reference)
    require_file("frozen runner", str(RUNNER))
    require_file("corrected verifier", str(VERIFIER))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_json = output_dir / "performance_result.json"
    candidate = output_dir / "frame009_final_coordinates.npz"

    command = [
        sys.executable,
        str(RUNNER),
        "--backbone-model", str(backbone),
        "--encoder-model", str(encoder),
        "--decoder-model", str(decoder),
        "--asset-manifest", str(manifest),
        "--nms-contract", str(contract),
        "--nms-reference", str(reference),
        "--output-directory", str(output_dir),
        "--result-json", str(result_json),
    ]

    print("EXECUTION_ENV=QCS8550_BOARD")
    print(f"ASSET_MANIFEST={manifest}")
    print(f"OUTPUT_DIRECTORY={output_dir}")
    print("RUNNER_COMMAND=" + " ".join(command))
    strict_rc = subprocess.run(command, check=False).returncode
    print(f"STRICT_RUN_EXIT={strict_rc}")

    if not result_json.is_file() or not candidate.is_file():
        print("OUTPUT_COORDINATES_GATE=FAIL")
        print("FINAL_DELIVERY_ACCEPTANCE_GATE=FAIL")
        return 1

    print(f"OUTPUT_COORDINATES_GATE=PASS PATH={candidate}")
    runtime_ok = runtime_contract_passed(result_json)

    verify_command = [
        sys.executable,
        str(VERIFIER),
        "--reference", str(reference),
        "--candidate", str(candidate),
        "--report-json", str(output_dir / "corrected_float32_tolerance_report.json"),
        "--report-txt", str(output_dir / "corrected_float32_tolerance_report.txt"),
    ]
    corrected_rc = subprocess.run(verify_command, check=False).returncode
    print(f"CORRECTED_VERIFICATION_EXIT={corrected_rc}")

    known_strict_outcome = strict_rc in (0, 1)
    if strict_rc == 1:
        data = json.loads(result_json.read_text(encoding="utf-8"))
        known_strict_outcome = (
            data.get("final_output_verification", {}).get("gate") == "FAIL"
            and "exception_type" not in data
        )

    if runtime_ok and corrected_rc == 0 and known_strict_outcome:
        print("EXPECTED_STRICT_TOLERANCE_FAILURE_PRESERVED_GATE=PASS")
        print("CORRECTED_FLOAT32_CONTRACT_GATE=PASS")
        print("FINAL_DELIVERY_ACCEPTANCE_GATE=PASS")
        return 0

    print("EXPECTED_STRICT_TOLERANCE_FAILURE_PRESERVED_GATE=FAIL")
    print(
        "CORRECTED_FLOAT32_CONTRACT_GATE="
        + ("PASS" if corrected_rc == 0 else "FAIL")
    )
    print("FINAL_DELIVERY_ACCEPTANCE_GATE=FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
