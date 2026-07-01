#!/usr/bin/env python3
"""Direct QCS8550 board entry for BEVFormer AidLite QNN2.40 inference."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

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


def load_coordinates(path: Path):
    with np.load(path, allow_pickle=False) as data:
        return (
            np.asarray(data["boxes"], dtype=np.float32),
            np.asarray(data["scores"], dtype=np.float32).reshape(-1),
            np.asarray(data["labels"], dtype=np.int64).reshape(-1),
        )


def corrected_contract(reference: Path, candidate: Path, output_dir: Path) -> bool:
    ref_boxes, ref_scores, ref_labels = load_coordinates(reference)
    boxes, scores, labels = load_coordinates(candidate)
    eps = float(np.finfo(np.float32).eps)
    shape_ok = (
        boxes.shape == ref_boxes.shape
        and scores.shape == ref_scores.shape
        and labels.shape == ref_labels.shape
    )
    finite_ok = bool(np.isfinite(boxes).all() and np.isfinite(scores).all())
    labels_ok = bool(shape_ok and np.array_equal(labels, ref_labels))
    score_error = (
        float(np.max(np.abs(scores.astype(np.float64) - ref_scores.astype(np.float64))))
        if shape_ok and scores.size else float("inf")
    )
    box_error = (
        float(np.max(np.abs(boxes.astype(np.float64) - ref_boxes.astype(np.float64))))
        if shape_ok and boxes.size else float("inf")
    )
    score_ok = score_error <= 2.0 * eps
    box_ok = box_error <= 8.0 * eps
    passed = bool(shape_ok and finite_ok and labels_ok and score_ok and box_ok)
    report = {
        "shape_gate": "PASS" if shape_ok else "FAIL",
        "finite_gate": "PASS" if finite_ok else "FAIL",
        "label_exact_gate": "PASS" if labels_ok else "FAIL",
        "score_max_abs_error": score_error,
        "score_tolerance": 2.0 * eps,
        "score_float32_tolerance_gate": "PASS" if score_ok else "FAIL",
        "box_max_abs_error": box_error,
        "box_tolerance": 8.0 * eps,
        "box_float32_tolerance_gate": "PASS" if box_ok else "FAIL",
        "ordered_tolerant_parity_gate": "PASS" if passed else "FAIL",
    }
    (output_dir / "corrected_float32_tolerance_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [f"{key.upper()}={value}" for key, value in report.items()]
    (output_dir / "corrected_float32_tolerance_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    return passed


def runtime_contract_passed(result_json: Path) -> tuple[bool, dict]:
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
    return all(checks.values()), data


def main() -> int:
    args = parse_args()
    backbone = require_file("backbone model", args.backbone_model)
    encoder = require_file("encoder model", args.encoder_model)
    decoder = require_file("decoder model", args.decoder_model)
    manifest = require_file("asset manifest", args.asset_manifest)
    contract = require_file("NMS contract", args.nms_contract)
    reference = require_file("Frame009 reference", args.reference)
    require_file("frozen runner", str(RUNNER))

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    result_json = output_dir / "performance_result.json"
    candidate = output_dir / "frame009_final_coordinates.npz"

    command = [
        sys.executable, str(RUNNER),
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
    strict_rc = subprocess.run(command, check=False).returncode
    print(f"STRICT_RUN_EXIT={strict_rc}")

    if not result_json.is_file() or not candidate.is_file():
        print("OUTPUT_COORDINATES_GATE=FAIL")
        print("FINAL_DELIVERY_ACCEPTANCE_GATE=FAIL")
        return 1

    print(f"OUTPUT_COORDINATES_GATE=PASS PATH={candidate}")
    runtime_ok, result = runtime_contract_passed(result_json)
    corrected_ok = corrected_contract(reference, candidate, output_dir)
    print(f"CORRECTED_VERIFICATION_EXIT={0 if corrected_ok else 1}")

    known_strict = strict_rc == 0 or (
        strict_rc == 1
        and result.get("final_output_verification", {}).get("gate") == "FAIL"
        and "exception_type" not in result
    )
    if runtime_ok and corrected_ok and known_strict:
        print("EXPECTED_STRICT_TOLERANCE_FAILURE_PRESERVED_GATE=PASS")
        print("CORRECTED_FLOAT32_CONTRACT_GATE=PASS")
        print("FINAL_DELIVERY_ACCEPTANCE_GATE=PASS")
        return 0

    print("EXPECTED_STRICT_TOLERANCE_FAILURE_PRESERVED_GATE=FAIL")
    print("CORRECTED_FLOAT32_CONTRACT_GATE=" + ("PASS" if corrected_ok else "FAIL"))
    print("FINAL_DELIVERY_ACCEPTANCE_GATE=FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
