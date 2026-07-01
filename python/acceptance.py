from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_coordinates(path: Path):
    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)
        box_key = next(
            (key for key in ("boxes", "boxes_3d", "bboxes", "bbox") if key in keys),
            None,
        )
        score_key = next(
            (key for key in ("scores", "scores_3d", "score") if key in keys),
            None,
        )
        label_key = next(
            (key for key in ("labels", "labels_3d", "label") if key in keys),
            None,
        )
        if None in (box_key, score_key, label_key):
            raise RuntimeError(f"unsupported coordinate keys: {sorted(keys)}")
        return (
            np.asarray(data[box_key], dtype=np.float32),
            np.asarray(data[score_key], dtype=np.float32).reshape(-1),
            np.asarray(data[label_key], dtype=np.int64).reshape(-1),
        )


def verify_corrected_contract(
    reference_path: Path,
    candidate_path: Path,
    output_directory: Path,
) -> bool:
    ref_boxes, ref_scores, ref_labels = _load_coordinates(reference_path)
    boxes, scores, labels = _load_coordinates(candidate_path)
    eps = float(np.finfo(np.float32).eps)

    shape_ok = (
        boxes.shape == ref_boxes.shape == (300, 9)
        and scores.shape == ref_scores.shape == (300,)
        and labels.shape == ref_labels.shape == (300,)
    )
    finite_ok = bool(np.isfinite(boxes).all() and np.isfinite(scores).all())
    labels_ok = bool(shape_ok and np.array_equal(labels, ref_labels))
    score_error = (
        float(np.max(np.abs(scores.astype(np.float64) - ref_scores.astype(np.float64))))
        if shape_ok else float("inf")
    )
    box_error = (
        float(np.max(np.abs(boxes.astype(np.float64) - ref_boxes.astype(np.float64))))
        if shape_ok else float("inf")
    )
    score_ok = score_error <= 2.0 * eps
    box_ok = box_error <= 8.0 * eps
    passed = bool(shape_ok and finite_ok and labels_ok and score_ok and box_ok)

    report = {
        "float32_epsilon": eps,
        "shape_gate": "PASS" if shape_ok else "FAIL",
        "finite_gate": "PASS" if finite_ok else "FAIL",
        "label_exact_gate": "PASS" if labels_ok else "FAIL",
        "score_max_abs_error": score_error,
        "score_tolerance": 2.0 * eps,
        "score_error_in_epsilon": score_error / eps,
        "score_float32_tolerance_gate": "PASS" if score_ok else "FAIL",
        "box_max_abs_error": box_error,
        "box_tolerance": 8.0 * eps,
        "box_error_in_epsilon": box_error / eps,
        "box_float32_tolerance_gate": "PASS" if box_ok else "FAIL",
        "ordered_tolerant_parity_gate": "PASS" if passed else "FAIL",
    }

    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "corrected_float32_tolerance_report.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [f"{key.upper()}={value}" for key, value in report.items()]
    (output_directory / "corrected_float32_tolerance_report.txt").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print("\n".join(lines))
    return passed


def validate_runtime_result(result_path: Path) -> tuple[bool, dict]:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    checks = {
        "WARMUP_COUNT_GATE": len(data.get("warmup_frames", [])) == 3,
        "MEASURED_FRAME_COUNT_GATE": len(data.get("measured_frames", [])) == 10,
        "INTERPRETER_IDENTITY_STABLE_GATE": data.get("interpreter_identity_stable") is True,
        "CLEANUP_GATE": data.get("cleanup_gate") == "PASS",
        "NO_RUNTIME_EXCEPTION_GATE": "exception_type" not in data,
    }
    for key, passed in checks.items():
        print(f"{key}={'PASS' if passed else 'FAIL'}")
    strict_gate = data.get("final_output_verification", {}).get("gate", "FAIL")
    print(f"ORIGINAL_STRICT_OUTPUT_GATE={strict_gate}")
    return all(checks.values()), data


def final_acceptance(
    strict_exit: int,
    result_path: Path,
    candidate_path: Path,
    reference_path: Path,
    output_directory: Path,
) -> int:
    if not result_path.is_file() or not candidate_path.is_file():
        print("OUTPUT_COORDINATES_GATE=FAIL")
        print("FINAL_DELIVERY_ACCEPTANCE_GATE=FAIL")
        return 1

    print(f"OUTPUT_COORDINATES_GATE=PASS PATH={candidate_path}")
    runtime_ok, result = validate_runtime_result(result_path)
    corrected_ok = verify_corrected_contract(
        reference_path,
        candidate_path,
        output_directory,
    )
    print(f"CORRECTED_VERIFICATION_EXIT={0 if corrected_ok else 1}")

    known_strict_outcome = strict_exit == 0 or (
        strict_exit == 1
        and result.get("final_output_verification", {}).get("gate") == "FAIL"
        and "exception_type" not in result
    )

    if runtime_ok and corrected_ok and known_strict_outcome:
        print("EXPECTED_STRICT_TOLERANCE_FAILURE_PRESERVED_GATE=PASS")
        print("CORRECTED_FLOAT32_CONTRACT_GATE=PASS")
        print("FINAL_DELIVERY_ACCEPTANCE_GATE=PASS")
        return 0

    print("EXPECTED_STRICT_TOLERANCE_FAILURE_PRESERVED_GATE=FAIL")
    print("CORRECTED_FLOAT32_CONTRACT_GATE=" + ("PASS" if corrected_ok else "FAIL"))
    print("FINAL_DELIVERY_ACCEPTANCE_GATE=FAIL")
    return 1
