#!/usr/bin/env python3
"""Corrected float32-epsilon contract verifier for Frame009 NMSFreeCoder output.

This verifier is intentionally SEPARATE from the frozen Runner.
It applies the corrected tolerances defined in the accuracy baseline
(ACCURACY_BASELINE.txt), not the original strict 1e-7 threshold.

Contract:
  labels  — exact match, element-wise
  scores  — max abs error ≤ 2 × float32 epsilon (~2.384e-07)
  boxes   — max abs error ≤ 8 × float32 epsilon (~9.537e-07)
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

FLOAT32_EPSILON = float(np.finfo(np.float32).eps)  # 1.1920928955078125e-07
SCORE_TOLERANCE = 2.0 * FLOAT32_EPSILON             # 2.384e-07
BOX_TOLERANCE   = 8.0 * FLOAT32_EPSILON             # 9.537e-07


def load_reference(path):
    """Load boxes, scores, labels from a NumPy reference NPZ.

    Accepts common key variants: boxes/boxes_3d/bboxes, scores/scores_3d/score,
    labels/labels_3d/label.
    """
    with np.load(str(path), allow_pickle=False) as data:
        keys = set(data.files)

        box_key = next(
            (k for k in ("boxes", "boxes_3d", "bboxes", "bbox") if k in keys),
            None,
        )
        score_key = next(
            (k for k in ("scores", "scores_3d", "score") if k in keys),
            None,
        )
        label_key = next(
            (k for k in ("labels", "labels_3d", "label") if k in keys),
            None,
        )
        if None in (box_key, score_key, label_key):
            raise RuntimeError(
                "Unsupported reference NPZ keys: {}".format(sorted(keys))
            )
        return (
            np.ascontiguousarray(data[box_key], dtype=np.float32),
            np.ascontiguousarray(data[score_key], dtype=np.float32).reshape(-1),
            np.ascontiguousarray(data[label_key], dtype=np.int64).reshape(-1),
        )


def verify(reference_path, candidate_path):
    ref_boxes, ref_scores, ref_labels = load_reference(reference_path)
    cand_boxes, cand_scores, cand_labels = load_reference(candidate_path)

    shape_equal = (
        cand_boxes.shape == ref_boxes.shape
        and cand_scores.shape == ref_scores.shape
        and cand_labels.shape == ref_labels.shape
    )

    all_finite = bool(
        np.all(np.isfinite(cand_boxes))
        and np.all(np.isfinite(cand_scores))
        and np.all(np.isfinite(cand_labels))
    )

    label_exact = bool(shape_equal and np.array_equal(cand_labels, ref_labels))

    if shape_equal and cand_scores.size:
        score_error = float(
            np.max(
                np.abs(
                    cand_scores.astype(np.float64)
                    - ref_scores.astype(np.float64)
                )
            )
        )
    else:
        score_error = float("inf")

    if shape_equal and cand_boxes.size:
        box_error = float(
            np.max(
                np.abs(
                    cand_boxes.astype(np.float64)
                    - ref_boxes.astype(np.float64)
                )
            )
        )
    else:
        box_error = float("inf")

    score_pass = bool(score_error <= SCORE_TOLERANCE)
    box_pass = bool(box_error <= BOX_TOLERANCE)

    all_pass = bool(
        shape_equal and all_finite and label_exact and score_pass and box_pass
    )

    return {
        "shape_equal": shape_equal,
        "finite": all_finite,
        "label_exact": label_exact,
        "score_max_abs_error": score_error,
        "score_tolerance": SCORE_TOLERANCE,
        "score_float32_epsilon_multiple": score_error / FLOAT32_EPSILON if FLOAT32_EPSILON else float("inf"),
        "score_gate": "PASS" if score_pass else "FAIL",
        "box_max_abs_error": box_error,
        "box_tolerance": BOX_TOLERANCE,
        "box_float32_epsilon_multiple": box_error / FLOAT32_EPSILON if FLOAT32_EPSILON else float("inf"),
        "box_gate": "PASS" if box_pass else "FAIL",
        "float32_epsilon": FLOAT32_EPSILON,
        "gate": "PASS" if all_pass else "FAIL",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Corrected float32-epsilon contract verifier"
    )
    parser.add_argument("--reference", required=True, help="Frame009 NumPy-native reference .npz")
    parser.add_argument("--candidate", required=True, help="Candidate frame009_final_coordinates.npz")
    parser.add_argument("--report-json", required=True, help="Output JSON report path")
    parser.add_argument("--report-txt", required=True, help="Output TXT report path")
    args = parser.parse_args()

    result = verify(args.reference, args.candidate)

    # Write JSON
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    # Write human-readable TXT report
    lines = [
        "AUDIT_TYPE=CORRECTED_FLOAT32_EPSILON_CONTRACT_VERIFICATION",
        "REFERENCE={}".format(args.reference),
        "CANDIDATE={}".format(args.candidate),
        "FLOAT32_EPSILON={:.15e}".format(FLOAT32_EPSILON),
        "SCORE_TOLERANCE={:.15e}  ({} x epsilon)".format(
            SCORE_TOLERANCE, SCORE_TOLERANCE / FLOAT32_EPSILON
        ),
        "BOX_TOLERANCE={:.15e}  ({} x epsilon)".format(
            BOX_TOLERANCE, BOX_TOLERANCE / FLOAT32_EPSILON
        ),
        "",
        "SHAPE_GATE={}".format("PASS" if result["shape_equal"] else "FAIL"),
        "FINITE_GATE={}".format("PASS" if result["finite"] else "FAIL"),
        "LABEL_EXACT_GATE={}".format("PASS" if result["label_exact"] else "FAIL"),
        "SCORE_FLOAT32_TOLERANCE_GATE={}".format(result["score_gate"]),
        "BOX_FLOAT32_TOLERANCE_GATE={}".format(result["box_gate"]),
        "",
        "SCORE_MAX_ABS_ERROR={:.15e}  ({} x epsilon)".format(
            result["score_max_abs_error"], result["score_float32_epsilon_multiple"]
        ),
        "BOX_MAX_ABS_ERROR={:.15e}  ({} x epsilon)".format(
            result["box_max_abs_error"], result["box_float32_epsilon_multiple"]
        ),
        "",
        "ORDERED_TOLERANT_PARITY_GATE={}".format(result["gate"]),
        "CORRECTED_VERIFICATION_GATE={}".format(result["gate"]),
    ]
    Path(args.report_txt).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Emit gates to stdout and exit accordingly
    for line in lines:
        print(line)

    if result["gate"] == "PASS":
        print("CORRECTED_VERIFICATION_EXIT=0")
        sys.exit(0)
    else:
        print("CORRECTED_VERIFICATION_EXIT=1")
        sys.exit(1)


if __name__ == "__main__":
    main()
