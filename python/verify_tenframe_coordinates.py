#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import argparse
import json
import traceback

import numpy as np


TENFRAME_KEYS = ("boxes", "scores", "labels", "frame_indices")
FRAME009_KEYS = ("boxes", "scores", "labels")
EXPECTED_FRAME_INDICES = np.arange(10, dtype=np.int64)
EXPECTED_TENFRAME_SHAPES = {
    "boxes": (10, 300, 9),
    "scores": (10, 300),
    "labels": (10, 300),
    "frame_indices": (10,),
}
EXPECTED_TENFRAME_DTYPES = {
    "boxes": np.dtype(np.float32),
    "scores": np.dtype(np.float32),
    "labels": np.dtype(np.int64),
    "frame_indices": np.dtype(np.int64),
}
EXPECTED_FRAME009_SHAPES = {
    "boxes": (300, 9),
    "scores": (300,),
    "labels": (300,),
}
EXPECTED_FRAME009_DTYPES = {
    "boxes": np.dtype(np.float32),
    "scores": np.dtype(np.float32),
    "labels": np.dtype(np.int64),
}


def _load_npz(path, expected_keys):
    path = Path(path)
    if not path.is_file():
        raise RuntimeError("NPZ file missing: {}".format(path))

    with np.load(str(path), allow_pickle=False) as data:
        actual_keys = tuple(data.files)
        arrays = {
            key: np.array(data[key], copy=True)
            for key in expected_keys
            if key in data.files
        }

    return actual_keys, arrays


def verify(tenframe_path, frame009_path):
    report = {
        "audit_type": "BEVFORMER_TENFRAME_COORDINATE_CONTRACT_V1",
        "tenframe_path": str(Path(tenframe_path)),
        "frame009_path": str(Path(frame009_path)),
        "gate": "FAIL",
    }

    try:
        tenframe_keys, tenframe = _load_npz(
            tenframe_path,
            TENFRAME_KEYS,
        )
        frame009_keys, frame009 = _load_npz(
            frame009_path,
            FRAME009_KEYS,
        )

        report["tenframe_keys"] = list(tenframe_keys)
        report["frame009_keys"] = list(frame009_keys)
        report["tenframe_key_gate"] = (
            "PASS" if tenframe_keys == TENFRAME_KEYS else "FAIL"
        )
        report["frame009_key_gate"] = (
            "PASS" if frame009_keys == FRAME009_KEYS else "FAIL"
        )

        missing_tenframe = [
            key for key in TENFRAME_KEYS if key not in tenframe
        ]
        missing_frame009 = [
            key for key in FRAME009_KEYS if key not in frame009
        ]
        if missing_tenframe or missing_frame009:
            raise RuntimeError(
                "Missing keys tenframe={} frame009={}".format(
                    missing_tenframe,
                    missing_frame009,
                )
            )

        shape_gates = {}
        dtype_gates = {}
        for key in TENFRAME_KEYS:
            shape_gates[key] = (
                "PASS"
                if tenframe[key].shape == EXPECTED_TENFRAME_SHAPES[key]
                else "FAIL"
            )
            dtype_gates[key] = (
                "PASS"
                if tenframe[key].dtype == EXPECTED_TENFRAME_DTYPES[key]
                else "FAIL"
            )

        frame009_shape_gates = {}
        frame009_dtype_gates = {}
        for key in FRAME009_KEYS:
            frame009_shape_gates[key] = (
                "PASS"
                if frame009[key].shape == EXPECTED_FRAME009_SHAPES[key]
                else "FAIL"
            )
            frame009_dtype_gates[key] = (
                "PASS"
                if frame009[key].dtype == EXPECTED_FRAME009_DTYPES[key]
                else "FAIL"
            )

        report["tenframe_shape_gates"] = shape_gates
        report["tenframe_dtype_gates"] = dtype_gates
        report["frame009_shape_gates"] = frame009_shape_gates
        report["frame009_dtype_gates"] = frame009_dtype_gates

        finite_gate = bool(
            np.isfinite(tenframe["boxes"]).all()
            and np.isfinite(tenframe["scores"]).all()
        )
        frame_indices_exact = bool(
            np.array_equal(
                tenframe["frame_indices"],
                EXPECTED_FRAME_INDICES,
            )
        )
        boxes_exact = bool(
            np.array_equal(tenframe["boxes"][9], frame009["boxes"])
        )
        scores_exact = bool(
            np.array_equal(tenframe["scores"][9], frame009["scores"])
        )
        labels_exact = bool(
            np.array_equal(tenframe["labels"][9], frame009["labels"])
        )

        report["finite_gate"] = "PASS" if finite_gate else "FAIL"
        report["frame_indices_exact_gate"] = (
            "PASS" if frame_indices_exact else "FAIL"
        )
        report["frame009_boxes_exact_gate"] = (
            "PASS" if boxes_exact else "FAIL"
        )
        report["frame009_scores_exact_gate"] = (
            "PASS" if scores_exact else "FAIL"
        )
        report["frame009_labels_exact_gate"] = (
            "PASS" if labels_exact else "FAIL"
        )

        all_shape_pass = all(
            value == "PASS" for value in shape_gates.values()
        ) and all(
            value == "PASS" for value in frame009_shape_gates.values()
        )
        all_dtype_pass = all(
            value == "PASS" for value in dtype_gates.values()
        ) and all(
            value == "PASS" for value in frame009_dtype_gates.values()
        )

        passed = bool(
            report["tenframe_key_gate"] == "PASS"
            and report["frame009_key_gate"] == "PASS"
            and all_shape_pass
            and all_dtype_pass
            and finite_gate
            and frame_indices_exact
            and boxes_exact
            and scores_exact
            and labels_exact
        )
        report["gate"] = "PASS" if passed else "FAIL"

    except Exception as exc:
        report["gate"] = "FAIL"
        report["exception_type"] = type(exc).__name__
        report["exception_message"] = str(exc)
        report["traceback"] = traceback.format_exc()

    return report


def write_reports(report, json_path, text_path):
    json_path = Path(json_path)
    text_path = Path(text_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "AUDIT_TYPE=BEVFORMER_TENFRAME_COORDINATE_CONTRACT_V1",
        "TENFRAME_COORDINATE_KEY_GATE={}".format(
            report.get("tenframe_key_gate", "FAIL")
        ),
        "FRAME009_COORDINATE_KEY_GATE={}".format(
            report.get("frame009_key_gate", "FAIL")
        ),
        "TENFRAME_COORDINATE_FINITE_GATE={}".format(
            report.get("finite_gate", "FAIL")
        ),
        "TENFRAME_FRAME_INDICES_EXACT_GATE={}".format(
            report.get("frame_indices_exact_gate", "FAIL")
        ),
        "TENFRAME_FRAME009_BOXES_EXACT_GATE={}".format(
            report.get("frame009_boxes_exact_gate", "FAIL")
        ),
        "TENFRAME_FRAME009_SCORES_EXACT_GATE={}".format(
            report.get("frame009_scores_exact_gate", "FAIL")
        ),
        "TENFRAME_FRAME009_LABELS_EXACT_GATE={}".format(
            report.get("frame009_labels_exact_gate", "FAIL")
        ),
        "TENFRAME_COORDINATE_CONTRACT_GATE={}".format(
            report.get("gate", "FAIL")
        ),
    ]

    for group_name in (
        "tenframe_shape_gates",
        "tenframe_dtype_gates",
        "frame009_shape_gates",
        "frame009_dtype_gates",
    ):
        group = report.get(group_name, {})
        for key in sorted(group):
            lines.append(
                "{}_{}_GATE={}".format(
                    group_name.upper(),
                    key.upper(),
                    group[key],
                )
            )

    if "exception_type" in report:
        lines.append(
            "EXCEPTION_TYPE={}".format(report["exception_type"])
        )
        lines.append(
            "EXCEPTION_MESSAGE={}".format(
                report.get("exception_message", "")
            )
        )

    text_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenframe", required=True)
    parser.add_argument("--frame009", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--report-txt", required=True)
    args = parser.parse_args()

    report = verify(args.tenframe, args.frame009)
    write_reports(report, args.report_json, args.report_txt)
    print(Path(args.report_txt).read_text(encoding="utf-8"), end="")
    print("TENFRAME_REPORT_JSON={}".format(args.report_json))
    print("TENFRAME_REPORT_TXT={}".format(args.report_txt))
    return 0 if report["gate"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
