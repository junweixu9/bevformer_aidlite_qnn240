#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
import importlib.util
import json
import os
import sys
import traceback

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parent
BASE_RUNNER_PATH = (
    PACKAGE_DIR / "bevformer_aidlite_qnn240_e2e_performance_v1.py"
)
VERIFIER_PATH = PACKAGE_DIR / "verify_tenframe_coordinates.py"
TENFRAME_FILENAME = "tenframe_final_coordinates.npz"
FRAME009_FILENAME = "frame009_final_coordinates.npz"
REPORT_JSON_FILENAME = "tenframe_coordinate_export_report.json"
REPORT_TEXT_FILENAME = "tenframe_coordinate_export_report.txt"


def load_local_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load local module: {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_option_value(argv, option):
    prefix = option + "="
    for index, value in enumerate(argv):
        if value == option:
            if index + 1 >= len(argv):
                raise RuntimeError(
                    "Missing value after command option {}".format(option)
                )
            return argv[index + 1]
        if value.startswith(prefix):
            return value[len(prefix):]
    raise RuntimeError("Required command option absent: {}".format(option))


def evaluate_base_runner_outcome(
    base_rc,
    result_path,
    warmup_count,
    measured_count,
):
    result_path = Path(result_path)
    if not result_path.is_file() or result_path.stat().st_size == 0:
        raise RuntimeError(
            "Base runner result JSON missing: {}".format(result_path)
        )

    runtime_result = json.loads(
        result_path.read_text(encoding="utf-8")
    )
    no_exception = "exception_type" not in runtime_result
    runtime_ok = bool(
        len(runtime_result.get("warmup_frames", []))
        == int(warmup_count)
        and len(runtime_result.get("measured_frames", []))
        == int(measured_count)
        and runtime_result.get("interpreter_identity_stable") is True
        and runtime_result.get("cleanup_gate") == "PASS"
        and no_exception
    )
    strict_gate = runtime_result.get(
        "final_output_verification",
        {},
    ).get("gate", "FAIL")
    known_strict_outcome = bool(
        int(base_rc) == 0
        or (
            int(base_rc) == 1
            and strict_gate == "FAIL"
            and no_exception
        )
    )
    passed = bool(runtime_ok and known_strict_outcome)

    return {
        "runtime_result": runtime_result,
        "runtime_ok": runtime_ok,
        "strict_gate": strict_gate,
        "known_strict_outcome": known_strict_outcome,
        "gate": "PASS" if passed else "FAIL",
    }


def stack_collected(records, measured_count=10):
    if len(records) != measured_count:
        raise RuntimeError(
            "Measured coordinate record count mismatch: expected={} actual={}"
            .format(measured_count, len(records))
        )

    frame_indices = np.asarray(
        [record["frame_index"] for record in records],
        dtype=np.int64,
    )
    expected_indices = np.arange(measured_count, dtype=np.int64)
    if not np.array_equal(frame_indices, expected_indices):
        raise RuntimeError(
            "Measured frame indices mismatch: {}".format(
                frame_indices.tolist()
            )
        )

    boxes = np.ascontiguousarray(
        np.stack([record["boxes"] for record in records], axis=0),
        dtype=np.float32,
    )
    scores = np.ascontiguousarray(
        np.stack([record["scores"] for record in records], axis=0),
        dtype=np.float32,
    )
    labels = np.ascontiguousarray(
        np.stack([record["labels"] for record in records], axis=0),
        dtype=np.int64,
    )

    expected_shapes = {
        "boxes": (measured_count, 300, 9),
        "scores": (measured_count, 300),
        "labels": (measured_count, 300),
    }
    actual_shapes = {
        "boxes": boxes.shape,
        "scores": scores.shape,
        "labels": labels.shape,
    }
    if actual_shapes != expected_shapes:
        raise RuntimeError(
            "Ten-frame coordinate shape mismatch: {}".format(actual_shapes)
        )

    if not np.isfinite(boxes).all():
        raise RuntimeError("Ten-frame boxes contain non-finite values")
    if not np.isfinite(scores).all():
        raise RuntimeError("Ten-frame scores contain non-finite values")

    return boxes, scores, labels, frame_indices


def write_npz_atomic(path, boxes, scores, labels, frame_indices):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")

    try:
        with temporary.open("wb") as handle:
            np.savez(
                handle,
                boxes=boxes,
                scores=scores,
                labels=labels,
                frame_indices=frame_indices,
            )
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    try:
        output_directory = Path(
            get_option_value(sys.argv[1:], "--output-directory")
        )
        base_runner = load_local_module(
            BASE_RUNNER_PATH,
            "bevformer_aidlite_qnn240_frozen_performance_runner",
        )
        verifier = load_local_module(
            VERIFIER_PATH,
            "bevformer_tenframe_coordinate_verifier",
        )

        original_execute_frame = base_runner.execute_frame
        collected = []
        call_count = 0

        def wrapped_execute_frame(
            frame_index,
            interpreters,
            frame_assets,
            previous_live_bev,
            nms_contract,
        ):
            nonlocal call_count

            result = original_execute_frame(
                frame_index,
                interpreters,
                frame_assets,
                previous_live_bev,
                nms_contract,
            )

            call_index = call_count
            call_count += 1

            measured_begin = int(base_runner.WARMUP_COUNT)
            measured_end = measured_begin + int(
                base_runner.MEASURED_FRAME_COUNT
            )

            if measured_begin <= call_index < measured_end:
                measured_index = call_index - measured_begin
                (
                    _,
                    boxes,
                    scores,
                    labels,
                    timing,
                ) = result

                timing_index = int(timing["frame_index"])
                if (
                    int(frame_index) != measured_index
                    or timing_index != measured_index
                ):
                    raise RuntimeError(
                        "Measured frame identity mismatch: "
                        "call={} argument={} timing={}".format(
                            measured_index,
                            frame_index,
                            timing_index,
                        )
                    )

                collected.append(
                    {
                        "frame_index": measured_index,
                        "boxes": np.ascontiguousarray(
                            boxes,
                            dtype=np.float32,
                        ).copy(),
                        "scores": np.ascontiguousarray(
                            scores,
                            dtype=np.float32,
                        ).reshape(-1).copy(),
                        "labels": np.ascontiguousarray(
                            labels,
                            dtype=np.int64,
                        ).reshape(-1).copy(),
                    }
                )

            return result

        base_runner.execute_frame = wrapped_execute_frame
        try:
            base_rc = int(base_runner.main())
        finally:
            base_runner.execute_frame = original_execute_frame

        result_json_path = Path(
            get_option_value(sys.argv[1:], "--result-json")
        )
        base_outcome = evaluate_base_runner_outcome(
            base_rc,
            result_json_path,
            warmup_count=int(base_runner.WARMUP_COUNT),
            measured_count=int(base_runner.MEASURED_FRAME_COUNT),
        )

        if base_outcome["gate"] != "PASS":
            print("BASE_RUNNER_EXIT={}".format(base_rc))
            print(
                "BASE_RUNNER_RUNTIME_GATE={}".format(
                    "PASS" if base_outcome["runtime_ok"] else "FAIL"
                )
            )
            print(
                "BASE_RUNNER_STRICT_OUTPUT_GATE={}".format(
                    base_outcome["strict_gate"]
                )
            )
            print(
                "BASE_RUNNER_KNOWN_STRICT_OUTCOME_GATE={}".format(
                    "PASS"
                    if base_outcome["known_strict_outcome"]
                    else "FAIL"
                )
            )
            print("TENFRAME_COORDINATE_EXPORT_GATE=FAIL")
            return base_rc if base_rc != 0 else 1

        expected_calls = int(
            base_runner.WARMUP_COUNT
            + base_runner.MEASURED_FRAME_COUNT
        )
        if call_count != expected_calls:
            raise RuntimeError(
                "execute_frame call count mismatch: expected={} actual={}"
                .format(expected_calls, call_count)
            )

        boxes, scores, labels, frame_indices = stack_collected(
            collected,
            measured_count=int(base_runner.MEASURED_FRAME_COUNT),
        )

        tenframe_path = output_directory / TENFRAME_FILENAME
        frame009_path = output_directory / FRAME009_FILENAME
        report_json_path = output_directory / REPORT_JSON_FILENAME
        report_text_path = output_directory / REPORT_TEXT_FILENAME

        if not frame009_path.is_file():
            raise RuntimeError(
                "Base Frame009 output missing: {}".format(frame009_path)
            )

        write_npz_atomic(
            tenframe_path,
            boxes,
            scores,
            labels,
            frame_indices,
        )

        report = verifier.verify(tenframe_path, frame009_path)
        report.update(
            {
                "base_runner_exit": base_rc,
                "base_runner_runtime_gate": (
                    "PASS" if base_outcome["runtime_ok"] else "FAIL"
                ),
                "base_runner_strict_output_gate":
                    base_outcome["strict_gate"],
                "base_runner_known_strict_outcome_gate": (
                    "PASS"
                    if base_outcome["known_strict_outcome"]
                    else "FAIL"
                ),
                "base_runner_outcome_gate": base_outcome["gate"],
                "execute_frame_call_count": call_count,
                "warmup_count": int(base_runner.WARMUP_COUNT),
                "measured_frame_count": int(
                    base_runner.MEASURED_FRAME_COUNT
                ),
                "measured_loop_file_write": False,
                "sidecar_memory_collection_enabled": True,
                "sidecar_collection_position":
                    "AFTER_EXECUTE_FRAME_INTERNAL_TIMING",
                "sidecar_file_write_position":
                    "AFTER_BASE_RUNNER_MAIN_RETURN",
            }
        )
        verifier.write_reports(
            report,
            report_json_path,
            report_text_path,
        )

        print(report_text_path.read_text(encoding="utf-8"), end="")
        print("BASE_RUNNER_EXIT={}".format(base_rc))
        print(
            "BASE_RUNNER_RUNTIME_GATE={}".format(
                "PASS" if base_outcome["runtime_ok"] else "FAIL"
            )
        )
        print(
            "BASE_RUNNER_STRICT_OUTPUT_GATE={}".format(
                base_outcome["strict_gate"]
            )
        )
        print(
            "BASE_RUNNER_KNOWN_STRICT_OUTCOME_GATE={}".format(
                "PASS"
                if base_outcome["known_strict_outcome"]
                else "FAIL"
            )
        )
        print("BASE_RUNNER_OUTCOME_GATE={}".format(base_outcome["gate"]))
        print("EXECUTE_FRAME_CALL_COUNT={}".format(call_count))
        print("SIDECAR_MEMORY_COLLECTION_ENABLED=YES")
        print("MEASURED_LOOP_FILE_WRITE=NO")
        print("TENFRAME_COORDINATES={}".format(tenframe_path))
        print("TENFRAME_REPORT_JSON={}".format(report_json_path))
        print("TENFRAME_REPORT_TXT={}".format(report_text_path))
        print(
            "TENFRAME_COORDINATE_EXPORT_GATE={}".format(
                report["gate"]
            )
        )
        return 0 if report["gate"] == "PASS" else 2

    except Exception as exc:
        print("TENFRAME_COORDINATE_EXPORT_GATE=FAIL")
        print("EXCEPTION_TYPE={}".format(type(exc).__name__))
        print("EXCEPTION_MESSAGE={}".format(exc))
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
