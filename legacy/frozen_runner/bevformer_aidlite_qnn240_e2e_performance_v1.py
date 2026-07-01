#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import argparse
import importlib.util
import inspect
import json
import math
import os
import statistics
import time
import traceback

import numpy as np


WARMUP_COUNT = 3
MEASURED_FRAME_COUNT = 10
SCORE_TOLERANCE = 1.0e-7
BOX_TOLERANCE = 1.0e-6

PACKAGE_DIR = Path(__file__).resolve().parent


def load_local_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        str(path),
    )

    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Cannot load local module: {}".format(path)
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


functional = load_local_module(
    PACKAGE_DIR / "functional_mother.py",
    "bevformer_functional_mother",
)

portable_nms = load_local_module(
    PACKAGE_DIR / "portable_numpy_nmsfreecoder.py",
    "bevformer_portable_numpy_nms",
)


def elapsed_ms(start_ns):
    return (time.perf_counter_ns() - start_ns) / 1.0e6


def timed_set_input(interpreter, name, value):
    contiguous = np.ascontiguousarray(
        value,
        dtype=np.float32,
    )

    start = time.perf_counter_ns()

    rc = functional.normalize_rc(
        interpreter.set_input_tensor(
            in_tensor_tag=name,
            input_data=contiguous,
        )
    )

    duration = elapsed_ms(start)

    if rc != 0:
        raise RuntimeError(
            "set_input_tensor failed name={} rc={}".format(
                name,
                rc,
            )
        )

    return duration


def timed_invoke(interpreter, name):
    start = time.perf_counter_ns()
    rc = functional.normalize_rc(interpreter.invoke())
    duration = elapsed_ms(start)

    if rc != 0:
        raise RuntimeError(
            "{} invoke failed rc={}".format(name, rc)
        )

    return duration


def timed_get_output(interpreter, name, shape):
    start = time.perf_counter_ns()

    value = interpreter.get_output_tensor(
        out_tensor_tag=name
    )

    duration = elapsed_ms(start)

    if value is None:
        raise RuntimeError(
            "get_output_tensor returned None: {}".format(name)
        )

    array = np.asarray(
        value,
        dtype=np.float32,
    ).reshape(-1)

    expected = math.prod(shape)

    if array.size != expected:
        raise RuntimeError(
            "{} element mismatch expected={} actual={}".format(
                name,
                expected,
                array.size,
            )
        )

    array = np.ascontiguousarray(
        array.reshape(shape),
        dtype=np.float32,
    )

    if not np.isfinite(array).all():
        raise RuntimeError(
            "{} contains non-finite output".format(name)
        )

    return array, duration


def normalize_nms_result(result):
    if isinstance(result, dict):
        box_key = next(
            (
                key
                for key in (
                    "boxes",
                    "boxes_3d",
                    "bboxes",
                    "bbox",
                )
                if key in result
            ),
            None,
        )
        score_key = next(
            (
                key
                for key in (
                    "scores",
                    "scores_3d",
                    "score",
                )
                if key in result
            ),
            None,
        )
        label_key = next(
            (
                key
                for key in (
                    "labels",
                    "labels_3d",
                    "label",
                )
                if key in result
            ),
            None,
        )

        if None in (box_key, score_key, label_key):
            raise RuntimeError(
                "Unsupported NMS result keys: {}".format(
                    sorted(result.keys())
                )
            )

        boxes = result[box_key]
        scores = result[score_key]
        labels = result[label_key]

    elif isinstance(result, (tuple, list)) and len(result) == 3:
        boxes, scores, labels = result

    else:
        raise RuntimeError(
            "Unsupported NMS return type: {}".format(
                type(result).__name__
            )
        )

    return (
        np.ascontiguousarray(boxes, dtype=np.float32),
        np.ascontiguousarray(scores, dtype=np.float32).reshape(-1),
        np.ascontiguousarray(labels, dtype=np.int64).reshape(-1),
    )


def decode_coordinates(cls_scores, bbox_preds, contract):
    function = portable_nms.decode_numpy_nmsfreecoder
    signature = inspect.signature(function)
    parameters = signature.parameters

    if "contract" in parameters:
        parameter = parameters["contract"]

        if parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            result = function(
                cls_scores,
                bbox_preds,
                contract=contract,
            )
        else:
            result = function(
                cls_scores,
                bbox_preds,
                contract,
            )

    else:
        positional = [
            parameter
            for parameter in parameters.values()
            if parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if len(positional) >= 3:
            result = function(
                cls_scores,
                bbox_preds,
                contract,
            )
        else:
            result = function(
                cls_scores,
                bbox_preds,
            )

    return normalize_nms_result(result)


def load_reference(path):
    with np.load(str(path), allow_pickle=False) as data:
        keys = set(data.files)

        box_key = next(
            (
                key
                for key in (
                    "boxes",
                    "boxes_3d",
                    "bboxes",
                    "bbox",
                )
                if key in keys
            ),
            None,
        )
        score_key = next(
            (
                key
                for key in (
                    "scores",
                    "scores_3d",
                    "score",
                )
                if key in keys
            ),
            None,
        )
        label_key = next(
            (
                key
                for key in (
                    "labels",
                    "labels_3d",
                    "label",
                )
                if key in keys
            ),
            None,
        )

        if None in (box_key, score_key, label_key):
            raise RuntimeError(
                "Unsupported reference NPZ keys: {}".format(
                    sorted(keys)
                )
            )

        return (
            np.ascontiguousarray(
                data[box_key],
                dtype=np.float32,
            ),
            np.ascontiguousarray(
                data[score_key],
                dtype=np.float32,
            ).reshape(-1),
            np.ascontiguousarray(
                data[label_key],
                dtype=np.int64,
            ).reshape(-1),
        )


def preload_assets(manifest):
    frames = {}

    required_common = (
        "images",
        "can_bus",
        "lidar2img",
        "shift",
    )

    for frame_index in range(MEASURED_FRAME_COUNT):
        sample = "sample_{:03d}".format(frame_index)
        records = manifest["frames"][sample]["assets"]
        values = {}

        for name in required_common:
            if name not in records:
                raise RuntimeError(
                    "{} missing asset {}".format(sample, name)
                )

            values[name] = functional.load_record(records[name])

        if frame_index == 0:
            name = "prev_bev_reference_semantic"

            if name not in records:
                raise RuntimeError(
                    "{} missing asset {}".format(sample, name)
                )

            values[name] = functional.load_record(records[name])

        else:
            name = "rotation_can_bus"

            if name not in records:
                raise RuntimeError(
                    "{} missing asset {}".format(sample, name)
                )

            values[name] = functional.load_record(records[name])

        frames[frame_index] = values

    zero_prev = np.asarray(
        frames[0]["prev_bev_reference_semantic"],
        dtype=np.float32,
    )

    if np.count_nonzero(zero_prev) != 0:
        raise RuntimeError(
            "Frame000 prev_bev is not all zero"
        )

    return frames


def execute_frame(
    frame_index,
    interpreters,
    frame_assets,
    previous_live_bev,
    nms_contract,
):
    frame_start = time.perf_counter_ns()
    timing = {"frame_index": int(frame_index)}

    rotation_start = time.perf_counter_ns()

    if frame_index == 0:
        prev_bev_input = np.ascontiguousarray(
            frame_assets["prev_bev_reference_semantic"],
            dtype=np.float32,
        )
    else:
        if previous_live_bev is None:
            raise RuntimeError(
                "Previous live bev_embed unavailable"
            )

        _, prev_bev_input, _ = functional.rotate_prev_bev(
            previous_live_bev,
            frame_assets["rotation_can_bus"],
        )

    timing["prev_bev_rotate_ms"] = elapsed_ms(
        rotation_start
    )

    timing["backbone_set_input_ms"] = timed_set_input(
        interpreters["backbone"],
        "images",
        frame_assets["images"],
    )

    timing["backbone_invoke_ms"] = timed_invoke(
        interpreters["backbone"],
        "backbone",
    )

    img_feat, timing["backbone_get_output_ms"] = (
        timed_get_output(
            interpreters["backbone"],
            "img_feat",
            functional.SHAPES["img_feat"],
        )
    )

    handoff_start = time.perf_counter_ns()

    img_feat_native = np.ascontiguousarray(
        img_feat,
        dtype="<f2",
    )

    img_feat_encoder = np.ascontiguousarray(
        img_feat_native.astype(np.float32).reshape(
            functional.SHAPES["encoder_img_feat"]
        ),
        dtype=np.float32,
    )

    timing["backbone_to_encoder_handoff_ms"] = (
        elapsed_ms(handoff_start)
    )

    encoder_set_total = 0.0

    for tensor_name, value in (
        ("can_bus", frame_assets["can_bus"]),
        ("img_feat", img_feat_encoder),
        ("lidar2img", frame_assets["lidar2img"]),
        ("shift", frame_assets["shift"]),
        ("prev_bev", prev_bev_input),
    ):
        encoder_set_total += timed_set_input(
            interpreters["encoder"],
            tensor_name,
            value,
        )

    timing["encoder_set_input_ms"] = encoder_set_total

    timing["encoder_invoke_ms"] = timed_invoke(
        interpreters["encoder"],
        "encoder",
    )

    bev_embed, timing["encoder_get_output_ms"] = (
        timed_get_output(
            interpreters["encoder"],
            "bev_embed",
            functional.SHAPES["bev"],
        )
    )

    handoff_start = time.perf_counter_ns()

    decoder_bev = np.ascontiguousarray(
        bev_embed,
        dtype=np.float32,
    )

    timing["encoder_to_decoder_handoff_ms"] = (
        elapsed_ms(handoff_start)
    )

    timing["decoder_set_input_ms"] = timed_set_input(
        interpreters["decoder"],
        "bev_embed",
        decoder_bev,
    )

    timing["decoder_invoke_ms"] = timed_invoke(
        interpreters["decoder"],
        "decoder",
    )

    cls_scores, cls_get_ms = timed_get_output(
        interpreters["decoder"],
        "cls_scores",
        functional.SHAPES["decoder"],
    )

    bbox_preds, bbox_get_ms = timed_get_output(
        interpreters["decoder"],
        "bbox_preds",
        functional.SHAPES["decoder"],
    )

    timing["decoder_get_output_ms"] = (
        cls_get_ms + bbox_get_ms
    )

    nms_start = time.perf_counter_ns()

    boxes, scores, labels = decode_coordinates(
        cls_scores,
        bbox_preds,
        nms_contract,
    )

    timing["nmsfreecoder_ms"] = elapsed_ms(nms_start)
    timing["coordinate_count"] = int(labels.size)

    timing["three_model_invoke_ms"] = (
        timing["backbone_invoke_ms"]
        + timing["encoder_invoke_ms"]
        + timing["decoder_invoke_ms"]
    )

    timing["frame_total_wall_ms"] = elapsed_ms(frame_start)

    return (
        bev_embed,
        boxes,
        scores,
        labels,
        timing,
    )


def summarize(values):
    array = np.asarray(values, dtype=np.float64)

    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def summarize_frames(frames):
    keys = sorted(
        {
            key
            for frame in frames
            for key, value in frame.items()
            if isinstance(value, (int, float))
            and key not in ("frame_index", "coordinate_count")
        }
    )

    return {
        key: summarize([frame[key] for frame in frames])
        for key in keys
    }


def verify_final_output(
    boxes,
    scores,
    labels,
    reference_path,
):
    ref_boxes, ref_scores, ref_labels = load_reference(
        reference_path
    )

    shape_equal = (
        boxes.shape == ref_boxes.shape
        and scores.shape == ref_scores.shape
        and labels.shape == ref_labels.shape
    )

    label_exact = bool(
        shape_equal
        and np.array_equal(labels, ref_labels)
    )

    if shape_equal and scores.size:
        score_error = float(
            np.max(
                np.abs(
                    scores.astype(np.float64)
                    - ref_scores.astype(np.float64)
                )
            )
        )
    else:
        score_error = float("inf")

    if shape_equal and boxes.size:
        box_error = float(
            np.max(
                np.abs(
                    boxes.astype(np.float64)
                    - ref_boxes.astype(np.float64)
                )
            )
        )
    else:
        box_error = float("inf")

    passed = bool(
        shape_equal
        and label_exact
        and score_error <= SCORE_TOLERANCE
        and box_error <= BOX_TOLERANCE
    )

    return {
        "shape_equal": bool(shape_equal),
        "label_exact": label_exact,
        "score_max_abs_error": score_error,
        "box_max_abs_error": box_error,
        "score_tolerance": SCORE_TOLERANCE,
        "box_tolerance": BOX_TOLERANCE,
        "gate": "PASS" if passed else "FAIL",
    }


def main():
    process_main_start = time.perf_counter_ns()

    parser = argparse.ArgumentParser()

    parser.add_argument("--backbone-model", required=True)
    parser.add_argument("--encoder-model", required=True)
    parser.add_argument("--decoder-model", required=True)
    parser.add_argument("--asset-manifest", required=True)

    parser.add_argument(
        "--nms-contract",
        default=str(
            PACKAGE_DIR / "nms_runtime_contract.json"
        ),
    )
    parser.add_argument(
        "--nms-reference",
        default=str(
            PACKAGE_DIR
            / "frame009_numpy_native_reference.npz"
        ),
    )

    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--result-json", required=True)

    args = parser.parse_args()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "audit_type":
            "AIDLITE_QNN240_E2E_PERFORMANCE_RUNNER_V1",
        "pid": os.getpid(),
        "status": "FAIL",
        "measurement_scope":
            "SIX_CAMERA_TENSOR_TO_3D_COORDINATE_ARRAY",
        "jpeg_or_video_decode_included": False,
        "interpreter_load_in_steady_state": False,
        "asset_disk_io_in_measured_loop": False,
        "measured_loop_file_write": False,
        "warmup_count": WARMUP_COUNT,
        "measured_frame_count": MEASURED_FRAME_COUNT,
        "warmup_state_reset_before_measurement": True,
        "models_loaded_once": True,
        "intermediate_handoff_mode": "NUMPY_MEMORY",
        "model_load_timing_ms": {},
        "warmup_frames": [],
        "measured_frames": [],
        "summary_ms": {},
        "final_output_verification": {},
        "cleanup": [],
    }

    loaded = []
    cleanup_pass = True
    final_boxes = None
    final_scores = None
    final_labels = None

    try:
        manifest_path = Path(args.asset_manifest)

        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )

        if manifest.get("status") != "PASS":
            raise RuntimeError(
                "Asset manifest status is not PASS"
            )

        if manifest.get("frame_indices") != list(
            range(MEASURED_FRAME_COUNT)
        ):
            raise RuntimeError(
                "Asset manifest frame range is not 000-009"
            )

        preload_start = time.perf_counter_ns()
        frames = preload_assets(manifest)
        result["asset_preload_ms"] = elapsed_ms(preload_start)

        nms_contract = json.loads(
            Path(args.nms_contract).read_text(
                encoding="utf-8"
            )
        )

        import aidlite

        enum_contract = (
            int(aidlite.FrameworkType.TYPE_QNN240),
            int(aidlite.ImplementType.TYPE_LOCAL),
            int(aidlite.AccelerateType.TYPE_DSP),
        )

        if enum_contract != (109, 3, 3):
            raise RuntimeError(
                "AidLite enum contract mismatch: {}".format(
                    enum_contract
                )
            )

        result["aidlite_library_version"] = str(
            aidlite.get_library_version()
        )
        result["aidlite_python_version"] = str(
            aidlite.get_py_library_version()
        )

        interpreters = {}

        model_load_total_start = time.perf_counter_ns()

        for model_name, model_path in (
            ("backbone", args.backbone_model),
            ("encoder", args.encoder_model),
            ("decoder", args.decoder_model),
        ):
            model_load_start = time.perf_counter_ns()

            interpreter, model_record = (
                functional.create_loaded_interpreter(
                    aidlite,
                    model_name,
                    model_path,
                )
            )

            result["model_load_timing_ms"][model_name] = (
                elapsed_ms(model_load_start)
            )

            interpreters[model_name] = interpreter
            loaded.append((model_name, interpreter))
            result.setdefault("models", []).append(model_record)

        result["model_load_total_ms"] = elapsed_ms(
            model_load_total_start
        )

        if len(loaded) != 3:
            raise RuntimeError(
                "Three interpreters are not simultaneously loaded"
            )

        initial_ids = {
            name: id(interpreter)
            for name, interpreter in loaded
        }

        warmup_previous_bev = None

        for frame_index in range(WARMUP_COUNT):
            (
                warmup_previous_bev,
                _,
                _,
                _,
                warmup_timing,
            ) = execute_frame(
                frame_index,
                interpreters,
                frames[frame_index],
                warmup_previous_bev,
                nms_contract,
            )

            result["warmup_frames"].append(
                {
                    "frame_index": frame_index,
                    "frame_total_wall_ms":
                        warmup_timing["frame_total_wall_ms"],
                }
            )

            if frame_index == 0:
                result[
                    "runner_main_to_first_coordinate_ms"
                ] = elapsed_ms(process_main_start)

        # Mandatory temporal reset: warmup outputs are discarded.
        measured_previous_bev = None

        for frame_index in range(MEASURED_FRAME_COUNT):
            (
                measured_previous_bev,
                final_boxes,
                final_scores,
                final_labels,
                timing,
            ) = execute_frame(
                frame_index,
                interpreters,
                frames[frame_index],
                measured_previous_bev,
                nms_contract,
            )

            result["measured_frames"].append(timing)

        final_ids = {
            name: id(interpreters[name])
            for name in interpreters
        }

        result["interpreter_identity_stable"] = (
            final_ids == initial_ids
        )

        if not result["interpreter_identity_stable"]:
            raise RuntimeError(
                "Interpreter identity changed during performance run"
            )

        if (
            final_boxes is None
            or final_scores is None
            or final_labels is None
        ):
            raise RuntimeError(
                "Final coordinate output is unavailable"
            )

        result["summary_ms"] = summarize_frames(
            result["measured_frames"]
        )

        result["final_output_verification"] = (
            verify_final_output(
                final_boxes,
                final_scores,
                final_labels,
                Path(args.nms_reference),
            )
        )

        result["status"] = (
            "PASS"
            if result["final_output_verification"]["gate"]
            == "PASS"
            else "FAIL"
        )

    except Exception as exc:
        result["status"] = "FAIL"
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["traceback"] = traceback.format_exc()

    finally:
        for model_name, interpreter in reversed(loaded):
            try:
                destroy_rc, destroy_method = (
                    functional.destroy_interpreter(interpreter)
                )
            except Exception as exc:
                destroy_rc = 126
                destroy_method = (
                    "EXCEPTION:{}:{}".format(
                        type(exc).__name__,
                        exc,
                    )
                )

            result["cleanup"].append(
                {
                    "name": model_name,
                    "method": destroy_method,
                    "return_code": destroy_rc,
                }
            )

            if destroy_rc != 0:
                cleanup_pass = False

        result["cleanup_gate"] = (
            "PASS" if cleanup_pass else "FAIL"
        )

        if not cleanup_pass:
            result["status"] = "FAIL"

        if (
            final_boxes is not None
            and final_scores is not None
            and final_labels is not None
        ):
            np.savez(
                str(
                    output_directory
                    / "frame009_final_coordinates.npz"
                ),
                boxes=final_boxes,
                scores=final_scores,
                labels=final_labels,
            )

        result_path.write_text(
            json.dumps(
                result,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        report_lines = [
            "AUDIT_TYPE=AIDLITE_QNN240_E2E_PERFORMANCE_RUNNER_V1",
            "MEASUREMENT_SCOPE={}".format(
                result["measurement_scope"]
            ),
            "WARMUP_COUNT={}".format(WARMUP_COUNT),
            "MEASURED_FRAME_COUNT={}".format(
                MEASURED_FRAME_COUNT
            ),
            "ASSET_DISK_IO_IN_MEASURED_LOOP=NO",
            "MEASURED_LOOP_FILE_WRITE=NO",
            "INTERPRETER_LOAD_IN_STEADY_STATE=NO",
            "INTERMEDIATE_HANDOFF_MODE=NUMPY_MEMORY",
            "INTERPRETER_IDENTITY_STABLE={}".format(
                result.get("interpreter_identity_stable")
            ),
            "FINAL_OUTPUT_VERIFICATION_GATE={}".format(
                result.get(
                    "final_output_verification",
                    {},
                ).get("gate", "FAIL")
            ),
            "CLEANUP_GATE={}".format(
                result.get("cleanup_gate", "FAIL")
            ),
            "FINAL_STATUS={}".format(result["status"]),
        ]

        for metric in (
            "backbone_invoke_ms",
            "encoder_invoke_ms",
            "decoder_invoke_ms",
            "three_model_invoke_ms",
            "nmsfreecoder_ms",
            "frame_total_wall_ms",
        ):
            record = result.get(
                "summary_ms",
                {},
            ).get(metric)

            if not record:
                continue

            for statistic in (
                "mean",
                "median",
                "p90",
                "p95",
                "p99",
                "min",
                "max",
            ):
                report_lines.append(
                    "{}_{}={}".format(
                        metric.upper(),
                        statistic.upper(),
                        record[statistic],
                    )
                )

        report_lines.append(
            "FINAL_VERDICT={}".format(
                "AIDLITE_QNN240_E2E_PERFORMANCE_V1_CLOSED"
                if result["status"] == "PASS"
                else "AIDLITE_QNN240_E2E_PERFORMANCE_V1_FAILED"
            )
        )

        (output_directory / "performance_report.txt").write_text(
            "\n".join(report_lines) + "\n",
            encoding="utf-8",
        )

        print("\n".join(report_lines))
        print("RESULT_JSON={}".format(result_path))
        print(
            "PERFORMANCE_REPORT={}".format(
                output_directory / "performance_report.txt"
            )
        )

    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
