#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import rotate as scipy_rotate


EXPECTED_CONTEXT_SHA256 = {
    "backbone": "ec709ae41f0dc6b25e2f8950abc8e15a5c85301ac31eea1fa1750927e364c945",
    "encoder": "2868cb971a6e291ff9d3d874b2dfe2238c4b545ffdb8af1b01b7b56697ce71a1",
    "decoder": "dacebf6428168bbe0e29410f05285a7fde454860a607c73f983149e607b78d7c",
}

ZERO_PREV_BEV_SHA256 = (
    "ac8ced7a23a8f82aee71ae9e8f5a206f7676a49598be376e53e93ba0d1664a1f"
)

SHAPES = {
    "images": (6, 3, 450, 800),
    "img_feat": (6, 256, 15, 25),
    "encoder_img_feat": (1, 6, 256, 15, 25),
    "can_bus": (1, 18),
    "shift": (1, 2),
    "lidar2img": (1, 6, 4, 4),
    "bev": (1, 2500, 256),
    "decoder": (1, 900, 10),
}

EXPECTED_TENSORS = {
    "backbone": {
        "inputs": {"images": 6480000},
        "outputs": {"img_feat": 576000},
    },
    "encoder": {
        "inputs": {
            "can_bus": 18,
            "img_feat": 576000,
            "lidar2img": 96,
            "shift": 2,
            "prev_bev": 640000,
        },
        "outputs": {"bev_embed": 640000},
    },
    "decoder": {
        "inputs": {"bev_embed": 640000},
        "outputs": {"cls_scores": 9000, "bbox_preds": 9000},
    },
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_native_fp16(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f2").reshape(-1).tobytes()
    ).hexdigest()


def sha256_semantic_fp32(array: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array, dtype="<f4").reshape(-1).tobytes()
    ).hexdigest()


def compare(reference: np.ndarray, output: np.ndarray) -> dict[str, Any]:
    reference = np.ascontiguousarray(reference, dtype=np.float32).reshape(-1)
    output = np.ascontiguousarray(output, dtype=np.float32).reshape(-1)
    if reference.size != output.size:
        raise RuntimeError(
            f"comparison size mismatch reference={reference.size} output={output.size}"
        )

    reference64 = reference.astype(np.float64)
    output64 = output.astype(np.float64)
    difference = output64 - reference64
    absolute = np.abs(difference)

    ref_norm = float(np.linalg.norm(reference64))
    out_norm = float(np.linalg.norm(output64))
    if ref_norm == 0.0 and out_norm == 0.0:
        cosine = 1.0
    elif ref_norm == 0.0 or out_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(reference64, output64) / (ref_norm * out_norm))

    exact = reference.view(np.uint32) == output.view(np.uint32)
    exact_count = int(np.count_nonzero(exact))

    return {
        "element_count": int(reference.size),
        "cosine_similarity": cosine,
        "mae": float(np.mean(absolute)),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "max_abs_error": float(np.max(absolute)) if absolute.size else 0.0,
        "exact_element_count": exact_count,
        "exact_element_ratio": float(exact_count / reference.size) if reference.size else 1.0,
        "bit_exact": bool(exact_count == reference.size),
        "reference_sha256": sha256_semantic_fp32(reference),
        "output_sha256": sha256_semantic_fp32(output),
    }


def normalize_rc(value: Any) -> int:
    return 0 if value is None else int(value)


def create_model(aidlite: Any, model_path: str) -> Any:
    try:
        return aidlite.Model.create_instance(model_path=model_path)
    except TypeError:
        return aidlite.Model.create_instance(model_path)


def build_interpreter(aidlite: Any, model: Any, config: Any) -> tuple[Any, str]:
    for method_name in (
        "build_interpreter_from_model_and_config",
        "build_interpretper_from_model_and_config",
    ):
        if hasattr(aidlite.InterpreterBuilder, method_name):
            method = getattr(aidlite.InterpreterBuilder, method_name)
            try:
                return method(model=model, config=config), method_name
            except TypeError:
                return method(model, config), method_name
    raise RuntimeError("No supported InterpreterBuilder method")


def destroy_interpreter(interpreter: Any) -> tuple[int, str]:
    for method_name in ("destroy", "destory"):
        if hasattr(interpreter, method_name):
            return normalize_rc(getattr(interpreter, method_name)()), method_name
    return 127, "NOT_FOUND"


def flatten_tensor_info(groups: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if groups is None:
        return records
    for graph_index, group in enumerate(groups):
        try:
            tensors = list(group)
        except TypeError:
            tensors = [group]
        for tensor_index, info in enumerate(tensors):
            shape_value = getattr(info, "shape", [])
            try:
                shape = [int(value) for value in shape_value]
            except TypeError:
                shape = []
            records.append(
                {
                    "graph_index": graph_index,
                    "tensor_index": tensor_index,
                    "name": str(getattr(info, "name", "")),
                    "element_count": int(getattr(info, "element_count", -1)),
                    "element_type": str(getattr(info, "element_type", "")),
                    "shape": shape,
                }
            )
    return records


def create_loaded_interpreter(
    aidlite: Any,
    model_name: str,
    model_path: str,
) -> tuple[Any, dict[str, Any]]:
    actual_sha = sha256_file(model_path)
    if actual_sha != EXPECTED_CONTEXT_SHA256[model_name]:
        raise RuntimeError(
            f"{model_name}: Context SHA mismatch "
            f"expected={EXPECTED_CONTEXT_SHA256[model_name]} actual={actual_sha}"
        )
    print(f"{model_name.upper()}_CONTEXT_SHA_GATE=PASS")

    model = create_model(aidlite, model_path)
    config = aidlite.Config.create_instance()
    if model is None or config is None:
        raise RuntimeError(f"{model_name}: Model/Config creation failed")

    config.framework_type = aidlite.FrameworkType.TYPE_QNN240
    config.implement_type = aidlite.ImplementType.TYPE_LOCAL
    config.accelerate_type = aidlite.AccelerateType.TYPE_DSP
    config.qnn_shared_buffer = 0

    interpreter, builder_method = build_interpreter(aidlite, model, config)
    if interpreter is None:
        raise RuntimeError(f"{model_name}: interpreter creation failed")

    init_rc = normalize_rc(interpreter.init())
    load_rc = normalize_rc(interpreter.load_model())
    if init_rc != 0 or load_rc != 0:
        raise RuntimeError(
            f"{model_name}: init/load failed init={init_rc} load={load_rc}"
        )

    input_records = flatten_tensor_info(interpreter.get_input_tensor_info())
    output_records = flatten_tensor_info(interpreter.get_output_tensor_info())
    actual_inputs = {item["name"]: item["element_count"] for item in input_records}
    actual_outputs = {item["name"]: item["element_count"] for item in output_records}
    expected = EXPECTED_TENSORS[model_name]

    if actual_inputs != expected["inputs"] or actual_outputs != expected["outputs"]:
        raise RuntimeError(
            f"{model_name}: tensor contract mismatch "
            f"inputs={actual_inputs} outputs={actual_outputs}"
        )

    print(f"{model_name.upper()}_LOAD_ONLY_GATE=PASS")
    return interpreter, {
        "name": model_name,
        "path": model_path,
        "sha256": actual_sha,
        "builder_method": builder_method,
        "init_rc": init_rc,
        "load_model_rc": load_rc,
        "object_id": id(interpreter),
        "inputs": input_records,
        "outputs": output_records,
    }


def set_input(interpreter: Any, name: str, array: np.ndarray) -> float:
    value = np.ascontiguousarray(array, dtype=np.float32)
    start = time.perf_counter_ns()
    rc = normalize_rc(
        interpreter.set_input_tensor(in_tensor_tag=name, input_data=value)
    )
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if rc != 0:
        raise RuntimeError(f"set_input_tensor failed name={name} rc={rc}")
    return elapsed


def invoke(interpreter: Any, label: str) -> float:
    start = time.perf_counter_ns()
    rc = normalize_rc(interpreter.invoke())
    elapsed = (time.perf_counter_ns() - start) / 1e6
    print(f"{label.upper()}_INVOKE_EXIT={rc}")
    if rc != 0:
        raise RuntimeError(f"{label}: invoke failed rc={rc}")
    return elapsed


def get_output(
    interpreter: Any,
    name: str,
    shape: tuple[int, ...],
) -> tuple[np.ndarray, float]:
    start = time.perf_counter_ns()
    value = interpreter.get_output_tensor(out_tensor_tag=name)
    elapsed = (time.perf_counter_ns() - start) / 1e6
    if value is None:
        raise RuntimeError(f"get_output_tensor returned None name={name}")

    array = np.asarray(value, dtype=np.float32).reshape(-1)
    if array.size != math.prod(shape):
        raise RuntimeError(
            f"{name}: element mismatch expected={math.prod(shape)} actual={array.size}"
        )
    array = np.ascontiguousarray(array.reshape(shape), dtype=np.float32)
    if not np.isfinite(array).all():
        raise RuntimeError(f"{name}: non-finite output")
    return array, elapsed


def load_record(record: dict[str, Any]) -> np.ndarray:
    path = Path(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256_file(path) != record["sha256"]:
        raise RuntimeError(f"staged SHA mismatch: {path}")

    dtype = "<f2" if "float16" in record["dtype"] else "<f4"
    shape = tuple(int(value) for value in record["shape"])
    array = np.fromfile(path, dtype=dtype)
    if array.size != math.prod(shape):
        raise RuntimeError(f"staged element mismatch: {path}")
    return np.ascontiguousarray(array.reshape(shape))


def rotate_prev_bev(
    previous_live_bev: np.ndarray,
    rotation_can_bus: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    angle = float(rotation_can_bus.reshape(-1)[-1])

    previous_native = np.ascontiguousarray(previous_live_bev, dtype="<f2")
    previous_semantic = previous_native.astype(np.float32).reshape(SHAPES["bev"])

    bev_grid = previous_semantic[0].reshape(50, 50, 256)
    rotated_fp32 = scipy_rotate(
        bev_grid,
        angle=angle,
        axes=(0, 1),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(SHAPES["bev"]).astype(np.float32)

    rotated_native = np.ascontiguousarray(rotated_fp32, dtype="<f2")
    rotated_semantic = np.ascontiguousarray(
        rotated_native.astype(np.float32).reshape(SHAPES["bev"]),
        dtype=np.float32,
    )
    return rotated_native, rotated_semantic, angle


def run_frame(
    *,
    frame_index: int,
    interpreters: dict[str, Any],
    assets: dict[str, np.ndarray],
    prev_bev_input: np.ndarray,
    result: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = f"frame{frame_index:03d}"
    frame_result: dict[str, Any] = {
        "frame_index": frame_index,
        "metrics": {},
        "native_sha_checks": {},
        "timing_ms": {},
    }
    result["frames"][label] = frame_result

    frame_result["timing_ms"]["backbone_set_input"] = set_input(
        interpreters["backbone"], "images", assets["images"]
    )
    frame_result["timing_ms"]["backbone_invoke"] = invoke(
        interpreters["backbone"], f"{label}_backbone"
    )
    img_feat, frame_result["timing_ms"]["backbone_get_output"] = get_output(
        interpreters["backbone"], "img_feat", SHAPES["img_feat"]
    )
    frame_result["metrics"]["img_feat"] = compare(
        assets["img_feat_reference"], img_feat
    )

    img_feat_native = np.ascontiguousarray(img_feat, dtype="<f2")
    img_feat_encoder = np.ascontiguousarray(
        img_feat_native.astype(np.float32).reshape(SHAPES["encoder_img_feat"]),
        dtype=np.float32,
    )
    frame_result["metrics"]["img_feat_encoder_handoff"] = compare(
        assets["img_feat_encoder_reference"], img_feat_encoder
    )

    encoder_set_total = 0.0
    for tensor_name, value in (
        ("can_bus", assets["can_bus"]),
        ("img_feat", img_feat_encoder),
        ("lidar2img", assets["lidar2img"]),
        ("shift", assets["shift"]),
        ("prev_bev", prev_bev_input),
    ):
        encoder_set_total += set_input(interpreters["encoder"], tensor_name, value)

    frame_result["timing_ms"]["encoder_set_input_total"] = encoder_set_total
    frame_result["timing_ms"]["encoder_invoke"] = invoke(
        interpreters["encoder"], f"{label}_encoder"
    )
    bev_embed, frame_result["timing_ms"]["encoder_get_output"] = get_output(
        interpreters["encoder"], "bev_embed", SHAPES["bev"]
    )
    frame_result["metrics"]["bev_embed"] = compare(
        assets["bev_embed_reference"], bev_embed
    )

    frame_result["timing_ms"]["decoder_set_input"] = set_input(
        interpreters["decoder"], "bev_embed", bev_embed
    )
    frame_result["timing_ms"]["decoder_invoke"] = invoke(
        interpreters["decoder"], f"{label}_decoder"
    )
    cls_scores, frame_result["timing_ms"]["decoder_get_cls"] = get_output(
        interpreters["decoder"], "cls_scores", SHAPES["decoder"]
    )
    bbox_preds, frame_result["timing_ms"]["decoder_get_bbox"] = get_output(
        interpreters["decoder"], "bbox_preds", SHAPES["decoder"]
    )

    frame_result["metrics"]["cls_scores"] = compare(
        assets["cls_scores_reference"], cls_scores
    )
    frame_result["metrics"]["bbox_preds"] = compare(
        assets["bbox_preds_reference"], bbox_preds
    )

    frame_result["native_sha_checks"] = {
        "img_feat_encoder": (
            sha256_native_fp16(img_feat_native)
            == assets["_native_reference_sha"]["img_feat_encoder"]
        ),
        "bev_embed": (
            sha256_native_fp16(bev_embed)
            == assets["_native_reference_sha"]["bev_embed"]
        ),
        "cls_scores": (
            sha256_native_fp16(cls_scores)
            == assets["_native_reference_sha"]["cls_scores"]
        ),
        "bbox_preds": (
            sha256_native_fp16(bbox_preds)
            == assets["_native_reference_sha"]["bbox_preds"]
        ),
    }

    semantic_pass = all(
        metric["bit_exact"] for metric in frame_result["metrics"].values()
    )
    native_pass = all(frame_result["native_sha_checks"].values())
    frame_result["semantic_bit_exact_gate"] = "PASS" if semantic_pass else "FAIL"
    frame_result["native_sha_gate"] = "PASS" if native_pass else "FAIL"
    frame_result["status"] = "PASS" if semantic_pass and native_pass else "FAIL"

    print(
        f"{label.upper()}_ALL_OUTPUTS_SEMANTIC_BIT_EXACT_GATE="
        + ("PASS" if semantic_pass else "FAIL")
    )
    print(
        f"{label.upper()}_ALL_NATIVE_SHA_GATE="
        + ("PASS" if native_pass else "FAIL")
    )
    print(
        f"{label.upper()}_FRAME_GATE="
        + ("PASS" if frame_result["status"] == "PASS" else "FAIL")
    )

    return bev_embed, cls_scores, bbox_preds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone-model", required=True)
    parser.add_argument("--encoder-model", required=True)
    parser.add_argument("--decoder-model", required=True)
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--result-json", required=True)
    args = parser.parse_args()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    result: dict[str, Any] = {
        "audit_type": "AIDLITE_QNN240_FRAME000_TO_FRAME009_RECURSIVE_V2",
        "pid": os.getpid(),
        "status": "FAIL",
        "frame_indices": list(range(10)),
        "models_loaded_once": True,
        "intermediate_handoff_mode": "NUMPY_MEMORY",
        "intermediate_raw_file_write_performed": False,
        "models": [],
        "frames": {},
        "recursion": {},
        "cleanup": [],
    }

    loaded: list[tuple[str, Any]] = []
    cleanup_pass = True

    try:
        import aidlite

        if (
            int(aidlite.FrameworkType.TYPE_QNN240),
            int(aidlite.ImplementType.TYPE_LOCAL),
            int(aidlite.AccelerateType.TYPE_DSP),
        ) != (109, 3, 3):
            raise RuntimeError("AidLite enum contract mismatch")

        manifest_path = Path(args.asset_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            raise RuntimeError("Asset manifest is not PASS")
        if manifest.get("frame_indices") != list(range(10)):
            raise RuntimeError("Unexpected frame range in manifest")

        result["asset_manifest_sha256"] = sha256_file(manifest_path)
        result["aidlite_version"] = str(aidlite.get_library_version())
        result["aidlite_python_version"] = str(aidlite.get_py_library_version())

        interpreters: dict[str, Any] = {}
        for model_name, model_path in (
            ("backbone", args.backbone_model),
            ("encoder", args.encoder_model),
            ("decoder", args.decoder_model),
        ):
            interpreter, record = create_loaded_interpreter(
                aidlite, model_name, model_path
            )
            interpreters[model_name] = interpreter
            loaded.append((model_name, interpreter))
            result["models"].append(record)

        if len(loaded) != 3:
            raise RuntimeError("Three interpreters were not simultaneously loaded")

        initial_ids = {name: id(obj) for name, obj in loaded}
        print("THREE_INTERPRETERS_RESIDENT_BEFORE_TEN_FRAME_INVOKE_GATE=PASS")

        previous_live_bev: np.ndarray | None = None
        final_cls_scores: np.ndarray | None = None
        final_bbox_preds: np.ndarray | None = None
        recursion_passes: list[bool] = []

        for frame_index in range(10):
            sample = f"sample_{frame_index:03d}"
            frame_manifest = manifest["frames"][sample]["assets"]

            assets = {
                name: load_record(record)
                for name, record in frame_manifest.items()
            }
            assets["_native_reference_sha"] = {
                "img_feat_encoder": frame_manifest[
                    "img_feat_encoder_reference"
                ]["native_reference_sha256"],
                "bev_embed": frame_manifest[
                    "bev_embed_reference"
                ]["native_reference_sha256"],
                "cls_scores": frame_manifest[
                    "cls_scores_reference"
                ]["native_reference_sha256"],
                "bbox_preds": frame_manifest[
                    "bbox_preds_reference"
                ]["native_reference_sha256"],
            }

            frozen_prev_native = assets["prev_bev_reference_native"]
            frozen_prev_semantic = assets["prev_bev_reference_semantic"]

            if frame_index == 0:
                native_sha = sha256_native_fp16(frozen_prev_native)
                zero_count = int(np.count_nonzero(frozen_prev_native))
                scene_start_ok = (
                    native_sha == ZERO_PREV_BEV_SHA256 and zero_count == 0
                )
                result["recursion"]["frame000_scene_start"] = {
                    "native_sha256": native_sha,
                    "nonzero_count": zero_count,
                    "gate": "PASS" if scene_start_ok else "FAIL",
                }
                print(
                    "FRAME000_SCENE_START_ZERO_PREV_BEV_GATE="
                    + ("PASS" if scene_start_ok else "FAIL")
                )
                if not scene_start_ok:
                    raise RuntimeError("Frame000 scene-start prev_bev mismatch")
                prev_bev_input = np.ascontiguousarray(
                    frozen_prev_semantic, dtype=np.float32
                )
            else:
                if previous_live_bev is None:
                    raise RuntimeError("Previous live bev_embed is unavailable")

                rotation_can_bus = assets["rotation_can_bus"]
                generated_native, generated_semantic, angle = rotate_prev_bev(
                    previous_live_bev, rotation_can_bus
                )
                native_exact = bool(
                    np.array_equal(
                        generated_native.reshape(-1),
                        frozen_prev_native.astype("<f2").reshape(-1),
                    )
                )
                semantic_metric = compare(
                    frozen_prev_semantic, generated_semantic
                )
                recursion_ok = native_exact and semantic_metric["bit_exact"]
                recursion_passes.append(recursion_ok)

                result["recursion"][f"frame{frame_index:03d}"] = {
                    "source": f"frame{frame_index - 1:03d}_live_bev_embed",
                    "rotation_angle_deg": angle,
                    "generated_native_sha256": sha256_native_fp16(
                        generated_native
                    ),
                    "frozen_native_sha256": sha256_native_fp16(
                        frozen_prev_native
                    ),
                    "native_bit_exact": native_exact,
                    "semantic_comparison": semantic_metric,
                    "frozen_reference_used_as_model_input": False,
                    "gate": "PASS" if recursion_ok else "FAIL",
                }
                print(
                    f"FRAME{frame_index:03d}_LIVE_PREV_BEV_RECURSION_GATE="
                    + ("PASS" if recursion_ok else "FAIL")
                )
                print(
                    f"FRAME{frame_index:03d}_FROZEN_PREV_BEV_USED_AS_MODEL_INPUT=NO"
                )
                if not recursion_ok:
                    raise RuntimeError(
                        f"Frame{frame_index:03d} recursive prev_bev mismatch"
                    )
                prev_bev_input = generated_semantic

            (
                previous_live_bev,
                final_cls_scores,
                final_bbox_preds,
            ) = run_frame(
                frame_index=frame_index,
                interpreters=interpreters,
                assets=assets,
                prev_bev_input=prev_bev_input,
                result=result,
            )

            current_ids = {name: id(interpreters[name]) for name in interpreters}
            if current_ids != initial_ids:
                raise RuntimeError("Interpreter identity changed during sequence")

        frame_passes = [
            record["status"] == "PASS"
            for record in result["frames"].values()
        ]
        all_frames_pass = len(frame_passes) == 10 and all(frame_passes)
        all_recursion_pass = (
            len(recursion_passes) == 9 and all(recursion_passes)
        )
        interpreter_identity_pass = {
            name: id(interpreters[name]) for name in interpreters
        } == initial_ids

        if final_cls_scores is None or final_bbox_preds is None:
            raise RuntimeError("Frame009 final Decoder outputs are unavailable")

        final_cls_path = output_directory / "frame009_cls_scores_fp32.raw"
        final_bbox_path = output_directory / "frame009_bbox_preds_fp32.raw"
        np.ascontiguousarray(final_cls_scores, dtype="<f4").tofile(final_cls_path)
        np.ascontiguousarray(final_bbox_preds, dtype="<f4").tofile(final_bbox_path)
        result["final_outputs"] = {
            "cls_scores": {
                "path": str(final_cls_path),
                "sha256": sha256_file(final_cls_path),
            },
            "bbox_preds": {
                "path": str(final_bbox_path),
                "sha256": sha256_file(final_bbox_path),
            },
        }

        result["all_ten_frames_pass"] = all_frames_pass
        result["all_nine_recursive_handoffs_pass"] = all_recursion_pass
        result["interpreter_identity_stable"] = interpreter_identity_pass
        result["status"] = (
            "PASS"
            if all_frames_pass
            and all_recursion_pass
            and interpreter_identity_pass
            else "FAIL"
        )

        final_frame = result["frames"]["frame009"]
        result["final_frame_summary"] = {
            "status": final_frame["status"],
            "semantic_bit_exact_gate": final_frame[
                "semantic_bit_exact_gate"
            ],
            "native_sha_gate": final_frame["native_sha_gate"],
        }

        print(
            "ALL_TEN_FRAMES_OUTPUT_GATE="
            + ("PASS" if all_frames_pass else "FAIL")
        )
        print(
            "ALL_NINE_LIVE_RECURSIVE_HANDOFFS_GATE="
            + ("PASS" if all_recursion_pass else "FAIL")
        )
        print(
            "THREE_INTERPRETER_IDENTITY_STABLE_GATE="
            + ("PASS" if interpreter_identity_pass else "FAIL")
        )
        print("INTERMEDIATE_IMG_FEAT_FILE_WRITTEN=NO")
        print("INTERMEDIATE_BEV_EMBED_FILE_WRITTEN=NO")
        print("INTERMEDIATE_PREV_BEV_FILE_WRITTEN=NO")
        print("INTERMEDIATE_HANDOFF_MODE=NUMPY_MEMORY")

    except Exception as exc:
        result["status"] = "FAIL"
        result["exception_type"] = type(exc).__name__
        result["exception_message"] = str(exc)
        result["traceback"] = traceback.format_exc()
        print(f"EXCEPTION_TYPE={type(exc).__name__}")
        print(f"EXCEPTION_MESSAGE={exc}")
        traceback.print_exc()

    finally:
        for model_name, interpreter in reversed(loaded):
            try:
                destroy_rc, destroy_method = destroy_interpreter(interpreter)
            except Exception as exc:
                destroy_rc = 126
                destroy_method = f"EXCEPTION:{type(exc).__name__}:{exc}"

            result["cleanup"].append(
                {
                    "name": model_name,
                    "method": destroy_method,
                    "return_code": destroy_rc,
                }
            )
            if destroy_rc != 0:
                cleanup_pass = False
            print(f"{model_name.upper()}_DESTROY_EXIT={destroy_rc}")

        result["cleanup_gate"] = "PASS" if cleanup_pass else "FAIL"
        if not cleanup_pass:
            result["status"] = "FAIL"
        print(
            "THREE_INTERPRETER_CLEANUP_GATE="
            + ("PASS" if cleanup_pass else "FAIL")
        )

        result_path = Path(args.result_json)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        print(f"RESULT_JSON={result_path}")

    final_pass = result.get("status") == "PASS" and cleanup_pass
    print(
        "AIDLITE_QNN240_FRAME000_TO_FRAME009_RECURSIVE_V2_GATE="
        + ("PASS" if final_pass else "FAIL")
    )
    print(
        "NEXT_ACTION="
        + (
            "ADD_HOST_NMSFREECODER_FOR_FRAME009"
            if final_pass
            else "STOP_AND_DIAGNOSE_TEN_FRAME_RECURSIVE_FAILURE"
        )
    )
    return 0 if final_pass else 1


if __name__ == "__main__":
    sys.exit(main())
