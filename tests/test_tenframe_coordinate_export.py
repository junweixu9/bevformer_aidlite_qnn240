from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


VERIFIER = load_module(
    ROOT / "python" / "verify_tenframe_coordinates.py",
    "verify_tenframe_coordinates_test",
)
SIDECAR = load_module(
    ROOT
    / "python"
    / "bevformer_aidlite_qnn240_e2e_tenframe_export_v1.py",
    "tenframe_sidecar_test",
)


def make_arrays():
    boxes = np.arange(
        10 * 300 * 9,
        dtype=np.float32,
    ).reshape(10, 300, 9)
    scores = np.linspace(
        0.0,
        1.0,
        num=10 * 300,
        dtype=np.float32,
    ).reshape(10, 300)
    labels = (
        np.arange(10 * 300, dtype=np.int64).reshape(10, 300)
        % 10
    )
    frame_indices = np.arange(10, dtype=np.int64)
    return boxes, scores, labels, frame_indices


def write_contract_files(tmp_path):
    boxes, scores, labels, frame_indices = make_arrays()
    tenframe = tmp_path / "tenframe.npz"
    frame009 = tmp_path / "frame009.npz"
    np.savez(
        tenframe,
        boxes=boxes,
        scores=scores,
        labels=labels,
        frame_indices=frame_indices,
    )
    np.savez(
        frame009,
        boxes=boxes[9],
        scores=scores[9],
        labels=labels[9],
    )
    return tenframe, frame009


def test_tenframe_contract_accepts_exact_frame009(tmp_path):
    tenframe, frame009 = write_contract_files(tmp_path)
    report = VERIFIER.verify(tenframe, frame009)
    assert report["frame_indices_exact_gate"] == "PASS"
    assert report["frame009_boxes_exact_gate"] == "PASS"
    assert report["frame009_scores_exact_gate"] == "PASS"
    assert report["frame009_labels_exact_gate"] == "PASS"
    assert report["gate"] == "PASS"


def test_tenframe_contract_rejects_frame009_mismatch(tmp_path):
    tenframe, frame009 = write_contract_files(tmp_path)
    with np.load(str(frame009), allow_pickle=False) as data:
        boxes = data["boxes"].copy()
        scores = data["scores"].copy()
        labels = data["labels"].copy()
    boxes[0, 0] = np.nextafter(
        boxes[0, 0],
        np.float32(np.inf),
    )
    np.savez(frame009, boxes=boxes, scores=scores, labels=labels)
    report = VERIFIER.verify(tenframe, frame009)
    assert report["frame009_boxes_exact_gate"] == "FAIL"
    assert report["gate"] == "FAIL"


def test_tenframe_contract_rejects_wrong_dtype(tmp_path):
    tenframe, frame009 = write_contract_files(tmp_path)
    boxes, scores, labels, frame_indices = make_arrays()
    np.savez(
        tenframe,
        boxes=boxes,
        scores=scores,
        labels=labels.astype(np.int32),
        frame_indices=frame_indices,
    )
    report = VERIFIER.verify(tenframe, frame009)
    assert report["tenframe_dtype_gates"]["labels"] == "FAIL"
    assert report["gate"] == "FAIL"


def test_stack_collected_produces_contract_shapes():
    boxes, scores, labels, _ = make_arrays()
    records = [
        {
            "frame_index": index,
            "boxes": boxes[index],
            "scores": scores[index],
            "labels": labels[index],
        }
        for index in range(10)
    ]
    result = SIDECAR.stack_collected(records)
    assert result[0].shape == (10, 300, 9)
    assert result[1].shape == (10, 300)
    assert result[2].shape == (10, 300)
    assert result[3].dtype == np.int64
    assert np.array_equal(result[3], np.arange(10, dtype=np.int64))


def test_stack_collected_rejects_nonsequential_indices():
    boxes, scores, labels, _ = make_arrays()
    records = [
        {
            "frame_index": index,
            "boxes": boxes[index],
            "scores": scores[index],
            "labels": labels[index],
        }
        for index in range(10)
    ]
    records[7]["frame_index"] = 8
    with pytest.raises(RuntimeError, match="frame indices mismatch"):
        SIDECAR.stack_collected(records)
