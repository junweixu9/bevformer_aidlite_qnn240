from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_contract", ROOT / "python" / "verify_contract.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_result(path: Path, boxes, scores, labels) -> None:
    np.savez(path, boxes=boxes, scores=scores, labels=labels)


def base_arrays():
    boxes = np.zeros((300, 9), dtype=np.float32)
    scores = np.linspace(0.0, 1.0, 300, dtype=np.float32)
    labels = np.arange(300, dtype=np.int64) % 10
    return boxes, scores, labels


def test_corrected_contract_accepts_one_and_four_epsilon(tmp_path):
    boxes, scores, labels = base_arrays()
    ref = tmp_path / "ref.npz"
    cand = tmp_path / "cand.npz"
    write_result(ref, boxes, scores, labels)

    cand_boxes = boxes.copy()
    cand_scores = scores.copy()
    cand_scores[100] = np.nextafter(cand_scores[100], np.float32(np.inf))
    value = cand_boxes[0, 0]
    for _ in range(4):
        value = np.nextafter(value, np.float32(np.inf))
    cand_boxes[0, 0] = value
    write_result(cand, cand_boxes, cand_scores, labels)

    result = MODULE.verify(ref, cand)
    assert result["label_exact"]
    assert result["score_gate"] == "PASS"
    assert result["box_gate"] == "PASS"
    assert result["gate"] == "PASS"


def test_contract_rejects_label_change(tmp_path):
    boxes, scores, labels = base_arrays()
    ref = tmp_path / "ref.npz"
    cand = tmp_path / "cand.npz"
    write_result(ref, boxes, scores, labels)
    changed = labels.copy()
    changed[0] = 9
    write_result(cand, boxes, scores, changed)
    assert MODULE.verify(ref, cand)["gate"] == "FAIL"


def test_contract_rejects_non_finite_candidate(tmp_path):
    boxes, scores, labels = base_arrays()
    ref = tmp_path / "ref.npz"
    cand = tmp_path / "cand.npz"
    write_result(ref, boxes, scores, labels)
    changed = boxes.copy()
    changed[0, 0] = np.nan
    write_result(cand, changed, scores, labels)
    assert MODULE.verify(ref, cand)["gate"] == "FAIL"
