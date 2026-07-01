import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from acceptance import final_acceptance, verify_corrected_contract


def write_coordinates(path: Path, boxes, scores, labels):
    np.savez(path, boxes=boxes, scores=scores, labels=labels)


def write_runtime_result(path: Path, strict_gate="FAIL"):
    data = {
        "warmup_frames": [{}, {}, {}],
        "measured_frames": [{} for _ in range(10)],
        "interpreter_identity_stable": True,
        "cleanup_gate": "PASS",
        "final_output_verification": {"gate": strict_gate},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_corrected_contract_requires_ordered_300_by_9_output(tmp_path):
    boxes = np.ones((300, 9), dtype=np.float32)
    scores = np.ones((300,), dtype=np.float32)
    labels = np.arange(300, dtype=np.int64) % 10
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "candidate.npz"
    write_coordinates(reference, boxes, scores, labels)
    write_coordinates(candidate, boxes, scores, labels)
    assert verify_corrected_contract(reference, candidate, tmp_path)


def test_final_acceptance_allows_only_known_strict_failure(tmp_path):
    boxes = np.ones((300, 9), dtype=np.float32)
    scores = np.ones((300,), dtype=np.float32)
    labels = np.arange(300, dtype=np.int64) % 10
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "frame009_final_coordinates.npz"
    result = tmp_path / "performance_result.json"
    write_coordinates(reference, boxes, scores, labels)
    write_coordinates(candidate, boxes, scores, labels)
    write_runtime_result(result, strict_gate="FAIL")

    assert final_acceptance(1, result, candidate, reference, tmp_path) == 0
    assert final_acceptance(137, result, candidate, reference, tmp_path) == 1


def test_runtime_exception_cannot_be_corrected_away(tmp_path):
    boxes = np.ones((300, 9), dtype=np.float32)
    scores = np.ones((300,), dtype=np.float32)
    labels = np.arange(300, dtype=np.int64) % 10
    reference = tmp_path / "reference.npz"
    candidate = tmp_path / "frame009_final_coordinates.npz"
    result = tmp_path / "performance_result.json"
    write_coordinates(reference, boxes, scores, labels)
    write_coordinates(candidate, boxes, scores, labels)
    write_runtime_result(result, strict_gate="FAIL")
    data = json.loads(result.read_text(encoding="utf-8"))
    data["exception_type"] = "RuntimeError"
    result.write_text(json.dumps(data), encoding="utf-8")

    assert final_acceptance(1, result, candidate, reference, tmp_path) == 1
