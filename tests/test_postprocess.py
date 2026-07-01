from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "postprocess", ROOT / "python" / "portable_numpy_nmsfreecoder.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

CONTRACT = {
    "num_classes": 10,
    "max_num": 300,
    "num_query": 900,
    "code_size": 10,
    "post_center_range": [-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    "score_threshold": None,
    "selected_sigmoid_mode": "direct",
}


def test_decode_returns_expected_shapes_and_dtypes():
    cls = np.full((1, 900, 10), -20.0, dtype=np.float32)
    bbox = np.zeros((1, 900, 10), dtype=np.float32)
    cls[0, 0, 3] = 20.0
    boxes, scores, labels = MODULE.decode_numpy_nmsfreecoder(cls, bbox, CONTRACT)
    assert boxes.shape == (300, 9)
    assert scores.shape == (300,)
    assert labels.shape == (300,)
    assert boxes.dtype == np.float32
    assert scores.dtype == np.float32
    assert labels.dtype == np.int64
    assert labels[0] == 3


def test_tie_breaking_is_deterministic():
    cls = np.zeros((1, 900, 10), dtype=np.float32)
    bbox = np.zeros((1, 900, 10), dtype=np.float32)
    first = MODULE.decode_numpy_nmsfreecoder(cls, bbox, CONTRACT)
    second = MODULE.decode_numpy_nmsfreecoder(cls, bbox, CONTRACT)
    for left, right in zip(first, second):
        np.testing.assert_array_equal(left, right)


def test_non_finite_logits_are_rejected():
    cls = np.zeros((1, 900, 10), dtype=np.float32)
    bbox = np.zeros((1, 900, 10), dtype=np.float32)
    cls[0, 0, 0] = np.nan
    try:
        MODULE.decode_numpy_nmsfreecoder(cls, bbox, CONTRACT)
    except ValueError as exc:
        assert "non-finite" in str(exc)
    else:
        raise AssertionError("non-finite logits should fail")
