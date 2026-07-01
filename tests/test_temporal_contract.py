import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "temporal_module", ROOT / "python" / "temporal.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_frame_zero_uses_zero_state():
    state = MODULE.TemporalState()
    zero = np.zeros(MODULE.BEV_SHAPE, dtype=np.float32)
    value = state.input_for_frame(0, zero, None)
    assert value.shape == MODULE.BEV_SHAPE
    assert value.dtype == np.float32
    assert np.count_nonzero(value) == 0


def test_nonzero_scene_start_fails():
    state = MODULE.TemporalState()
    invalid = np.ones(MODULE.BEV_SHAPE, dtype=np.float32)
    try:
        state.input_for_frame(0, invalid, None)
    except ValueError:
        return
    raise AssertionError("nonzero scene-start prev_bev was accepted")


def test_missing_live_state_fails():
    state = MODULE.TemporalState()
    rotation = np.zeros((1, 18), dtype=np.float32)
    try:
        state.input_for_frame(1, np.zeros(MODULE.BEV_SHAPE), rotation)
    except RuntimeError:
        return
    raise AssertionError("frame 001 was accepted without live previous_bev")


def test_zero_degree_rotation_matches_fp16_round_trip():
    values = np.linspace(-2.0, 2.0, np.prod(MODULE.BEV_SHAPE), dtype=np.float32)
    previous = values.reshape(MODULE.BEV_SHAPE)
    rotation = np.zeros((1, 18), dtype=np.float32)
    rotated = MODULE.rotate_prev_bev(previous, rotation)
    expected = previous.astype("<f2").astype(np.float32)
    np.testing.assert_array_equal(rotated, expected)
