from pathlib import Path
import sys

import numpy as np

PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(PYTHON_DIR))

import temporal


def test_scene_start_uses_zero_prev_bev():
    state = temporal.TemporalState()
    zero = np.zeros(temporal.BEV_SHAPE, dtype=np.float32)
    value = state.input_for_frame(0, zero, None)
    assert value.shape == temporal.BEV_SHAPE
    assert value.dtype == np.float32
    assert np.count_nonzero(value) == 0


def test_nonzero_scene_start_is_rejected():
    state = temporal.TemporalState()
    invalid = np.ones(temporal.BEV_SHAPE, dtype=np.float32)
    try:
        state.input_for_frame(0, invalid, None)
    except ValueError:
        return
    raise AssertionError("nonzero scene-start prev_bev was accepted")


def test_later_frame_requires_live_previous_bev():
    state = temporal.TemporalState()
    zero = np.zeros(temporal.BEV_SHAPE, dtype=np.float32)
    rotation = np.zeros((1, 18), dtype=np.float32)
    try:
        state.input_for_frame(1, zero, rotation)
    except RuntimeError:
        return
    raise AssertionError("frame 001 was accepted without a live previous bev")
