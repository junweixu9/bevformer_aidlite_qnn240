import importlib.util
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("temporal_module", root / "python" / "temporal.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_frame_zero_uses_zero_state():
    state = module.TemporalState()
    zero = np.zeros(module.BEV_SHAPE, dtype=np.float32)
    value = state.input_for_frame(0, zero, None)
    assert value.shape == module.BEV_SHAPE
    assert not np.any(value)


def test_missing_live_state_fails():
    state = module.TemporalState()
    try:
        state.input_for_frame(1, np.zeros(module.BEV_SHAPE), np.zeros((1, 18)))
    except RuntimeError:
        return
    raise AssertionError("missing live state was accepted")
