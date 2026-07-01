import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("manifest_utils", ROOT / "python" / "utils.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def make_manifest():
    frames = {}
    for index in range(10):
        assets = {"images": {}, "can_bus": {}, "lidar2img": {}, "shift": {}}
        assets["prev_bev_reference_semantic" if index == 0 else "rotation_can_bus"] = {}
        frames[f"sample_{index:03d}"] = {"assets": assets}
    return {"status": "PASS", "frame_indices": list(range(10)), "frames": frames}


def test_valid_manifest():
    MOD.validate_manifest_data(make_manifest())


def test_wrong_frame_range():
    data = make_manifest()
    data["frame_indices"] = [0]
    try:
        MOD.validate_manifest_data(data)
    except ValueError:
        return
    raise AssertionError("invalid frame range was accepted")


def test_missing_rotation_asset():
    data = make_manifest()
    data["frames"]["sample_001"]["assets"].pop("rotation_can_bus")
    try:
        MOD.validate_manifest_data(data)
    except ValueError:
        return
    raise AssertionError("missing temporal asset was accepted")
