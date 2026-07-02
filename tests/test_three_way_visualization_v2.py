from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "visualization" / "render_three_way_comparison_v2.py"


def load_module():
    spec = importlib.util.spec_from_file_location("three_way_v2", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_split_camera_routes(tmp_path):
    module = load_module()
    source = Image.new("RGB", (100, 80), "red")
    for y in range(40, 80):
        for x in range(100):
            source.putpixel((x, y), (0, 255, 0))
    pred, gt = module.split_camera_montage(source)
    assert pred.size == (100, 40)
    assert gt.size == (100, 40)
    assert pred.getpixel((2, 2)) == (255, 0, 0)
    assert gt.getpixel((2, 2)) == (0, 255, 0)


def test_index_and_pairing(tmp_path):
    module = load_module()
    local = tmp_path / "local"
    qnn = tmp_path / "qnn"
    local.mkdir()
    qnn.mkdir()
    for index in range(2):
        token = f"token{index}"
        for root in (local, qnn):
            (root / f"frame_{index:03d}_{token}_camera.png").write_bytes(b"c")
            (root / f"frame_{index:03d}_{token}_bev.png").write_bytes(b"b")
    keys = module.validate_pairing(module.index_assets(local), module.index_assets(qnn), 2)
    assert [(key.index, key.token) for key in keys] == [(0, "token0"), (1, "token1")]


def test_html_has_interactive_controls(tmp_path):
    module = load_module()
    vis = tmp_path / "visualization"
    vis.mkdir()
    frames = [{
        "frame_index": 0,
        "sample_token": "abc",
        "camera": {"gt": "a.png", "pytorch": "b.png", "qnn": "c.png"},
        "bev": {"gt": "d.png", "pytorch": "e.png", "qnn": "f.png"},
    }]
    path = module.write_html(vis, frames)
    text = path.read_text(encoding="utf-8")
    assert "三路并排" in text
    assert "data-mode=\"pytorch\"" in text
    assert "modalImage" in text
    assert "ArrowLeft" in text and "ArrowRight" in text


def test_runner_uses_v2():
    runner = (ROOT / "tools" / "run_three_way_visualization.sh").read_text(encoding="utf-8")
    assert "render_three_way_comparison_v2.py" in runner
