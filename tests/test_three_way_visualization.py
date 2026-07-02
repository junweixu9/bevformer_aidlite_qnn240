from __future__ import annotations

import importlib.util
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "visualization" / "render_three_way_comparison.py"


def load_module():
    spec = importlib.util.spec_from_file_location("three_way", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_asset_index_and_pairing(tmp_path):
    module = load_module()
    local = tmp_path / "local"
    qnn = tmp_path / "qnn"
    local.mkdir()
    qnn.mkdir()
    for index in range(2):
        token = f"token{index}"
        for root in (local, qnn):
            (root / f"frame_{index:03d}_{token}_camera.png").write_bytes(b"camera")
            (root / f"frame_{index:03d}_{token}_bev.png").write_bytes(b"bev")

    local_assets = module.index_visualization_assets(local)
    qnn_assets = module.index_visualization_assets(qnn)
    keys = module.validate_asset_pairing(local_assets, qnn_assets, 2)

    assert [key.index for key in keys] == [0, 1]
    assert [key.token for key in keys] == ["token0", "token1"]


def test_camera_split_and_composition(tmp_path):
    module = load_module()
    source = Image.new("RGB", (120, 80), "red")
    for y in range(40, 80):
        for x in range(120):
            source.putpixel((x, y), (0, 255, 0))

    prediction, gt = module.split_camera_montage(source)
    assert prediction.size == (120, 40)
    assert gt.size == (120, 40)
    assert prediction.getpixel((1, 1)) == (255, 0, 0)
    assert gt.getpixel((1, 1)) == (0, 255, 0)

    local_path = tmp_path / "local.png"
    qnn_path = tmp_path / "qnn.png"
    output_path = tmp_path / "comparison.png"
    source.save(local_path)
    source.save(qnn_path)
    module.compose_camera_comparison(local_path, qnn_path, output_path)

    assert output_path.is_file()
    with Image.open(output_path) as output:
        assert output.width >= 120
        assert output.height > 3 * 40


def test_bev_composition_and_discovery(tmp_path):
    module = load_module()
    gt = tmp_path / "gt.png"
    local = tmp_path / "local.png"
    qnn = tmp_path / "qnn.png"
    Image.new("RGB", (90, 90), "green").save(gt)
    Image.new("RGB", (90, 90), "blue").save(local)
    Image.new("RGB", (90, 90), "navy").save(qnn)

    visualization = tmp_path / "visualization"
    visualization.mkdir()
    output = visualization / "frame_000_token_three_way_bev.png"
    module.compose_bev_comparison(gt, local, qnn, output)
    (visualization / "frame_000_token_three_way_camera.png").write_bytes(b"camera")
    (visualization / "camera.gif").write_bytes(b"gif")
    (visualization / "bev.gif").write_bytes(b"gif")
    (visualization / "index.html").write_text("<html></html>", encoding="utf-8")
    gt_dir = visualization / "gt_bev"
    gt_dir.mkdir()
    (gt_dir / "frame_000_token_gt_only_bev.png").write_bytes(b"gt")

    outputs = module.discover_outputs(visualization)
    assert outputs == {
        "camera_comparison_count": 1,
        "bev_comparison_count": 1,
        "gif_count": 2,
        "html_count": 1,
        "gt_bev_count": 1,
    }


def test_repository_contracts():
    text = SCRIPT.read_text(encoding="utf-8")
    assert "/home/xujunwei/BEVFormer_ai" not in text
    assert (ROOT / "tools" / "run_three_way_visualization.sh").is_file()
    assert (ROOT / "tools" / "serve_latest_three_way_visualization.sh").is_file()
    assert (ROOT / "tools" / "three_way_visualization.env.example").is_file()
