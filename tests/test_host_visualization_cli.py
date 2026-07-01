from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT / "visualization" / "render_host_a.py"
IMPLEMENTATION_PATH = ROOT / "visualization" / "render_host_a_impl.py"

def load_module():
    spec = importlib.util.spec_from_file_location("cli", CLI_PATH)
    assert spec and spec.loader
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_sha256(tmp_path):
    m = load_module()
    p = tmp_path / "v.bin"
    p.write_bytes(b"abc")
    assert m.sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

def test_discover(tmp_path):
    m = load_module()
    v = tmp_path / "visualization"
    v.mkdir()
    for i in range(10):
        (v / f"frame_{i:03d}_camera.png").write_bytes(b"c")
        (v / f"frame_{i:03d}_bev.png").write_bytes(b"b")
    (v / "camera.gif").write_bytes(b"g")
    (v / "bev.gif").write_bytes(b"g")
    (v / "index.html").write_text("<html></html>")
    (v / "results.json").write_text("{}")
    o = m.discover_outputs(tmp_path)
    assert o["camera_png_count"] == 10
    assert o["bev_png_count"] == 10
    assert o["gif_count"] == 2
    assert o["html_count"] == 1

def test_no_hardcoded_paths():
    text = IMPLEMENTATION_PATH.read_text(encoding="utf-8")
    assert "/home/xujunwei/BEVFormer_ai" not in text
    assert "local_bevformer_full_demo_unseen10_fixed_20260630_114535" not in text

def test_shell_exists():
    s = ROOT / "tools" / "run_host_a_visualization.sh"
    assert s.is_file() and (s.stat().st_mode & 0o111)
