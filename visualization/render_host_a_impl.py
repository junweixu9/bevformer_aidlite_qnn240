# Repository-adapted Container-A renderer.
# Derived from the validated local Golden Renderer.
# Runtime paths are injected by render_host_a.py.
import glob
import html
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
from nuscenes.nuscenes import NuScenes
from PIL import Image

ROOT = Path(os.environ["BEVFORMER_SOURCE_ROOT"]).expanduser().resolve()
CFG = ROOT / "projects/configs/bevformer/bevformer_tiny_unseen10_20260623_203417.py"
RESULT = Path(os.environ["BEVFORMER_RESULT_PKL"]).expanduser().resolve()
OUT = Path(os.environ["BEVFORMER_VIS_OUTPUT_DIR"]).expanduser().resolve()
VISDIR = OUT / "visualization"
JSON_PREFIX = VISDIR / "formatted_result"
VISUAL_PY = ROOT / "tools/analysis_tools/visual.py"
DATA_ROOT = ROOT / "data/nuscenes"

sys.path.insert(0, str(ROOT))
VISDIR.mkdir(parents=True, exist_ok=True)

# 注册 BEVFormer 自定义模块。
import projects.mmdet3d_plugin  # noqa: F401,E402


def find_json_path(obj):
    if isinstance(obj, str) and obj.endswith(".json"):
        return obj
    if isinstance(obj, dict):
        for value in obj.values():
            found = find_json_path(value)
            if found:
                return found
    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = find_json_path(value)
            if found:
                return found
    return None


print("============================================================")
print("AUDIT_TYPE=BEVFORMER_UNSEEN10_VISUALIZATION")
print("CONFIG=" + str(CFG))
print("RESULT=" + str(RESULT))
print("VISDIR=" + str(VISDIR))

# 1. 将 pkl 结果转换为 NuScenes 标准 JSON。
cfg = Config.fromfile(str(CFG))
dataset = build_dataset(cfg.data.test)
outputs = mmcv.load(str(RESULT))

if len(outputs) != len(dataset):
    raise RuntimeError(
        "Result/dataset length mismatch: "
        f"results={len(outputs)}, dataset={len(dataset)}"
    )

formatted = dataset.format_results(
    outputs,
    jsonfile_prefix=str(JSON_PREFIX),
)

result_files = formatted[0] if isinstance(formatted, tuple) else formatted
json_path = find_json_path(result_files)

if not json_path or not Path(json_path).is_file():
    raise RuntimeError(f"Cannot locate results_nusc.json in: {result_files!r}")

print("NUSCENES_JSON=" + str(json_path))

# 2. 动态加载 BEVFormer 官方可视化脚本。
spec = importlib.util.spec_from_file_location(
    "bevformer_visual",
    str(VISUAL_PY),
)
visual = importlib.util.module_from_spec(spec)
spec.loader.exec_module(visual)

visual.nusc = NuScenes(
    version="v1.0-trainval",
    dataroot=str(DATA_ROOT),
    verbose=False,
)

pred_data = mmcv.load(str(json_path))
sample_tokens = list(pred_data["results"].keys())

print("SAMPLE_TOKEN_COUNT=" + str(len(sample_tokens)))

# 3. 为每帧生成六相机投影图和 BEV 图。
for frame_index, sample_token in enumerate(sample_tokens):
    output_base = VISDIR / f"frame_{frame_index:03d}_{sample_token}"

    visual.render_sample_data(
        sample_token,
        pred_data=pred_data,
        out_path=str(output_base),
        verbose=False,
    )

    print(
        f"FRAME_{frame_index:03d}_VISUALIZATION_GATE=PASS "
        f"TOKEN={sample_token}"
    )


def collect_images(marker):
    paths = []
    for path in sorted(glob.glob(str(VISDIR / f"frame_*{marker}*"))):
        if not os.path.isfile(path):
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            paths.append(path)
        except Exception:
            continue
    return paths


def make_gif(image_paths, output_path, max_width=1600, duration_ms=800):
    if not image_paths:
        return False

    frames = []
    resampling = getattr(Image, "Resampling", Image).LANCZOS

    for path in image_paths:
        with Image.open(path) as source:
            frame = source.convert("RGB")

            if frame.width > max_width:
                height = round(frame.height * max_width / frame.width)
                frame = frame.resize((max_width, height), resampling)

            frames.append(frame.copy())

    frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    return True


camera_images = collect_images("_camera")
bev_images = collect_images("_bev")

camera_gif = VISDIR / "bevformer_10frames_camera.gif"
bev_gif = VISDIR / "bevformer_10frames_bev.gif"

camera_gif_ok = make_gif(camera_images, camera_gif)
bev_gif_ok = make_gif(bev_images, bev_gif)

# 4. 生成浏览器可直接打开的 HTML 页面。
html_path = VISDIR / "index.html"

with html_path.open("w", encoding="utf-8") as file:
    file.write(
        """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>BEVFormer 十帧可视化</title>
<style>
body { font-family: Arial, sans-serif; margin: 24px; background: #eee; }
h1, h2 { color: #222; }
section { background: white; padding: 16px; margin-bottom: 20px; }
img { max-width: 100%; height: auto; border: 1px solid #bbb; }
.grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.frame { background: white; padding: 14px; margin: 16px 0; }
</style>
</head>
<body>
<h1>BEVFormer Tiny 连续十帧检测可视化</h1>
<p>Camera：六路相机预测框与 Ground Truth。</p>
<p>BEV：鸟瞰图预测框与 Ground Truth。</p>
"""
    )

    if camera_gif_ok:
        file.write(
            '<section><h2>六路相机连续结果</h2>'
            '<img src="bevformer_10frames_camera.gif"></section>'
        )

    if bev_gif_ok:
        file.write(
            '<section><h2>BEV 连续结果</h2>'
            '<img src="bevformer_10frames_bev.gif"></section>'
        )

    for index in range(max(len(camera_images), len(bev_images))):
        file.write(f'<div class="frame"><h2>Frame {index:03d}</h2><div class="grid">')

        if index < len(camera_images):
            name = html.escape(Path(camera_images[index]).name)
            file.write(f'<div><h3>Camera</h3><img src="{name}"></div>')

        if index < len(bev_images):
            name = html.escape(Path(bev_images[index]).name)
            file.write(f'<div><h3>BEV</h3><img src="{name}"></div>')

        file.write("</div></div>")

    file.write("</body></html>")

print("CAMERA_IMAGE_COUNT=" + str(len(camera_images)))
print("BEV_IMAGE_COUNT=" + str(len(bev_images)))
print("CAMERA_GIF_GATE=" + ("PASS" if camera_gif_ok else "FAIL"))
print("BEV_GIF_GATE=" + ("PASS" if bev_gif_ok else "FAIL"))
print("HTML_GATE=" + ("PASS" if html_path.is_file() else "FAIL"))
print("HTML=" + str(html_path))

passed = (
    len(sample_tokens) == 10
    and len(camera_images) == 10
    and len(bev_images) == 10
    and camera_gif_ok
    and bev_gif_ok
    and html_path.is_file()
)

print(
    "BEVFORMER_VISUALIZATION_GATE="
    + ("PASS" if passed else "FAIL")
)
print("============================================================")

raise SystemExit(0 if passed else 2)
