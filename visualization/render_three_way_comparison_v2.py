#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Mapping, NamedTuple

EXPECTED_CONDA_ENV = "open-mmlab"
FRAME_FILE_RE = re.compile(r"^frame_(\d{3})_(.+)_(camera|bev)\.png$")


class FrameKey(NamedTuple):
    index: int
    token: str


def require_directory(name: str, path: Path) -> None:
    if not path.is_dir():
        raise RuntimeError(f"{name} is not a directory: {path}")


def index_assets(root: Path) -> Dict[FrameKey, Dict[str, Path]]:
    assets: Dict[FrameKey, Dict[str, Path]] = {}
    for path in sorted(root.glob("frame_*_*.png")):
        match = FRAME_FILE_RE.match(path.name)
        if not match:
            continue
        key = FrameKey(int(match.group(1)), match.group(2))
        kind = match.group(3)
        bucket = assets.setdefault(key, {})
        if kind in bucket:
            raise RuntimeError(f"Duplicate {kind} asset for {key}: {path}")
        bucket[kind] = path
    return assets


def validate_pairing(
    local_assets: Mapping[FrameKey, Mapping[str, Path]],
    qnn_assets: Mapping[FrameKey, Mapping[str, Path]],
    expected_count: int,
) -> List[FrameKey]:
    if set(local_assets) != set(qnn_assets):
        raise RuntimeError(
            "Local/QNN frame-token mismatch: "
            f"only_local={sorted(set(local_assets) - set(qnn_assets))}, "
            f"only_qnn={sorted(set(qnn_assets) - set(local_assets))}"
        )
    keys = sorted(local_assets)
    if len(keys) != expected_count:
        raise RuntimeError(f"Expected {expected_count} frames, got {len(keys)}")
    if [key.index for key in keys] != list(range(expected_count)):
        raise RuntimeError("Frame indices are not contiguous from zero")
    for key in keys:
        for route, mapping in (("local", local_assets), ("qnn", qnn_assets)):
            missing = {"camera", "bev"}.difference(mapping[key])
            if missing:
                raise RuntimeError(f"Missing {route} assets for {key}: {sorted(missing)}")
    return keys


def split_camera_montage(image):
    if image.width <= 0 or image.height < 2:
        raise RuntimeError(f"Invalid camera image size: {image.size}")
    split_y = image.height // 2
    return (
        image.crop((0, 0, image.width, split_y)),
        image.crop((0, split_y, image.width, image.height)),
    )


def save_camera_routes(local_path: Path, qnn_path: Path, output_dir: Path, stem: str):
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(local_path) as local_source, Image.open(qnn_path) as qnn_source:
        local_pred, gt = split_camera_montage(local_source.convert("RGB"))
        qnn_pred, _ = split_camera_montage(qnn_source.convert("RGB"))
        paths = {
            "gt": output_dir / f"{stem}_gt.png",
            "pytorch": output_dir / f"{stem}_pytorch.png",
            "qnn": output_dir / f"{stem}_qnn240.png",
        }
        gt.save(paths["gt"], format="PNG")
        local_pred.save(paths["pytorch"], format="PNG")
        qnn_pred.save(paths["qnn"], format="PNG")
    return paths


def build_gt_boxes(nusc, sample_token: str):
    import numpy as np
    from nuscenes.eval.detection.data_classes import DetectionBox
    from nuscenes.eval.detection.utils import category_to_detection_name

    boxes = []
    sample = nusc.get("sample", sample_token)
    for annotation_token in sample["anns"]:
        annotation = nusc.get("sample_annotation", annotation_token)
        detection_name = category_to_detection_name(annotation["category_name"])
        if detection_name is None:
            continue
        velocity = np.asarray(nusc.box_velocity(annotation_token)[:2], dtype=np.float64)
        velocity = np.nan_to_num(velocity, nan=0.0)
        boxes.append(
            DetectionBox(
                sample_token=sample_token,
                translation=tuple(annotation["translation"]),
                size=tuple(annotation["size"]),
                rotation=tuple(annotation["rotation"]),
                velocity=tuple(float(value) for value in velocity),
                ego_translation=(0.0, 0.0, 0.0),
                num_pts=-1,
                detection_name=detection_name,
                detection_score=-1.0,
                attribute_name="",
            )
        )
    return boxes


def render_gt_bev(nusc, sample_token: str, output_path: Path) -> None:
    from nuscenes.eval.common.data_classes import EvalBoxes
    from nuscenes.eval.detection.render import visualize_sample

    gt_boxes = EvalBoxes()
    pred_boxes = EvalBoxes()
    gt_boxes.add_boxes(sample_token, build_gt_boxes(nusc, sample_token))
    pred_boxes.add_boxes(sample_token, [])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualize_sample(
        nusc,
        sample_token,
        gt_boxes,
        pred_boxes,
        conf_th=1.0,
        verbose=False,
        savepath=str(output_path),
    )


def copy_bev_routes(
    nusc,
    sample_token: str,
    local_path: Path,
    qnn_path: Path,
    output_dir: Path,
    stem: str,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "gt": output_dir / f"{stem}_gt.png",
        "pytorch": output_dir / f"{stem}_pytorch.png",
        "qnn": output_dir / f"{stem}_qnn240.png",
    }
    render_gt_bev(nusc, sample_token, paths["gt"])
    shutil.copy2(local_path, paths["pytorch"])
    shutil.copy2(qnn_path, paths["qnn"])
    return paths


def relative_map(paths: Mapping[str, Path], root: Path) -> Dict[str, str]:
    return {name: str(path.relative_to(root)) for name, path in paths.items()}


def write_html(vis_dir: Path, frames: List[dict]) -> Path:
    safe_frames = json.dumps(frames, ensure_ascii=False).replace("</", "<\\/")
    frame_buttons = "".join(
        f'<button class="frame-button" data-index="{frame["frame_index"]}">{frame["frame_index"]:03d}</button>'
        for frame in frames
    )
    path = vis_dir / "index.html"
    path.write_text(
        f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BEVFormer 三路对比</title>
<style>
:root {{ --bg:#0b1220; --panel:#121c2e; --line:#2b3a52; --text:#eef4ff; --muted:#9eb0c9; --accent:#65a7ff; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font-family:Inter,Arial,sans-serif; }}
header {{ position:sticky; top:0; z-index:20; background:rgba(11,18,32,.96); border-bottom:1px solid var(--line); backdrop-filter:blur(12px); }}
.header-inner {{ max-width:1920px; margin:auto; padding:14px 22px; }}
h1 {{ margin:0 0 6px; font-size:24px; }}
.subtitle {{ color:var(--muted); font-size:14px; }}
.controls {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:12px; align-items:center; }}
button {{ background:#1a2840; color:var(--text); border:1px solid var(--line); border-radius:7px; padding:8px 12px; cursor:pointer; }}
button:hover, button.active {{ border-color:var(--accent); background:#20385d; }}
main {{ max-width:1920px; margin:auto; padding:22px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; margin-bottom:20px; }}
.frame-title {{ display:flex; justify-content:space-between; gap:16px; align-items:baseline; margin-bottom:14px; }}
.token {{ color:var(--muted); font-family:monospace; overflow-wrap:anywhere; font-size:12px; }}
.section-title {{ margin:18px 0 10px; font-size:18px; }}
.route-tabs {{ display:flex; gap:8px; margin-bottom:10px; }}
.viewer {{ background:#08101d; border:1px solid var(--line); border-radius:9px; overflow:hidden; min-height:260px; }}
.single-view {{ display:flex; justify-content:center; }}
.single-view img {{ max-width:100%; height:auto; display:block; cursor:zoom-in; }}
.compare-strip {{ display:flex; gap:12px; overflow-x:auto; padding:12px; scroll-snap-type:x mandatory; }}
.route-panel {{ flex:1 0 min(760px, 84vw); scroll-snap-align:start; background:#0d1727; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
.route-label {{ padding:9px 12px; font-weight:700; border-bottom:1px solid var(--line); }}
.route-panel img {{ display:block; width:100%; height:auto; cursor:zoom-in; }}
.bev-grid {{ display:grid; grid-template-columns:repeat(3,minmax(480px,1fr)); gap:12px; overflow-x:auto; padding-bottom:5px; }}
.bev-grid .route-panel {{ min-width:480px; }}
.lineage {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:12px; }}
.lineage div {{ border:1px solid var(--line); border-radius:8px; padding:10px; color:var(--muted); }}
.lineage strong {{ color:var(--text); }}
.hidden {{ display:none !important; }}
.modal {{ position:fixed; inset:0; z-index:100; background:rgba(0,0,0,.94); display:flex; align-items:center; justify-content:center; padding:20px; }}
.modal img {{ max-width:98vw; max-height:94vh; object-fit:contain; }}
.modal-close {{ position:fixed; top:16px; right:18px; font-size:20px; }}
@media(max-width:900px) {{ .lineage {{ grid-template-columns:1fr; }} main {{ padding:10px; }} .card {{ padding:10px; }} }}
</style>
</head>
<body>
<header><div class="header-inner">
  <h1>BEVFormer：GT / Local PyTorch Golden / QNN2.40</h1>
  <div class="subtitle">同一 sample token、同一帧、同一视角。Camera 可切换原图或横向并排；BEV 固定三列。点击任意图片可全屏查看。</div>
  <div class="controls">
    <button id="prev">上一帧</button><button id="play">播放</button><button id="next">下一帧</button>
    {frame_buttons}
  </div>
</div></header>
<main>
  <section class="card">
    <div class="frame-title"><h2 id="frameHeading"></h2><div class="token" id="token"></div></div>
    <div class="lineage">
      <div><strong>GT</strong><br>nuScenes 标注，任务真值。</div>
      <div><strong>Local PyTorch Golden</strong><br>原始模型算法级参考。</div>
      <div><strong>QNN2.40 / QCS8550</strong><br>板端部署预测；NMSFreeCoder 为 Host NumPy。</div>
    </div>

    <h3 class="section-title">Camera</h3>
    <div class="route-tabs">
      <button data-mode="compare" class="mode-button active">三路并排</button>
      <button data-mode="gt" class="mode-button">GT</button>
      <button data-mode="pytorch" class="mode-button">PyTorch</button>
      <button data-mode="qnn" class="mode-button">QNN2.40</button>
    </div>
    <div id="cameraCompare" class="viewer compare-strip"></div>
    <div id="cameraSingle" class="viewer single-view hidden"></div>

    <h3 class="section-title">BEV</h3>
    <div id="bevGrid" class="bev-grid"></div>
    <p class="subtitle">BEV 中绿色为 GT、蓝色为预测；点云仅用于可视化背景，不参与 camera-only 模型推理。</p>
  </section>
</main>
<div id="modal" class="modal hidden"><button class="modal-close" id="modalClose">关闭</button><img id="modalImage" alt="full size"></div>
<script>
const frames = {safe_frames};
const routeLabels = {{gt:'GT', pytorch:'Local PyTorch Golden', qnn:'QNN2.40 / QCS8550'}};
let frameIndex = 0;
let mode = 'compare';
let timer = null;

function panel(label, src) {{
  return `<div class="route-panel"><div class="route-label">${{label}}</div><img src="${{src}}" data-full="${{src}}" alt="${{label}}"></div>`;
}}
function render() {{
  const f = frames[frameIndex];
  document.getElementById('frameHeading').textContent = `Frame ${{String(f.frame_index).padStart(3,'0')}}`;
  document.getElementById('token').textContent = `sample token: ${{f.sample_token}}`;
  document.querySelectorAll('.frame-button').forEach((b,i)=>b.classList.toggle('active',i===frameIndex));
  const compare = document.getElementById('cameraCompare');
  const single = document.getElementById('cameraSingle');
  if (mode === 'compare') {{
    compare.classList.remove('hidden'); single.classList.add('hidden');
    compare.innerHTML = ['gt','pytorch','qnn'].map(r=>panel(routeLabels[r],f.camera[r])).join('');
  }} else {{
    compare.classList.add('hidden'); single.classList.remove('hidden');
    single.innerHTML = `<img src="${{f.camera[mode]}}" data-full="${{f.camera[mode]}}" alt="${{routeLabels[mode]}}">`;
  }}
  document.getElementById('bevGrid').innerHTML = ['gt','pytorch','qnn'].map(r=>panel(routeLabels[r],f.bev[r])).join('');
}}
function setFrame(next) {{ frameIndex = (next + frames.length) % frames.length; render(); }}
function stop() {{ if (timer) clearInterval(timer); timer=null; document.getElementById('play').textContent='播放'; }}
document.getElementById('prev').onclick=()=>{{stop();setFrame(frameIndex-1)}};
document.getElementById('next').onclick=()=>{{stop();setFrame(frameIndex+1)}};
document.getElementById('play').onclick=()=>{{ if(timer){{stop();return;}} document.getElementById('play').textContent='暂停'; timer=setInterval(()=>setFrame(frameIndex+1),1200); }};
document.querySelectorAll('.frame-button').forEach((b,i)=>b.onclick=()=>{{stop();setFrame(i)}});
document.querySelectorAll('.mode-button').forEach(b=>b.onclick=()=>{{ mode=b.dataset.mode; document.querySelectorAll('.mode-button').forEach(x=>x.classList.toggle('active',x===b)); render(); }});
document.addEventListener('click',e=>{{ if(e.target.matches('img[data-full]')){{ document.getElementById('modalImage').src=e.target.dataset.full; document.getElementById('modal').classList.remove('hidden'); }} }});
document.getElementById('modalClose').onclick=()=>document.getElementById('modal').classList.add('hidden');
document.getElementById('modal').onclick=e=>{{if(e.target.id==='modal')e.currentTarget.classList.add('hidden')}};
document.addEventListener('keydown',e=>{{ if(e.key==='ArrowLeft')setFrame(frameIndex-1); if(e.key==='ArrowRight')setFrame(frameIndex+1); if(e.key==='Escape')document.getElementById('modal').classList.add('hidden'); }});
render();
</script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build an interactive three-way BEVFormer comparison page.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--local-visualization-dir", required=True)
    parser.add_argument("--qnn-visualization-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-frame-count", type=int, default=10)
    parser.add_argument("--skip-conda-check", action="store_true")
    args = parser.parse_args()

    if not args.skip_conda_check and os.environ.get("CONDA_DEFAULT_ENV") != EXPECTED_CONDA_ENV:
        raise RuntimeError(f"Requires conda env {EXPECTED_CONDA_ENV!r}")

    source_root = Path(args.source_root).expanduser().resolve()
    local_dir = Path(args.local_visualization_dir).expanduser().resolve()
    qnn_dir = Path(args.qnn_visualization_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    for name, path in (("source_root", source_root), ("local_visualization_dir", local_dir), ("qnn_visualization_dir", qnn_dir)):
        require_directory(name, path)
    data_root = source_root / "data" / "nuscenes"
    require_directory("nuscenes_data_root", data_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output dir not empty: {output_dir}")

    local_assets = index_assets(local_dir)
    qnn_assets = index_assets(qnn_dir)
    keys = validate_pairing(local_assets, qnn_assets, args.expected_frame_count)
    print(f"FRAME_TOKEN_ALIGNMENT_GATE=PASS COUNT={len(keys)}")

    os.environ.setdefault("MPLBACKEND", "Agg")
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version="v1.0-trainval", dataroot=str(data_root), verbose=False)

    vis_dir = output_dir / "visualization"
    camera_dir = vis_dir / "assets" / "camera"
    bev_dir = vis_dir / "assets" / "bev"
    frames = []
    for key in keys:
        stem = f"frame_{key.index:03d}_{key.token}"
        camera_paths = save_camera_routes(local_assets[key]["camera"], qnn_assets[key]["camera"], camera_dir, stem)
        bev_paths = copy_bev_routes(nusc, key.token, local_assets[key]["bev"], qnn_assets[key]["bev"], bev_dir, stem)
        frames.append({
            "frame_index": key.index,
            "sample_token": key.token,
            "camera": relative_map(camera_paths, vis_dir),
            "bev": relative_map(bev_paths, vis_dir),
        })
        print(f"FRAME_{key.index:03d}_ASSET_GATE=PASS TOKEN={key.token}")

    html_path = write_html(vis_dir, frames)
    manifest_path = vis_dir / "three_way_manifest.json"
    manifest_path.write_text(json.dumps({"frames": frames}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    camera_count = len(list(camera_dir.glob("*.png")))
    bev_count = len(list(bev_dir.glob("*.png")))
    final_gate = (
        camera_count == args.expected_frame_count * 3
        and bev_count == args.expected_frame_count * 3
        and html_path.is_file()
        and manifest_path.is_file()
    )
    report = {
        "audit_type": "GT_PYTORCH_QNN240_THREE_WAY_VISUALIZATION_V2",
        "frame_count": len(frames),
        "camera_asset_count": camera_count,
        "bev_asset_count": bev_count,
        "html": str(html_path),
        "manifest": str(manifest_path),
        "layout": "interactive native-resolution routes with full-screen modal",
        "final_gate": "PASS" if final_gate else "FAIL",
    }
    (output_dir / "three_way_visualization_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (output_dir / "three_way_visualization_report.txt").write_text(
        "\n".join([
            "AUDIT_TYPE=GT_PYTORCH_QNN240_THREE_WAY_VISUALIZATION_V2",
            f"FRAME_COUNT={len(frames)}",
            f"CAMERA_ASSET_COUNT={camera_count}",
            f"BEV_ASSET_COUNT={bev_count}",
            f"HTML={html_path}",
            f"FINAL_THREE_WAY_VISUALIZATION_GATE={'PASS' if final_gate else 'FAIL'}",
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"CAMERA_ASSET_COUNT={camera_count}")
    print(f"BEV_ASSET_COUNT={bev_count}")
    print(f"HTML={html_path}")
    print(f"FINAL_THREE_WAY_VISUALIZATION_GATE={'PASS' if final_gate else 'FAIL'}")
    return 0 if final_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
