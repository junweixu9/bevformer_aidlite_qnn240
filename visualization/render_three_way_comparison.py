#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, NamedTuple, Tuple

EXPECTED_CONDA_ENV = "open-mmlab"
FRAME_FILE_RE = re.compile(r"^frame_(\d{3})_(.+)_(camera|bev)\.png$")


class FrameKey(NamedTuple):
    index: int
    token: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_directory(name: str, path: Path) -> None:
    if not path.is_dir():
        raise RuntimeError(f"{name} is not a directory: {path}")


def index_visualization_assets(root: Path) -> Dict[FrameKey, Dict[str, Path]]:
    assets: Dict[FrameKey, Dict[str, Path]] = {}
    for path in sorted(root.glob("frame_*_*.png")):
        match = FRAME_FILE_RE.match(path.name)
        if not match:
            continue
        key = FrameKey(index=int(match.group(1)), token=match.group(2))
        kind = match.group(3)
        bucket = assets.setdefault(key, {})
        if kind in bucket:
            raise RuntimeError(f"Duplicate {kind} asset for frame {key.index}: {path}")
        bucket[kind] = path
    return assets


def validate_asset_pairing(
    local_assets: Mapping[FrameKey, Mapping[str, Path]],
    qnn_assets: Mapping[FrameKey, Mapping[str, Path]],
    expected_frame_count: int,
) -> List[FrameKey]:
    local_keys = set(local_assets)
    qnn_keys = set(qnn_assets)
    if local_keys != qnn_keys:
        only_local = sorted(local_keys - qnn_keys)
        only_qnn = sorted(qnn_keys - local_keys)
        raise RuntimeError(
            "Local/QNN frame-token mismatch: "
            f"only_local={only_local}, only_qnn={only_qnn}"
        )

    keys = sorted(local_keys)
    if len(keys) != expected_frame_count:
        raise RuntimeError(
            f"Expected {expected_frame_count} paired frames, got {len(keys)}"
        )

    expected_indices = list(range(expected_frame_count))
    actual_indices = [key.index for key in keys]
    if actual_indices != expected_indices:
        raise RuntimeError(
            f"Frame indices must be contiguous {expected_indices}, got {actual_indices}"
        )

    for key in keys:
        for route_name, route_assets in (
            ("local", local_assets),
            ("qnn", qnn_assets),
        ):
            missing = {"camera", "bev"}.difference(route_assets[key])
            if missing:
                raise RuntimeError(
                    f"Missing {route_name} assets for frame {key.index}: {sorted(missing)}"
                )
    return keys


def split_camera_montage(image):
    if image.width <= 0 or image.height < 2:
        raise RuntimeError(f"Invalid camera montage size: {image.size}")
    split_y = image.height // 2
    prediction = image.crop((0, 0, image.width, split_y))
    ground_truth = image.crop((0, split_y, image.width, image.height))
    return prediction, ground_truth


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "Arial Bold.ttf" if bold else "Arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def resize_to_width(image, width: int):
    from PIL import Image

    if width <= 0:
        raise ValueError("width must be positive")
    if image.width == width:
        return image.copy()
    height = max(1, round(image.height * width / image.width))
    resampling = getattr(Image, "Resampling", Image).LANCZOS
    return image.resize((width, height), resampling)


def panel_with_title(image, title: str, width: int, title_height: int = 58):
    from PIL import Image, ImageDraw

    resized = resize_to_width(image, width)
    canvas = Image.new("RGB", (width, title_height + resized.height), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(25, bold=True)
    box = draw.textbbox((0, 0), title, font=font)
    text_width = box[2] - box[0]
    text_height = box[3] - box[1]
    draw.text(
        ((width - text_width) // 2, (title_height - text_height) // 2 - box[1]),
        title,
        fill="black",
        font=font,
    )
    canvas.paste(resized, (0, title_height))
    draw.rectangle((0, 0, width - 1, canvas.height - 1), outline=(180, 180, 180), width=2)
    return canvas


def stack_vertical(panels: Iterable, gap: int = 22, margin: int = 24):
    from PIL import Image

    panel_list = list(panels)
    if not panel_list:
        raise ValueError("At least one panel is required")
    width = max(panel.width for panel in panel_list) + 2 * margin
    height = sum(panel.height for panel in panel_list) + gap * (len(panel_list) - 1) + 2 * margin
    canvas = Image.new("RGB", (width, height), (242, 244, 247))
    y = margin
    for panel in panel_list:
        x = (width - panel.width) // 2
        canvas.paste(panel, (x, y))
        y += panel.height + gap
    return canvas


def stack_horizontal(panels: Iterable, gap: int = 18, margin: int = 24):
    from PIL import Image

    panel_list = list(panels)
    if not panel_list:
        raise ValueError("At least one panel is required")
    width = sum(panel.width for panel in panel_list) + gap * (len(panel_list) - 1) + 2 * margin
    height = max(panel.height for panel in panel_list) + 2 * margin
    canvas = Image.new("RGB", (width, height), (242, 244, 247))
    x = margin
    for panel in panel_list:
        y = (height - panel.height) // 2
        canvas.paste(panel, (x, y))
        x += panel.width + gap
    return canvas


def compose_camera_comparison(local_camera: Path, qnn_camera: Path, output_path: Path) -> None:
    from PIL import Image

    with Image.open(local_camera) as local_source, Image.open(qnn_camera) as qnn_source:
        local_prediction, local_gt = split_camera_montage(local_source.convert("RGB"))
        qnn_prediction, _ = split_camera_montage(qnn_source.convert("RGB"))

        target_width = min(1800, max(local_gt.width, local_prediction.width, qnn_prediction.width))
        panels = [
            panel_with_title(local_gt, "Ground Truth (nuScenes annotations)", target_width),
            panel_with_title(local_prediction, "Local PyTorch Golden (prediction)", target_width),
            panel_with_title(qnn_prediction, "QNN2.40 on QCS8550 (prediction)", target_width),
        ]
        comparison = stack_vertical(panels)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(output_path, format="PNG")


def compose_bev_comparison(
    gt_bev: Path,
    local_bev: Path,
    qnn_bev: Path,
    output_path: Path,
) -> None:
    from PIL import Image

    with Image.open(gt_bev) as gt_source, Image.open(local_bev) as local_source, Image.open(qnn_bev) as qnn_source:
        panel_width = 680
        panels = [
            panel_with_title(gt_source.convert("RGB"), "GT only (green)", panel_width),
            panel_with_title(
                local_source.convert("RGB"),
                "Local PyTorch (blue) + GT (green)",
                panel_width,
            ),
            panel_with_title(
                qnn_source.convert("RGB"),
                "QNN2.40 (blue) + GT (green)",
                panel_width,
            ),
        ]
        comparison = stack_horizontal(panels)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.save(output_path, format="PNG")


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


def render_gt_only_bev(nusc, sample_token: str, output_path: Path) -> None:
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


def make_gif(image_paths: Iterable[Path], output_path: Path, max_width: int = 1800, duration_ms: int = 900) -> None:
    from PIL import Image

    frames = []
    for path in image_paths:
        with Image.open(path) as source:
            frame = source.convert("RGB")
            if frame.width > max_width:
                frame = resize_to_width(frame, max_width)
            frames.append(frame.copy())
    if not frames:
        raise RuntimeError("No frames available for GIF")
    target_size = frames[0].size
    normalized = [frame if frame.size == target_size else frame.resize(target_size) for frame in frames]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=normalized[1:],
        duration=duration_ms,
        loop=0,
    )


def write_html(visualization_dir: Path, frames: List[Mapping[str, object]]) -> Path:
    html_path = visualization_dir / "index.html"
    sections = []
    for frame in frames:
        index = int(frame["frame_index"])
        token = html.escape(str(frame["sample_token"]))
        camera_name = html.escape(str(frame["camera_comparison"]))
        bev_name = html.escape(str(frame["bev_comparison"]))
        sections.append(
            f"""
<section class="frame">
  <h2>Frame {index:03d}</h2>
  <p class="token">sample token: {token}</p>
  <h3>Camera comparison</h3>
  <img src="{camera_name}" alt="Frame {index:03d} camera comparison">
  <h3>BEV comparison</h3>
  <img src="{bev_name}" alt="Frame {index:03d} BEV comparison">
</section>
"""
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BEVFormer three-way comparison</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin: 0; background: #eef1f5; color: #172033; font-family: Arial, sans-serif; }}
main {{ width: min(1800px, calc(100% - 32px)); margin: 24px auto 48px; }}
header, section {{ background: white; border: 1px solid #d8dee8; border-radius: 10px; box-shadow: 0 2px 10px rgba(30, 45, 70, .06); }}
header {{ padding: 22px 26px; margin-bottom: 22px; }}
.frame {{ padding: 20px; margin: 20px 0; }}
h1, h2, h3 {{ margin-top: 0; }}
.legend {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; margin-top: 18px; }}
.legend div {{ padding: 12px; background: #f5f7fa; border-radius: 7px; }}
img {{ display: block; width: 100%; height: auto; border: 1px solid #d5dae3; border-radius: 6px; margin-bottom: 24px; }}
.token {{ color: #667085; font-family: monospace; overflow-wrap: anywhere; }}
.note {{ color: #4b5565; line-height: 1.55; }}
@media (max-width: 900px) {{ .legend {{ grid-template-columns: 1fr; }} main {{ width: min(100% - 16px, 1800px); }} }}
</style>
</head>
<body>
<main>
<header>
  <h1>BEVFormer: GT vs Local PyTorch Golden vs QNN2.40</h1>
  <p class="note">Camera panels show GT and the two prediction routes separately. BEV panels show GT-only, then each prediction in blue over the same GT in green. LiDAR points are visualization background only and are not model input.</p>
  <div class="legend">
    <div><strong>GT</strong><br>nuScenes annotations; task truth.</div>
    <div><strong>Local PyTorch Golden</strong><br>Original algorithm-level reference.</div>
    <div><strong>QNN2.40 / QCS8550</strong><br>Deployed board prediction.</div>
  </div>
  <h2>Ten-frame overview</h2>
  <h3>Camera</h3>
  <img src="bevformer_three_way_camera.gif" alt="Three-way camera GIF">
  <h3>BEV</h3>
  <img src="bevformer_three_way_bev.gif" alt="Three-way BEV GIF">
</header>
{''.join(sections)}
</main>
</body>
</html>
"""
    html_path.write_text(document, encoding="utf-8")
    return html_path


def discover_outputs(visualization_dir: Path) -> Dict[str, int]:
    return {
        "camera_comparison_count": len(list(visualization_dir.glob("*_three_way_camera.png"))),
        "bev_comparison_count": len(list(visualization_dir.glob("*_three_way_bev.png"))),
        "gif_count": len(list(visualization_dir.glob("*.gif"))),
        "html_count": len(list(visualization_dir.glob("*.html"))),
        "gt_bev_count": len(list((visualization_dir / "gt_bev").glob("*_gt_only_bev.png"))),
    }


def write_reports(output_dir: Path, report: Mapping[str, object]) -> Tuple[Path, Path]:
    json_path = output_dir / "three_way_visualization_report.json"
    text_path = output_dir / "three_way_visualization_report.txt"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    outputs = report["outputs"]
    lines = [
        "AUDIT_TYPE=GT_PYTORCH_QNN240_THREE_WAY_VISUALIZATION",
        f"FRAME_COUNT={report['frame_count']}",
        f"CAMERA_COMPARISON_COUNT={outputs['camera_comparison_count']}",
        f"BEV_COMPARISON_COUNT={outputs['bev_comparison_count']}",
        f"GT_BEV_COUNT={outputs['gt_bev_count']}",
        f"GIF_COUNT={outputs['gif_count']}",
        f"HTML_COUNT={outputs['html_count']}",
        f"FRAME_TOKEN_ALIGNMENT_GATE={report['frame_token_alignment_gate']}",
        f"FINAL_THREE_WAY_VISUALIZATION_GATE={report['final_gate']}",
    ]
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, text_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compose GT, Local PyTorch Golden and QNN2.40 visualizations."
    )
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--local-visualization-dir", required=True)
    parser.add_argument("--qnn-visualization-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-frame-count", type=int, default=10)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--skip-conda-check", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    local_dir = Path(args.local_visualization_dir).expanduser().resolve()
    qnn_dir = Path(args.qnn_visualization_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not args.skip_conda_check and os.environ.get("CONDA_DEFAULT_ENV") != EXPECTED_CONDA_ENV:
        raise RuntimeError(f"Requires conda env {EXPECTED_CONDA_ENV!r}")

    require_directory("source_root", source_root)
    require_directory("local_visualization_dir", local_dir)
    require_directory("qnn_visualization_dir", qnn_dir)
    data_root = source_root / "data" / "nuscenes"
    require_directory("nuscenes_data_root", data_root)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing_output:
        raise RuntimeError(f"Output dir not empty: {output_dir}")
    visualization_dir = output_dir / "visualization"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    local_assets = index_visualization_assets(local_dir)
    qnn_assets = index_visualization_assets(qnn_dir)
    keys = validate_asset_pairing(local_assets, qnn_assets, args.expected_frame_count)
    print(f"FRAME_TOKEN_ALIGNMENT_GATE=PASS COUNT={len(keys)}")

    os.environ.setdefault("MPLBACKEND", "Agg")
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version="v1.0-trainval", dataroot=str(data_root), verbose=False)

    frame_records: List[MutableMapping[str, object]] = []
    camera_outputs: List[Path] = []
    bev_outputs: List[Path] = []

    for key in keys:
        stem = f"frame_{key.index:03d}_{key.token}"
        gt_bev = visualization_dir / "gt_bev" / f"{stem}_gt_only_bev.png"
        camera_output = visualization_dir / f"{stem}_three_way_camera.png"
        bev_output = visualization_dir / f"{stem}_three_way_bev.png"

        render_gt_only_bev(nusc, key.token, gt_bev)
        compose_camera_comparison(
            local_assets[key]["camera"],
            qnn_assets[key]["camera"],
            camera_output,
        )
        compose_bev_comparison(
            gt_bev,
            local_assets[key]["bev"],
            qnn_assets[key]["bev"],
            bev_output,
        )

        camera_outputs.append(camera_output)
        bev_outputs.append(bev_output)
        frame_records.append(
            {
                "frame_index": key.index,
                "sample_token": key.token,
                "local_camera": str(local_assets[key]["camera"]),
                "local_bev": str(local_assets[key]["bev"]),
                "qnn_camera": str(qnn_assets[key]["camera"]),
                "qnn_bev": str(qnn_assets[key]["bev"]),
                "gt_bev": str(gt_bev.relative_to(visualization_dir)),
                "camera_comparison": camera_output.name,
                "bev_comparison": bev_output.name,
            }
        )
        print(f"FRAME_{key.index:03d}_THREE_WAY_VISUALIZATION_GATE=PASS TOKEN={key.token}")

    make_gif(camera_outputs, visualization_dir / "bevformer_three_way_camera.gif")
    make_gif(bev_outputs, visualization_dir / "bevformer_three_way_bev.gif")
    write_html(visualization_dir, frame_records)

    manifest_path = visualization_dir / "three_way_manifest.json"
    manifest_path.write_text(
        json.dumps({"frames": frame_records}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    outputs = discover_outputs(visualization_dir)
    final_gate = (
        outputs["camera_comparison_count"] == args.expected_frame_count
        and outputs["bev_comparison_count"] == args.expected_frame_count
        and outputs["gt_bev_count"] == args.expected_frame_count
        and outputs["gif_count"] >= 2
        and outputs["html_count"] >= 1
    )
    report = {
        "audit_type": "GT_PYTORCH_QNN240_THREE_WAY_VISUALIZATION",
        "source_root": str(source_root),
        "local_visualization_dir": str(local_dir),
        "qnn_visualization_dir": str(qnn_dir),
        "frame_count": len(keys),
        "expected_frame_count": args.expected_frame_count,
        "frame_token_alignment_gate": "PASS",
        "layout": {
            "camera": "vertical: GT, Local PyTorch Golden, QNN2.40",
            "bev": "horizontal: GT-only, Local+GT, QNN2.40+GT",
        },
        "outputs": outputs,
        "manifest": str(manifest_path),
        "final_gate": "PASS" if final_gate else "FAIL",
    }
    write_reports(output_dir, report)

    for name, count in outputs.items():
        print(f"{name.upper()}={count}")
    print(f"HTML={visualization_dir / 'index.html'}")
    print(f"FINAL_THREE_WAY_VISUALIZATION_GATE={'PASS' if final_gate else 'FAIL'}")
    return 0 if final_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
