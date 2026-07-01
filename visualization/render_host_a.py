#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, runpy, sys
from pathlib import Path
from typing import Dict, Tuple

EXPECTED_CONDA_ENV = "open-mmlab"

def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def require_file(name, path):
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"{name} missing or empty: {path}")

def require_directory(name, path):
    if not path.is_dir():
        raise RuntimeError(f"{name} is not a directory: {path}")

def inspect_result(result_path):
    import mmcv
    result = mmcv.load(str(result_path))
    if isinstance(result, dict) and "bbox_results" in result:
        return len(result["bbox_results"]), "bbox_results"
    elif isinstance(result, dict) and "results" in result:
        return len(result["results"]), "results"
    elif isinstance(result, (list, tuple)):
        return len(result), "direct_sequence"
    raise RuntimeError(f"Unsupported: {type(result).__name__}")

def discover_outputs(output_dir):
    camera_pngs = sorted(output_dir.rglob("*_camera.png"))
    bev_pngs = sorted(output_dir.rglob("*_bev.png"))
    gifs = sorted(output_dir.rglob("*.gif"))
    htmls = sorted(output_dir.rglob("*.html"))
    jsons = sorted(output_dir.rglob("*.json"))
    all_files = sorted(p for p in output_dir.rglob("*") if p.is_file())
    return {
        "camera_png_count": len(camera_pngs), "bev_png_count": len(bev_pngs),
        "gif_count": len(gifs), "html_count": len(htmls), "json_count": len(jsons),
        "files": [{"path": str(p.relative_to(output_dir)), "size_bytes": p.stat().st_size} for p in all_files]
    }

def write_reports(output_dir, report):
    rj = output_dir / "host_a_visualization_report.json"
    rt = output_dir / "host_a_visualization_report.txt"
    rj.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    o = report["outputs"]
    lines = [
        f"AUDIT_TYPE=HOST_A_BEVFORMER_VISUALIZATION",
        f"RESULT_FRAME_COUNT={report['result_frame_count']}",
        f"CAMERA_PNG_COUNT={o['camera_png_count']}", f"BEV_PNG_COUNT={o['bev_png_count']}",
        f"GIF_COUNT={o['gif_count']}", f"HTML_COUNT={o['html_count']}", f"JSON_COUNT={o['json_count']}",
        f"CAMERA_RENDER_GATE={report['camera_render_gate']}", f"BEV_RENDER_GATE={report['bev_render_gate']}",
        f"GIF_RENDER_GATE={report['gif_render_gate']}", f"HTML_RENDER_GATE={report['html_render_gate']}",
        f"FINAL_HOST_A_VISUALIZATION_GATE={report['final_gate']}",
    ]
    rt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rj, rt

def main():
    parser = argparse.ArgumentParser(description="Render BEVFormer result on Container A.")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--result-pkl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--implementation", default=str(Path(__file__).with_name("render_host_a_impl.py")))
    parser.add_argument("--expected-frame-count", type=int, default=10)
    parser.add_argument("--allow-existing-output", action="store_true")
    parser.add_argument("--skip-conda-check", action="store_true")
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    result_pkl = Path(args.result_pkl).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    implementation = Path(args.implementation).expanduser().resolve()

    if not args.skip_conda_check and os.environ.get("CONDA_DEFAULT_ENV") != EXPECTED_CONDA_ENV:
        raise RuntimeError(f"Requires conda env {EXPECTED_CONDA_ENV!r}")

    require_directory("source_root", source_root)
    require_file("result_pkl", result_pkl)
    require_file("implementation", implementation)

    if output_dir.exists() and any(output_dir.iterdir()) and not args.allow_existing_output:
        raise RuntimeError(f"Output dir not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count, result_route = inspect_result(result_pkl)
    print(f"RESULT_ROUTE={result_route}")
    print(f"RESULT_FRAME_COUNT={frame_count}")
    if frame_count != args.expected_frame_count:
        raise RuntimeError(f"Expected {args.expected_frame_count} frames, got {frame_count}")

    os.environ["BEVFORMER_SOURCE_ROOT"] = str(source_root)
    os.environ["BEVFORMER_RESULT_PKL"] = str(result_pkl)
    os.environ["BEVFORMER_VIS_OUTPUT_DIR"] = str(output_dir)
    os.environ.setdefault("MPLBACKEND", "Agg")

    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

    prev_cwd = Path.cwd()
    try:
        os.chdir(str(source_root))
        runpy.run_path(str(implementation), run_name="__main__")
    finally:
        os.chdir(str(prev_cwd))

    outputs = discover_outputs(output_dir)
    camera_gate = outputs["camera_png_count"] == args.expected_frame_count
    bev_gate = outputs["bev_png_count"] == args.expected_frame_count
    gif_gate = outputs["gif_count"] >= 2
    html_gate = outputs["html_count"] >= 1
    final_gate = camera_gate and bev_gate and gif_gate and html_gate

    report = {
        "audit_type": "HOST_A_BEVFORMER_VISUALIZATION",
        "source_root": str(source_root), "result_pkl": str(result_pkl),
        "result_sha256": sha256_file(result_pkl),
        "implementation_sha256": sha256_file(implementation),
        "result_route": result_route, "result_frame_count": frame_count,
        "expected_frame_count": args.expected_frame_count,
        "camera_render_gate": "PASS" if camera_gate else "FAIL",
        "bev_render_gate": "PASS" if bev_gate else "FAIL",
        "gif_render_gate": "PASS" if gif_gate else "FAIL",
        "html_render_gate": "PASS" if html_gate else "FAIL",
        "final_gate": "PASS" if final_gate else "FAIL",
        "outputs": outputs,
    }

    rj, rt = write_reports(output_dir, report)
    o = outputs
    for k in ["camera_png_count","bev_png_count","gif_count","html_count","json_count"]:
        print(f"{k.upper()}={o[k]}")
    print(f"CAMERA_RENDER_GATE={'PASS' if camera_gate else 'FAIL'}")
    print(f"BEV_RENDER_GATE={'PASS' if bev_gate else 'FAIL'}")
    print(f"GIF_RENDER_GATE={'PASS' if gif_gate else 'FAIL'}")
    print(f"HTML_RENDER_GATE={'PASS' if html_gate else 'FAIL'}")
    print(f"FINAL_HOST_A_VISUALIZATION_GATE={'PASS' if final_gate else 'FAIL'}")
    return 0 if final_gate else 1

if __name__ == "__main__":
    raise SystemExit(main())
