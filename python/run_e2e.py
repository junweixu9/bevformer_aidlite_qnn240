#!/usr/bin/env python3
"""BEVFormer AidLite QNN2.40 — 六路相机张量到 3D 坐标端到端入口。

本地用法（仅验证坐标精度，不执行推理）：
  python3 python/run_e2e.py --verify-only \\
    --candidate outputs/run_YYYYMMDD_HHMMSS/frame009_final_coordinates.npz

板端完整执行（需 SSH 到 QCS8550 板端）：
  bash tools/run_board.sh

完整参数说明（板端推理所需的所有参数）：
  python3 python/run_e2e.py --help
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
REFERENCE_DEFAULT = PACKAGE_DIR / "frame009_numpy_native_reference.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BEVFormer AidLite QNN2.40 — 六路相机张量 → 3D 坐标"
    )

    # ── Board execution parameters (documentation + pass-through) ──
    group_board = parser.add_argument_group("板端推理参数 (由 tools/run_board.sh 自动设置)")
    group_board.add_argument(
        "--backbone-model",
        default="/home/aidlux/.../backbone_context.bin",
        help="Backbone QNN2.40 serialized context 路径 (板端)",
    )
    group_board.add_argument(
        "--encoder-model",
        default="/home/aidlux/.../encoder_...serialized.bin",
        help="Encoder QNN2.40 serialized context 路径 (板端)",
    )
    group_board.add_argument(
        "--decoder-model",
        default="/home/aidlux/.../decoder_context.bin",
        help="Decoder QNN2.40 serialized context 路径 (板端)",
    )
    group_board.add_argument(
        "--asset-manifest",
        default="/home/aidlux/.../asset_manifest.json",
        help="十帧资产清单 JSON (frame_indices: [0..9])",
    )
    group_board.add_argument(
        "--output-directory",
        default="./outputs/run_YYYYMMDD_HHMMSS",
        help="输出目录 (performance_result.json, coordinates.npz)",
    )

    # ── Local verification parameters ──
    group_local = parser.add_argument_group("本地精度验证 (在 Container B 执行，无需板端)")
    group_local.add_argument(
        "--verify-only",
        action="store_true",
        help="仅对已有坐标执行 float32 epsilon 修正合同验证，不发起板端推理",
    )
    group_local.add_argument(
        "--reference",
        type=str,
        default=str(REFERENCE_DEFAULT),
        help="Frame009 NumPy 参考坐标 .npz (默认: python/frame009_numpy_native_reference.npz)",
    )
    group_local.add_argument(
        "--candidate",
        type=str,
        help="待验证的 frame009_final_coordinates.npz 路径",
    )

    # ── Board remote execution shortcut ──
    group_remote = parser.add_argument_group("板端远程执行捷径")
    group_remote.add_argument(
        "--run-board",
        action="store_true",
        help="直接调用 tools/run_board.sh 启动板端完整推理流程",
    )

    return parser.parse_args()


def run_local_verify(reference: str, candidate: str) -> int:
    """Run the float32-epsilon corrected contract verifier locally."""
    verify_script = PACKAGE_DIR / "verify_contract.py"
    if not verify_script.exists():
        print(f"ERROR: verifier not found at {verify_script}", file=sys.stderr)
        return 1

    result = subprocess.run(
        [
            sys.executable,
            str(verify_script),
            "--reference", reference,
            "--candidate", candidate,
            "--report-json", str(Path(candidate).parent / "corrected_float32_tolerance_report.json"),
            "--report-txt",  str(Path(candidate).parent / "corrected_float32_tolerance_report.txt"),
        ],
    )
    return result.returncode


def run_board_remote() -> int:
    """Invoke tools/run_board.sh for full board execution."""
    script = PROJECT_ROOT / "tools" / "run_board.sh"
    if not script.exists():
        print(f"ERROR: board runner not found at {script}", file=sys.stderr)
        return 1

    result = subprocess.run(["bash", str(script)])
    return result.returncode


def main() -> int:
    args = parse_args()

    if args.verify_only:
        if not args.candidate:
            print("ERROR: --verify-only 需要同时指定 --candidate PATH", file=sys.stderr)
            return 1
        print(f"[local verify] reference: {args.reference}")
        print(f"[local verify] candidate: {args.candidate}")
        return run_local_verify(args.reference, args.candidate)

    if args.run_board:
        print("[board] 启动板端远程执行...")
        return run_board_remote()

    # No action flags: print help summary
    print("BEVFormer AidLite QNN2.40 — 端到端入口")
    print()
    print("用法:")
    print("  板端完整执行:   bash tools/run_board.sh")
    print("  本地精度验证:   python3 python/run_e2e.py --verify-only --candidate <path>")
    print("  板端执行捷径:   python3 python/run_e2e.py --run-board")
    print()
    print("关键参数 (由 tools/run_board.sh 自动设置):")
    print("  --backbone-model    Backbone QNN2.40 Context 路径")
    print("  --encoder-model     Encoder QNN2.40 Context 路径")
    print("  --decoder-model     Decoder QNN2.40 Context 路径")
    print("  --asset-manifest    十帧资产清单 JSON")
    print("  --output-directory  输出目录")
    print()
    print("数据流:")
    print("  六路相机张量 (6×3×450×800 uint8)")
    print("    → Backbone (HTP) → img_feat (6×256×15×25 uint8)")
    print("    → NumPy 内存交接")
    print("    → Encoder (HTP) + prev_bev 时序递归 → bev_embed (1×2500×256 fp16)")
    print("    → Decoder (HTP) → cls_scores + bbox_preds (900×10 fp16)")
    print("    → NumPy NMSFreeCoder (板端 CPU) → 300 个九维 3D 检测框")
    print()
    print("正式性能: 总墙钟 mean=474.94ms, P95=481.73ms")
    print("当前边界: 不含原图解码/resize/normalize (V2 计划)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
