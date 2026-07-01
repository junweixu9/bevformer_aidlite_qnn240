#!/usr/bin/env python3
"""Compatibility entry.

New board inference should use ``python/run_test.py``. This file remains for
older commands and delegates to the same implementation instead of maintaining
another inference path.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
RUN_TEST = PACKAGE_DIR / "run_test.py"
VERIFIER = PACKAGE_DIR / "verify_contract.py"
REMOTE_RUNNER = PROJECT_ROOT / "tools" / "run_board.sh"
DEFAULT_REFERENCE = PACKAGE_DIR / "frame009_numpy_native_reference.npz"


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--run-board", action="store_true")
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    parser.add_argument("--candidate")
    known, remaining = parser.parse_known_args()

    if known.verify_only:
        if not known.candidate:
            print("ERROR: --verify-only requires --candidate", file=sys.stderr)
            return 2
        output_dir = Path(known.candidate).expanduser().resolve().parent
        return subprocess.run(
            [
                sys.executable,
                str(VERIFIER),
                "--reference", known.reference,
                "--candidate", known.candidate,
                "--report-json", str(output_dir / "corrected_float32_tolerance_report.json"),
                "--report-txt", str(output_dir / "corrected_float32_tolerance_report.txt"),
            ],
            check=False,
        ).returncode

    if known.run_board:
        return subprocess.run(["bash", str(REMOTE_RUNNER)], check=False).returncode

    if not remaining:
        print("Compatibility entry: use python/run_test.py for board inference.")
        return subprocess.run([sys.executable, str(RUN_TEST), "--help"], check=False).returncode

    return subprocess.run([sys.executable, str(RUN_TEST), *remaining], check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
