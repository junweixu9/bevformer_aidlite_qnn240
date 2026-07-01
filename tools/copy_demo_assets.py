#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy and rewrite the ten-frame demo asset manifest")
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()

    source_manifest = Path(args.source_manifest).expanduser().resolve()
    destination = Path(args.destination).expanduser().resolve()
    data = json.loads(source_manifest.read_text(encoding="utf-8"))

    if data.get("status") != "PASS":
        raise SystemExit("source manifest status is not PASS")
    if data.get("frame_indices") != list(range(10)):
        raise SystemExit("source manifest frame_indices must be 0..9")

    destination.mkdir(parents=True, exist_ok=True)
    for frame_index in range(10):
        sample = f"sample_{frame_index:03d}"
        assets = data["frames"][sample]["assets"]
        sample_dir = destination / sample
        sample_dir.mkdir(parents=True, exist_ok=True)

        for name, record in assets.items():
            source = Path(record["path"]).expanduser().resolve()
            if not source.is_file() or source.stat().st_size == 0:
                raise SystemExit(f"missing asset: {source}")
            actual = sha256_file(source)
            expected = record.get("sha256")
            if expected and actual != expected:
                raise SystemExit(
                    f"asset SHA mismatch name={name} expected={expected} actual={actual}"
                )
            suffix = "".join(source.suffixes) or ".raw"
            target = sample_dir / f"{name}{suffix}"
            shutil.copy2(source, target)
            record["path"] = str(target)
            record["sha256"] = sha256_file(target)

    installed_manifest = destination / "asset_manifest.json"
    installed_manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"ASSET_INSTALL_GATE=PASS PATH={installed_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
