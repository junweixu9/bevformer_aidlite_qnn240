from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_COMMON_ASSETS = ("images", "can_bus", "lidar2img", "shift")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest_data(data: dict[str, Any]) -> None:
    if data.get("status") != "PASS":
        raise ValueError("manifest status must be PASS")
    if data.get("frame_indices") != list(range(10)):
        raise ValueError("manifest frame_indices must be exactly 0..9")
    frames = data.get("frames")
    if not isinstance(frames, dict):
        raise ValueError("manifest frames must be an object")

    for frame_index in range(10):
        sample = f"sample_{frame_index:03d}"
        try:
            assets = frames[sample]["assets"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"missing assets for {sample}") from exc
        missing = [name for name in REQUIRED_COMMON_ASSETS if name not in assets]
        if missing:
            raise ValueError(f"{sample} missing assets: {','.join(missing)}")
        required_temporal = (
            "prev_bev_reference_semantic" if frame_index == 0 else "rotation_can_bus"
        )
        if required_temporal not in assets:
            raise ValueError(f"{sample} missing asset: {required_temporal}")


def load_and_validate_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest_data(data)
    return data
