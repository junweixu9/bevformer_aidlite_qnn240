from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BevFormerConfig:
    backbone_model: Path
    encoder_model: Path
    decoder_model: Path
    asset_manifest: Path
    nms_contract: Path
    reference: Path
    output_dir: Path


class BevFormerPipeline:
    """Stable project-level facade around the frozen validated runner.

    The frozen runner remains the numerical source of truth. This class gives
    callers a compact YOLO-style interface without changing model semantics.
    """

    def __init__(self, config: BevFormerConfig, runner_path: Path):
        self.config = config
        self.runner_path = Path(runner_path)

    def command(self) -> list[str]:
        result_json = self.config.output_dir / "performance_result.json"
        return [
            sys.executable,
            str(self.runner_path),
            "--backbone-model", str(self.config.backbone_model),
            "--encoder-model", str(self.config.encoder_model),
            "--decoder-model", str(self.config.decoder_model),
            "--asset-manifest", str(self.config.asset_manifest),
            "--nms-contract", str(self.config.nms_contract),
            "--nms-reference", str(self.config.reference),
            "--output-directory", str(self.config.output_dir),
            "--result-json", str(result_json),
        ]

    def run(self) -> int:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        return subprocess.run(self.command(), check=False).returncode
