from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from bevformer import BevFormerConfig, BevFormerPipeline


def test_pipeline_builds_explicit_frozen_runner_command(tmp_path):
    config = BevFormerConfig(
        backbone_model=Path("/models/backbone.bin"),
        encoder_model=Path("/models/encoder.bin"),
        decoder_model=Path("/models/decoder.bin"),
        asset_manifest=Path("/assets/manifest.json"),
        nms_contract=Path("/configs/nms.json"),
        reference=Path("/reference/frame009.npz"),
        output_dir=tmp_path,
    )
    pipeline = BevFormerPipeline(config, Path("/app/frozen_runner.py"))
    command = pipeline.command()

    assert command[1] == "/app/frozen_runner.py"
    assert command[command.index("--backbone-model") + 1] == "/models/backbone.bin"
    assert command[command.index("--encoder-model") + 1] == "/models/encoder.bin"
    assert command[command.index("--decoder-model") + 1] == "/models/decoder.bin"
    assert command[command.index("--asset-manifest") + 1] == "/assets/manifest.json"
    assert command[command.index("--nms-contract") + 1] == "/configs/nms.json"
    assert command[command.index("--nms-reference") + 1] == "/reference/frame009.npz"
    assert command[command.index("--result-json") + 1] == str(tmp_path / "performance_result.json")
