# GT / Local PyTorch / QNN2.40 three-way visualization

This route compares the same ten nuScenes frames across three roles:

1. **GT**: nuScenes annotations and task truth.
2. **Local PyTorch Golden**: the original BEVFormer algorithm-level reference.
3. **QNN2.40 on QCS8550**: the deployed board prediction.

It reuses the already-rendered Local PyTorch and QNN2.40 Camera/BEV assets. It does **not** run Backbone, Encoder, Decoder, NMSFreeCoder, ORT, QAIRT, QNN, or the board again.

## Interactive layout

- Only one frame is shown at a time, avoiding a very long page.
- Camera supports GT / PyTorch / QNN single-route full-width viewing and a horizontally scrollable three-way comparison mode.
- BEV remains a three-column comparison: GT-only, Local PyTorch+GT, QNN2.40+GT.
- Any image can be opened in a full-screen modal without browser-side downscaling.
- Frame buttons, previous/next controls, keyboard arrow navigation and automatic playback are provided.
- Green is GT; blue is prediction. LiDAR points are visualization background only and are not BEVFormer input.
- Local and QNN frame indices and sample tokens must match exactly.

## Configure once

```bash
cp tools/three_way_visualization.env.example tools/three_way_visualization.env
```

The private `tools/three_way_visualization.env` file is ignored by Git.

## Generate the comparison

```bash
bash tools/run_three_way_visualization.sh
```

A timestamped output directory is created under `THREE_WAY_VISUALIZATION_OUTPUT_ROOT`. A successful run ends with:

```text
FRAME_TOKEN_ALIGNMENT_GATE=PASS COUNT=10
CAMERA_ASSET_COUNT=30
BEV_ASSET_COUNT=30
FINAL_THREE_WAY_VISUALIZATION_GATE=PASS
THREE_WAY_HTML_GATE=PASS
```

## View the latest result with one command

```bash
bash tools/serve_latest_three_way_visualization.sh
```

Then open:

```text
http://127.0.0.1:8000/index.html
```

Use VS Code Remote port forwarding for port `8000`. Override the port with:

```bash
THREE_WAY_VISUALIZATION_PORT=8001 bash tools/serve_latest_three_way_visualization.sh
```

## Output contract

For ten frames, the output contains:

- 30 Camera PNG assets: GT, Local PyTorch and QNN2.40 for each frame.
- 30 BEV PNG assets: GT-only, Local PyTorch+GT and QNN2.40+GT for each frame.
- 1 interactive `index.html`.
- `three_way_manifest.json`.
- `three_way_visualization_report.json`.
- `three_way_visualization_report.txt`.

The page intentionally does not use pre-composed giant PNGs or GIFs. Native-resolution route assets are retained and selected interactively in the browser.
