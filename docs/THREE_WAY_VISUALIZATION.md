# GT / Local PyTorch / QNN2.40 three-way visualization

This route compares the same ten nuScenes frames across three roles:

1. **GT**: nuScenes annotations and task truth.
2. **Local PyTorch Golden**: the original BEVFormer algorithm-level reference.
3. **QNN2.40 on QCS8550**: the deployed board prediction.

It reuses the already-rendered Local PyTorch and QNN2.40 Camera/BEV assets. It does **not** run Backbone, Encoder, Decoder, NMSFreeCoder, ORT, QAIRT, QNN, or the board again.

## Layout

- **Camera:** three full-width rows per frame: GT, Local PyTorch prediction, QNN2.40 prediction. Full-width rows preserve the readability of all six camera views.
- **BEV:** three columns per frame: GT-only, Local PyTorch prediction over GT, QNN2.40 prediction over GT.
- **Colors in BEV:** green is GT; blue is prediction. LiDAR points are visualization background only and are not BEVFormer model input.
- The Camera GT row is taken from the GT half of the validated official-style renderer. Local and QNN frame indices and sample tokens must match exactly.

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

- 10 `*_three_way_camera.png`
- 10 `*_three_way_bev.png`
- 10 GT-only BEV PNGs
- 2 GIFs
- 1 `index.html`
- `three_way_manifest.json`
- `three_way_visualization_report.json`
- `three_way_visualization_report.txt`

The local and QNN visualization directories must contain the same frame indices and sample tokens. Token mismatch is a hard failure.
