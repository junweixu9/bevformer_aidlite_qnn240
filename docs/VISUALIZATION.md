## Host-A visualization

Camera and BEV visualization runs separately in Container A using the validated `open-mmlab` Conda environment. Visualization is host-side presentation and is outside the published QCS8550 steady-state latency path.

### Single-result route

Use this route to render one MMDetection3D result PKL:

```bash
cp tools/host_visualization.env.example tools/host_visualization.env
bash tools/run_host_a_visualization.sh
```

### Preferred three-way comparison

Use this route to place the task truth, algorithm reference and board deployment result in one page:

```text
GT (nuScenes) | Local PyTorch Golden | QNN2.40 on QCS8550
```

Configure and generate:

```bash
cp tools/three_way_visualization.env.example tools/three_way_visualization.env
bash tools/run_three_way_visualization.sh
```

View the latest generated page with one command:

```bash
bash tools/serve_latest_three_way_visualization.sh
```

See `docs/THREE_WAY_VISUALIZATION.md` for layout, lineage and output contracts.
