## Host-A visualization

Camera and BEV visualization runs separately in Container A using the validated `open-mmlab` Conda environment.

```bash
cp tools/host_visualization.env.example tools/host_visualization.env
bash tools/run_host_a_visualization.sh
```

Current smoke-test route: Frames 000-008 = Local PyTorch Golden, Frame 009 = QNN2.40 NumPy-native reference.
