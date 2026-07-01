# Ten-frame demonstration assets

Runtime assets are not committed to Git. Install the validated ten-frame sequence under:

```text
assets/unseen10/
├── asset_manifest.json
├── sample_000/
├── sample_001/
├── ...
└── sample_009/
```

The manifest must satisfy:

- `status` is `PASS`;
- `frame_indices` is exactly `[0,1,2,3,4,5,6,7,8,9]`;
- every frame contains `images`, `can_bus`, `lidar2img`, and `shift` records;
- frame 000 contains `prev_bev_reference_semantic` and it is all zero;
- frames 001–009 contain `rotation_can_bus`;
- each record provides `path`, `dtype`, `shape`, and `sha256`.

The measured loop preloads these files before timing. JPEG/PNG decode, resize, normalize, and disk I/O are not included in the published steady-state latency.
