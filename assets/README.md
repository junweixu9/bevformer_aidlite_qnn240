# Ten-frame demonstration assets

Runtime assets are installed separately from Git under:

```text
assets/unseen10/
├── asset_manifest.json
├── sample_000/
├── sample_001/
├── ...
└── sample_009/
```

Copy and rewrite an existing validated manifest with:

```bash
python3 tools/copy_demo_assets.py \
  --source-manifest /path/to/validated/asset_manifest.json \
  --destination assets/unseen10
```

The helper checks every source file SHA256, copies each asset into the standard project layout, rewrites record paths, and writes a new `asset_manifest.json`.

The manifest contract is:

- `status` equals `PASS`;
- `frame_indices` equals `[0,1,2,3,4,5,6,7,8,9]`;
- every frame contains `images`, `can_bus`, `lidar2img`, and `shift`;
- frame 000 contains an all-zero `prev_bev_reference_semantic`;
- frames 001–009 contain `rotation_can_bus`;
- every record contains `path`, `dtype`, `shape`, and `sha256`.

The measured loop preloads assets before timing. JPEG/PNG decode, resize, normalize, and source-asset disk I/O are excluded from the published steady-state latency.
