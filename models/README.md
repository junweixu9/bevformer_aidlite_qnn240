# Model installation

The three QNN2.40 serialized contexts are not committed to Git. Install them on the QCS8550 board under:

```text
models/QCS8550/QNN240/
├── backbone_context.bin
├── encoder_context.bin
└── decoder_context.bin
```

Expected SHA256 values are recorded in `models/EXPECTED_SHA256.txt`.

Example:

```bash
mkdir -p models/QCS8550/QNN240
cp /path/to/backbone_context.bin models/QCS8550/QNN240/
cp /path/to/encoder_context.bin models/QCS8550/QNN240/
cp /path/to/decoder_context.bin models/QCS8550/QNN240/
sha256sum -c models/EXPECTED_SHA256.txt
```

Do not substitute QAIRT/QNN 2.46 contexts. These identities are the validated QNN2.40 delivery assets.
