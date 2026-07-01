# Model installation

Install the three validated QNN2.40 Context files under:

```text
models/QCS8550/QNN240/
├── backbone_context.bin
├── encoder_context.bin
└── decoder_context.bin
```

From the repository root, run:

```bash
bash tools/copy_models.sh BACKBONE_SOURCE ENCODER_SOURCE DECODER_SOURCE models/QCS8550/QNN240
```

The helper verifies source and destination SHA256 values. Expected hashes are recorded in `models/EXPECTED_SHA256.txt`.

Do not substitute QAIRT/QNN 2.46 Context files.
