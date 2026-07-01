# Frozen Runner — read-only historical reference

These files constitute the validated numerical snapshot that serves as the
source of truth for all BEVFormer QNN2.40 inference.

## Status

- **Frozen on**: 2026-06-30
- **SHA256**: see `SHA256SUMS` in this directory
- **Verified on**: QCS8550, AidLite 2.4.0.265, QNN 2.40

## Files

| File | Role |
|------|------|
| `bevformer_aidlite_qnn240_e2e_performance_v1.py` | Main Runner: Backbone → Encoder → Decoder → NMSFreeCoder |
| `functional_mother.py` | Utility functions (SHA, Manifest, timing) |
| `portable_numpy_nmsfreecoder.py` | Board-side NumPy NMSFreeCoder |
| `run_bevformer_aidlite_qnn240_e2e_performance_v1.sh` | Board launcher (sets env vars and calls Python) |

## Contract

These files **must not** be modified as part of normal development.
Any new implementation (in `python/bevformer.py`, `python/run_test.py`, etc.)
must produce numerically identical output to this frozen baseline.

When the new implementation passes full numerical regression, these files
remain for historical auditability — they are not deleted.
