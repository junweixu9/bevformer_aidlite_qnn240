# Ten-frame coordinate sidecar export

## Purpose

The frozen performance runner executes three warmup frames and ten measured
frames, but its established output contract stores only
`frame009_final_coordinates.npz`.

The sidecar entry:

```text
python/bevformer_aidlite_qnn240_e2e_tenframe_export_v1.py
```

loads the frozen runner, wraps its module-level `execute_frame`, calls the
original `main()`, and records only the ten measured coordinate outputs in
memory.

## Non-interference boundaries

- The frozen performance runner is not modified.
- The first three calls are warmup and are not collected.
- Coordinate copies occur after `execute_frame` has calculated
  `frame_total_wall_ms`.
- No file is written from inside `execute_frame` or the measured loop.
- `tenframe_final_coordinates.npz` is written only after the frozen runner
  returns and cleanup has completed.
- The existing `frame009_final_coordinates.npz` remains mandatory.

The sidecar adds Python memory-copy overhead outside the frozen runner's
per-frame internal timing. Board validation must compare the original runner
and sidecar performance summaries before claiming no material process-level
regression.

## Output contract

`tenframe_final_coordinates.npz` contains exactly:

| Key | Shape | dtype |
|---|---:|---|
| `boxes` | `(10, 300, 9)` | `float32` |
| `scores` | `(10, 300)` | `float32` |
| `labels` | `(10, 300)` | `int64` |
| `frame_indices` | `(10,)` | `int64` |

`frame_indices` must equal `[0,1,2,3,4,5,6,7,8,9]`.

Frame 009 must be exact under `numpy.array_equal`:

- `boxes[9] == frame009 boxes`
- `scores[9] == frame009 scores`
- `labels[9] == frame009 labels`

The sidecar also writes:

- `tenframe_coordinate_export_report.json`
- `tenframe_coordinate_export_report.txt`

## Invocation

Use the same arguments as the frozen performance runner, replacing only the
Python entry file. The sidecar intentionally adds no private command-line
arguments, so the frozen runner receives its original CLI contract unchanged.

This Host integration does not execute a board. A separate experimental board
entry must be added later; the established `tools/run_board.sh` remains
unchanged.

## Accepted strict-runner outcome

The established delivery acceptance permits two frozen-runner outcomes:

1. exit `0`, or
2. exit `1` when the frozen strict coordinate gate is `FAIL`, no runtime
   exception occurred, all 3 warmup and 10 measured frames completed,
   interpreter identity remained stable, and cleanup passed.

The second case preserves the established strict float32-tolerance result.
The independent board entry must still run `acceptance.py` against Frame009
before declaring final delivery acceptance. The sidecar only treats the known
strict outcome as non-fatal so that it can write and exactly verify the
ten-frame coordinate asset.

## Independent board execution entry

Container B uses the independent entry:

```bash
bash tools/run_board_tenframe_export.sh
```

The entry preserves `tools/run_board.sh` and the frozen performance runner. It
deploys the sidecar and verifier, executes 3 warmup plus 10 measured frames,
runs the established Frame009 delivery acceptance, independently verifies the
ten-frame exact contract, pulls all output assets, and repeats the coordinate
and runtime checks on Container B.

The entry writes to timestamped remote and local output directories. It does
not overwrite a previous run.
