from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import rotate as scipy_rotate

BEV_SHAPE = (1, 2500, 256)


def rotate_prev_bev(previous_live_bev: np.ndarray, rotation_can_bus: np.ndarray) -> np.ndarray:
    angle = float(np.asarray(rotation_can_bus).reshape(-1)[-1])
    previous_native = np.ascontiguousarray(previous_live_bev, dtype="<f2")
    previous_semantic = previous_native.astype(np.float32).reshape(BEV_SHAPE)
    bev_grid = previous_semantic[0].reshape(50, 50, 256)
    rotated_fp32 = scipy_rotate(
        bev_grid,
        angle=angle,
        axes=(0, 1),
        reshape=False,
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(BEV_SHAPE).astype(np.float32)
    rotated_native = np.ascontiguousarray(rotated_fp32, dtype="<f2")
    return np.ascontiguousarray(rotated_native.astype(np.float32), dtype=np.float32)


@dataclass
class TemporalState:
    previous_bev: np.ndarray | None = None

    def reset(self) -> None:
        self.previous_bev = None

    def input_for_frame(
        self,
        frame_index: int,
        zero_prev_bev: np.ndarray,
        rotation_can_bus: np.ndarray | None,
    ) -> np.ndarray:
        if frame_index == 0:
            zero = np.ascontiguousarray(zero_prev_bev, dtype=np.float32).reshape(BEV_SHAPE)
            if np.count_nonzero(zero) != 0:
                raise ValueError("frame 000 prev_bev must be all zero")
            return zero
        if self.previous_bev is None:
            raise RuntimeError("previous live bev_embed is unavailable")
        if rotation_can_bus is None:
            raise ValueError("rotation_can_bus is required after frame 000")
        return rotate_prev_bev(self.previous_bev, rotation_can_bus)

    def update(self, bev_embed: np.ndarray) -> None:
        value = np.ascontiguousarray(bev_embed, dtype=np.float32).reshape(BEV_SHAPE)
        if not np.isfinite(value).all():
            raise ValueError("bev_embed contains non-finite values")
        self.previous_bev = value
