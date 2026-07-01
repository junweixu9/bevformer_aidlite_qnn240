from __future__ import annotations

from portable_numpy_nmsfreecoder import decode_numpy_nmsfreecoder


def decode(cls_scores, bbox_preds, contract):
    return decode_numpy_nmsfreecoder(cls_scores, bbox_preds, contract)
