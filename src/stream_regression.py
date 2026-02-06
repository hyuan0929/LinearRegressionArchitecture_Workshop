# src/stream_regression.py
from __future__ import annotations

import numpy as np


def stream_slope(y: list[float]) -> float:
    """
    Linear regression slope of y vs x=1..N.
    """
    y_arr = np.asarray(y, dtype=float)
    x = np.arange(1, len(y_arr) + 1, dtype=float)

    xm = x.mean()
    ym = y_arr.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0

    return float((((x - xm) * (y_arr - ym)).sum()) / denom)
