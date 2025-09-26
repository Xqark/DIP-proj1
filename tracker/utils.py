from __future__ import annotations

from typing import Tuple

import numpy as np

from .types import BoundingBox


Point = Tuple[int, int]


def extract_patch(frame: np.ndarray, top_left: Point, bottom_right: Point) -> np.ndarray:
    """Return a view of ``frame`` bounded by ``top_left`` and ``bottom_right`` inclusive."""
    x0, y0 = top_left
    x1, y1 = bottom_right
    return frame[y0 : y1 + 1, x0 : x1 + 1]


__all__ = ["extract_patch", "Point", "BoundingBox"]
