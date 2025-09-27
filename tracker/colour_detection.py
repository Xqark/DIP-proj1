from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from .types import BoundingBox


def _as_uint8_array(values: Tuple[int, int, int]) -> np.ndarray:
    return np.array(values, dtype=np.uint8)


@dataclass(frozen=True)
class ColourDetectorConfig:
    """Parameters describing how to locate the initial coloured target."""

    hsv_lower: Tuple[int, int, int]
    hsv_upper: Tuple[int, int, int]
    min_area_frac: float = 0.001
    max_area_frac: float = 0.05
    target_hue: int | None = None
    median_kernel_size: int = 5
    open_kernel_size: int = 5
    close_kernel_size: int = 13
    aspect_prior: float = 1.5
    aspect_sigma: float = 0.5
    margin_ratio: float = 0.05
    extra_ranges: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...] = ()

    def lower_array(self) -> np.ndarray:
        return _as_uint8_array(self.hsv_lower)

    def upper_array(self) -> np.ndarray:
        return _as_uint8_array(self.hsv_upper)


def build_hsv_mask(frame_hsv: np.ndarray, config: ColourDetectorConfig) -> np.ndarray:
    """Construct a binary mask selecting pixels within the configured HSV bands."""

    mask = cv2.inRange(frame_hsv, config.lower_array(), config.upper_array())

    for lower, upper in config.extra_ranges:
        mask = cv2.bitwise_or(mask, cv2.inRange(frame_hsv, _as_uint8_array(lower), _as_uint8_array(upper)))


    if config.median_kernel_size > 1:
        mask = cv2.medianBlur(mask, config.median_kernel_size)

    if config.open_kernel_size > 1:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((config.open_kernel_size, config.open_kernel_size), np.uint8),
        )

    if config.close_kernel_size > 1:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((config.close_kernel_size, config.close_kernel_size), np.uint8),
        )

    return mask


def detect_colour_bbox(frame_bgr: np.ndarray, config: ColourDetectorConfig) -> BoundingBox:
    """Locate the dominant colour region that corresponds to the target vehicle.

    The detection logic mirrors the exploratory script but exposes knobs so
    callers can adapt to other colour schemes (e.g., the red sequence).

    Not used by the NCCColourTracker, but kept for reference.
    """

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = build_hsv_mask(hsv, config)
    height, width = mask.shape[:2]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours detected for the given colour range.")

    frame_area = float(height * width)
    min_area = config.min_area_frac * frame_area
    max_area = config.max_area_frac * frame_area

    target_hue = config.target_hue
    hue_channel = hsv[:, :, 0]

    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue

        x, y, w_rect, h_rect = cv2.boundingRect(contour)
        aspect = w_rect / max(h_rect, 1)
        aspect_penalty = math.exp(-((aspect - config.aspect_prior) ** 2) / max(config.aspect_sigma, 1e-6))

        hue_score = 1.0
        if target_hue is not None:
            mask_roi = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(mask_roi, [contour], -1, color=255, thickness=cv2.FILLED)
            hue_diff = cv2.absdiff(hue_channel, np.full_like(hue_channel, target_hue))
            mean_hue_diff = cv2.mean(hue_diff, mask=mask_roi)[0]
            hue_score = math.exp(-(mean_hue_diff / 15.0) ** 2)

        score = area * aspect_penalty * hue_score
        candidates.append((score, contour))

    if not candidates:
        raise RuntimeError("Contours found, but none match the configured area/aspect ratios.")

    _, best_contour = max(candidates, key=lambda item: item[0])
    x, y, w_rect, h_rect = cv2.boundingRect(best_contour)

    margin = int(config.margin_ratio * max(w_rect, h_rect))
    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + w_rect + margin, width - 1)
    y1 = min(y + h_rect + margin, height - 1)
    return (x0, y0), (x1, y1)
