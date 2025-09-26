from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import cv2
import numpy as np

from .colour_detection import ColourDetectorConfig
from .fusion import compute_hsv_backprojection
from .types import BoundingBox
from .utils import extract_patch


@dataclass(frozen=True)
class SearchWindowConfig:
    """Defines how far the tracker may search around the previous location."""

    margin_px: int = 32
    margin_scale: float = 0.5  # Scale relative to template width/height.


@dataclass(frozen=True)
class TemplateUpdateConfig:
    """Template maintenance behaviour for the adaptive tracker."""

    enabled: bool = True
    alpha: float = 0.05  # Blend factor for exponential moving average.
    max_pixel_shift: int | None = 40  # Skip updates on large jumps to avoid drift.


@dataclass(frozen=True)
class TrackerConfig:
    """Aggregates knobs for the NCC + colour fusion tracker."""

    search_window: SearchWindowConfig = field(default_factory=SearchWindowConfig)
    template_update: TemplateUpdateConfig = field(default_factory=TemplateUpdateConfig)
    backproj_hist_bins: Tuple[int, int] = (30, 32)
    backproj_blur: int = 31


@dataclass(slots=True)
class TrackingResult:
    top_left: Tuple[int, int]
    bottom_right: Tuple[int, int]
    ncc_score: float
    fused_score: float
    search_top_left: Tuple[int, int]
    search_bottom_right: Tuple[int, int]


def _expand_bbox(
    bbox: BoundingBox,
    frame_shape: Tuple[int, int, int],
    template_shape: Tuple[int, int],
    config: SearchWindowConfig,
) -> BoundingBox:
    (x0, y0), (x1, y1) = bbox
    frame_h, frame_w = frame_shape[:2]
    template_h, template_w = template_shape

    width = x1 - x0 + 1
    height = y1 - y0 + 1

    margin_x = max(config.margin_px, int(width * config.margin_scale))
    margin_y = max(config.margin_px, int(height * config.margin_scale))

    sx0 = max(x0 - margin_x, 0)
    sy0 = max(y0 - margin_y, 0)
    sx1 = min(x1 + margin_x, frame_w - 1)
    sy1 = min(y1 + margin_y, frame_h - 1)

    # Ensure the search window still accommodates the template.
    if sx1 - sx0 + 1 < template_w:
        sx0 = max(0, min(sx0, frame_w - template_w))
        sx1 = sx0 + template_w - 1
    if sy1 - sy0 + 1 < template_h:
        sy0 = max(0, min(sy0, frame_h - template_h))
        sy1 = sy0 + template_h - 1

    return (sx0, sy0), (sx1, sy1)


def _clamp_bbox(
    top_left: Tuple[int, int],
    frame_shape: Tuple[int, int, int],
    template_shape: Tuple[int, int],
) -> BoundingBox:
    frame_h, frame_w = frame_shape[:2]
    template_h, template_w = template_shape

    x0 = min(max(top_left[0], 0), frame_w - template_w)
    y0 = min(max(top_left[1], 0), frame_h - template_h)

    x1 = x0 + template_w - 1
    y1 = y0 + template_h - 1
    return (x0, y0), (x1, y1)


class NCCColourTracker:
    """Template tracker that fuses NCC responses with HSV back-projection."""

    def __init__(
        self,
        initial_frame_bgr: np.ndarray,
        initial_bbox: BoundingBox,
        config: TrackerConfig | None = None,
        *,
        colour_config: ColourDetectorConfig | None = None,
    ) -> None:
        if config is None:
            config = TrackerConfig()
        self.config = config
        self.template_bgr = extract_patch(initial_frame_bgr, *initial_bbox)
        if self.template_bgr.size == 0:
            raise ValueError("Initial bounding box produces an empty template patch.")
        self.template_gray = cv2.cvtColor(self.template_bgr, cv2.COLOR_BGR2GRAY)
        self.template_shape = self.template_gray.shape  # (rows, cols)
        self.prev_bbox = initial_bbox
        self.colour_config = colour_config

    def reset_template(self, frame_bgr: np.ndarray, bbox: BoundingBox) -> None:
        """Replace the template with an externally provided bounding box."""
        self.template_bgr = extract_patch(frame_bgr, *bbox)
        if self.template_bgr.size == 0:
            raise ValueError("Reset bbox is invalid (empty patch).")
        self.template_gray = cv2.cvtColor(self.template_bgr, cv2.COLOR_BGR2GRAY)
        self.template_shape = self.template_gray.shape
        self.prev_bbox = bbox

    def _update_template(self, frame_bgr: np.ndarray, prev_top_left: Tuple[int, int], new_bbox: BoundingBox) -> None:
        update_cfg = self.config.template_update
        if not update_cfg.enabled:
            return

        new_top_left = new_bbox[0]
        if update_cfg.max_pixel_shift is not None:
            shift = np.linalg.norm(np.subtract(new_top_left, prev_top_left))
            if shift > update_cfg.max_pixel_shift:
                return

        fresh_patch = extract_patch(frame_bgr, *new_bbox).astype(np.float32)
        template = self.template_bgr.astype(np.float32)
        alpha = update_cfg.alpha
        self.template_bgr = cv2.addWeighted(fresh_patch, alpha, template, 1.0 - alpha, 0.0).astype(np.uint8)
        self.template_gray = cv2.cvtColor(self.template_bgr, cv2.COLOR_BGR2GRAY)
        self.template_shape = self.template_gray.shape

    def track(self, frame_bgr: np.ndarray) -> TrackingResult:
        search_bbox = _expand_bbox(
            self.prev_bbox,
            frame_bgr.shape,
            self.template_shape,
            self.config.search_window,
        )
        search_top_left, search_bottom_right = search_bbox
        search_patch = extract_patch(frame_bgr, search_top_left, search_bottom_right)

        template_h, template_w = self.template_shape
        if (
            search_patch.shape[0] < template_h
            or search_patch.shape[1] < template_w
        ):
            raise RuntimeError("Search window smaller than template—check search configuration.")

        search_gray = cv2.cvtColor(search_patch, cv2.COLOR_BGR2GRAY).astype(np.float32)
        template_gray = self.template_gray.astype(np.float32)

        ncc = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        ncc_norm = cv2.normalize(ncc, None, 0.0, 1.0, cv2.NORM_MINMAX)

        if self.colour_config is not None:
            backproj_full = compute_hsv_backprojection(
                frame_bgr,
                self.template_bgr,
                self.colour_config,
                hist_bins=self.config.backproj_hist_bins,
                blur_kernel=self.config.backproj_blur,
            )
            backproj_patch = extract_patch(
                backproj_full,
                search_top_left,
                search_bottom_right,
            )
        else:
            backproj_patch = compute_hsv_backprojection(
                search_patch,
                self.template_bgr,
                None,
                hist_bins=self.config.backproj_hist_bins,
                blur_kernel=self.config.backproj_blur,
            )

        backproj_small = cv2.resize(
            backproj_patch,
            (ncc.shape[1], ncc.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

        fused = (ncc_norm * backproj_small).astype(np.float32)
        _, fused_max, _, fused_loc = cv2.minMaxLoc(fused)
        ncc_score = float(ncc[fused_loc[1], fused_loc[0]])

        abs_top_left = (
            search_top_left[0] + fused_loc[0],
            search_top_left[1] + fused_loc[1],
        )
        bbox = _clamp_bbox(abs_top_left, frame_bgr.shape, self.template_shape)

        prev_top_left = self.prev_bbox[0]
        self.prev_bbox = bbox
        self._update_template(frame_bgr, prev_top_left, bbox)

        return TrackingResult(
            top_left=bbox[0],
            bottom_right=bbox[1],
            ncc_score=float(ncc_score),
            fused_score=float(fused_max),
            search_top_left=search_top_left,
            search_bottom_right=search_bottom_right,
        )

    @property
    def template(self) -> np.ndarray:
        return self.template_bgr
