from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from .colour_detection import ColourDetectorConfig, build_hsv_mask


def compute_ncc_response(image_gray: np.ndarray, template_gray: np.ndarray) -> np.ndarray:
    """Compute the normalized cross-correlation response map."""
    return cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)


def resize_response_for_display(
    response: np.ndarray,
    target_shape: Tuple[int, int],
    *,
    use_padding: bool = False,
) -> np.ndarray:
    """Resize an NCC response map to ``target_shape`` for visualization."""
    target_h, target_w = target_shape
    resp_h, resp_w = response.shape

    if use_padding:
        pad_h_total = target_h - resp_h
        pad_w_total = target_w - resp_w
        if pad_h_total < 0 or pad_w_total < 0:
            raise ValueError("Target shape must be at least as large as the response when padding.")

        pad_h_top = pad_h_total // 2
        pad_h_bottom = pad_h_total - pad_h_top
        pad_w_left = pad_w_total // 2
        pad_w_right = pad_w_total - pad_w_left

        min_val = float(response.min()) if response.size else 0.0
        return np.pad(
            response,
            ((pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)),
            mode="constant",
            constant_values=min_val,
        )

    return cv2.resize(response, (target_w, target_h), interpolation=cv2.INTER_CUBIC)


def compute_hsv_backprojection(
    frame_bgr: np.ndarray,
    template_bgr: np.ndarray,
    colour_config: ColourDetectorConfig | None,
    *,
    hist_bins: Tuple[int, int] = (30, 32),
    blur_kernel: int = 31,
) -> np.ndarray:
    """Compute an HSV backprojection map, optionally masked by colour heuristics."""
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    template_hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([template_hsv], [0, 1], None, hist_bins, [0, 180, 0, 256])
    hist = cv2.GaussianBlur(hist, (5, 5), 0)
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    backproj = cv2.calcBackProject([frame_hsv], [0, 1], hist, [0, 180, 0, 256], scale=1)

    if colour_config is not None:
        mask = build_hsv_mask(frame_hsv, colour_config)
        if mask is not None:
            mask_scale = mask.astype(backproj.dtype) / 255.0
            backproj = backproj * mask_scale

    if blur_kernel > 1:
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        backproj = cv2.GaussianBlur(backproj, (blur_kernel, blur_kernel), 0)

    return cv2.normalize(backproj, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)


def detect_bbox_from_response(
    fused_response: np.ndarray,
    template_shape: Tuple[int, int],
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Derive an image-space bounding box centred on the best fused response."""
    if fused_response.ndim != 2:
        raise ValueError("Fused response must be a single-channel array.")

    template_h, template_w = template_shape
    frame_h, frame_w = fused_response.shape
    if template_h > frame_h or template_w > frame_w:
        raise ValueError("Template larger than fused response.")

    _, _, _, max_loc = cv2.minMaxLoc(fused_response)
    center_x, center_y = max_loc

    half_width = template_w // 2
    half_height = template_h // 2

    x0 = center_x - half_width
    y0 = center_y - half_height

    x0 = int(np.clip(x0, 0, frame_w - template_w))
    y0 = int(np.clip(y0, 0, frame_h - template_h))

    x1 = x0 + template_w - 1
    y1 = y0 + template_h - 1

    return (x0, y0), (x1, y1)
