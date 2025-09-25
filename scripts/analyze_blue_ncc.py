from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

BLUE_DIR = Path(__file__).resolve().parent.parent / "sequences" / "blue"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

# Narrower HSV band focused on saturated blue paint.
HSV_BLUE_RANGE = (
    np.array([100, 80, 70], dtype=np.uint8),
    np.array([135, 255, 255], dtype=np.uint8),
)

# Expected contour area as a fraction of full frame.
MIN_AREA_FRAC = 0.001
MAX_AREA_FRAC = 0.03

# Focus the colour search to the mid-band of the frame where the horizon sits.
ROI_Y_FRACTION = (0.25, 0.65)


def load_first_frame() -> Tuple[np.ndarray, str]:
    frame_paths = sorted(BLUE_DIR.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {BLUE_DIR}")
    frame = cv2.imread(str(frame_paths[90]))
    if frame is None:
        raise RuntimeError(f"Failed to load {frame_paths[90]}")
    return frame, frame_paths[90].name


def detect_blue_bbox(frame_bgr: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    h, w = frame_bgr.shape[:2]
    roi_y0 = int(ROI_Y_FRACTION[0] * h)
    roi_y1 = int(ROI_Y_FRACTION[1] * h)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    base_mask = cv2.inRange(hsv, HSV_BLUE_RANGE[0], HSV_BLUE_RANGE[1])
    # Clip mask to the vertical band so the road (bottom rows) is suppressed.
    mask = np.zeros_like(base_mask)
    mask[roi_y0:roi_y1, :] = base_mask[roi_y0:roi_y1, :]

    # Clean isolated noise, keep compact blobs.
    mask = cv2.medianBlur(mask, 5)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((13, 13), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("Could not find blue region automatically; adjust HSV thresholds or provide manual bbox.")

    frame_area = h * w
    min_area = MIN_AREA_FRAC * frame_area
    max_area = MAX_AREA_FRAC * frame_area

    candidates: list[Tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or area > max_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        aspect = width / max(height, 1)
        # Cars look wider than tall but not extremely elongated; penalize extremes.
        aspect_penalty = math.exp(-((aspect - 1.5) ** 2) / 0.5)

        # Compute average hue closeness to central blue (120°).
        mask_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask_roi, [contour], -1, color=255, thickness=cv2.FILLED)
        hue = hsv[:, :, 0]
        hue_diff = cv2.absdiff(hue, np.full_like(hue, 120))
        mean_hue_diff = cv2.mean(hue_diff, mask=mask_roi)[0]
        hue_score = math.exp(-(mean_hue_diff / 15.0) ** 2)

        score = area * aspect_penalty * hue_score
        candidates.append((score, contour))

    if not candidates:
        raise RuntimeError("Contours found, but none matched expected size/aspect—check thresholds.")

    _, best = max(candidates, key=lambda item: item[0])
    x, y, width, height = cv2.boundingRect(best)

    margin = int(0.05 * max(width, height))
    x0 = max(x - margin, 0)
    y0 = max(y - margin, 0)
    x1 = min(x + width + margin, w - 1)
    y1 = min(y + height + margin, h - 1)
    return (x0, y0), (x1, y1)


def extract_patch(frame: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> np.ndarray:
    x0, y0 = top_left
    x1, y1 = bottom_right
    return frame[y0 : y1 + 1, x0 : x1 + 1]


def compute_ncc_response(image_gray: np.ndarray, template_gray: np.ndarray) -> np.ndarray:
    return cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)


def resize_response_for_display(response: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(response, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)


def compute_hsv_backprojection(frame_bgr: np.ndarray, template_bgr: np.ndarray) -> np.ndarray:
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    template_hsv = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([template_hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    backproj = cv2.calcBackProject([frame_hsv], [0, 1], hist, [0, 180, 0, 256], scale=1)
    backproj_blur = cv2.GaussianBlur(backproj, (31, 31), 0)
    return cv2.normalize(backproj_blur, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)


def highlight_box(frame_bgr: np.ndarray, top_left: Tuple[int, int], bottom_right: Tuple[int, int]) -> np.ndarray:
    vis = frame_bgr.copy()
    cv2.rectangle(vis, top_left, bottom_right, color=(0, 0, 255), thickness=3)
    return vis


def save_heatmap(data: np.ndarray, background: np.ndarray, title: str, filename: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB))
    plt.imshow(data, cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.colorbar(label="score")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()


def main() -> None:
    frame_bgr, frame_name = load_first_frame()
    top_left, bottom_right = detect_blue_bbox(frame_bgr)
    template = extract_patch(frame_bgr, top_left, bottom_right)

    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    ncc = compute_ncc_response(frame_gray, template_gray)
    ncc_display = resize_response_for_display(ncc, frame_gray.shape)
    ncc_display = cv2.normalize(ncc_display, None, 0.0, 1.0, cv2.NORM_MINMAX)

    hsv_prob = compute_hsv_backprojection(frame_bgr, template)
    fused = cv2.normalize(ncc_display * hsv_prob, None, 0.0, 1.0, cv2.NORM_MINMAX)

    vis_bbox = highlight_box(frame_bgr, top_left, bottom_right)

    cv2.imwrite(str(OUTPUT_DIR / f"{frame_name}_bbox.png"), vis_bbox)
    save_heatmap(ncc_display, frame_bgr, "Normalized cross-correlation", OUTPUT_DIR / f"{frame_name}_ncc_overlay.png")
    save_heatmap(hsv_prob, frame_bgr, "HSV backprojection", OUTPUT_DIR / f"{frame_name}_hsv_overlay.png")
    save_heatmap(fused, frame_bgr, "Fused NCC + colour", OUTPUT_DIR / f"{frame_name}_fused_overlay.png")

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ncc)
    print(f"Detected template bbox (x0, y0) -> (x1, y1): {top_left} -> {bottom_right}")
    print(f"Template size: {template.shape[1]} x {template.shape[0]}")
    print(f"Raw NCC max score: {max_val:.4f} at location (x={max_loc[0]}, y={max_loc[1]})")

    fused_flat = fused.flatten()
    topk_idx = np.argsort(fused_flat)[-10:][::-1]
    width = fused.shape[1]
    print("Top 5 fused scores (score, x, y):")
    for idx in topk_idx[:5]:
        y, x = divmod(idx, width)
        print(f"  {fused_flat[idx]:.4f}, {x}, {y}")

    print(f"Artifacts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
