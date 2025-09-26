from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracker import (  # noqa: E402
    build_default_sequence_configs,
    compute_hsv_backprojection,
    compute_ncc_response,
    detect_bbox_from_response,
    extract_template_from_sequence,
    load_frame,
    resize_response_for_display,
)
OUTPUT_DIR = PROJECT_ROOT / "analysis"
OUTPUT_DIR.mkdir(exist_ok=True)

SEQUENCE_CONFIGS = build_default_sequence_configs(PROJECT_ROOT)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse NCC + colour cues for a vehicle sequence.")
    parser.add_argument(
        "--sequence",
        choices=sorted(SEQUENCE_CONFIGS.keys()),
        default="blue",
        help="Which sequence to analyse.",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=None,
        help="Optional frame index to analyse (defaults to sequence's tuned starting frame).",
    )
    parser.add_argument(
        "--use-padding",
        action="store_true",
        help="Use zero-padding instead of interpolation for NCC response display.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence = SEQUENCE_CONFIGS[args.sequence]

    # Load the current frame to analyze
    frame_bgr, frame_name = load_frame(sequence, args.frame_index)
    frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    
    # Extract template from reference frame using ground truth bbox
    template = extract_template_from_sequence(sequence)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # Compute NCC response between template and current frame
    ncc = compute_ncc_response(frame_gray, template_gray)
    ncc_display = resize_response_for_display(ncc, frame_gray.shape, use_padding=args.use_padding)
    ncc_display = cv2.normalize(ncc_display, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Compute HSV backprojection using the template
    hsv_prob = compute_hsv_backprojection(frame_bgr, template, sequence.colour_config)
    
    # Fuse NCC and HSV responses
    fused = cv2.normalize(ncc_display * hsv_prob, None, 0.0, 1.0, cv2.NORM_MINMAX)

    # Detect bounding box in current frame based on fusion results
    detected_top_left, detected_bottom_right = detect_bbox_from_response(fused, template_gray.shape)
    vis_bbox = highlight_box(frame_bgr, detected_top_left, detected_bottom_right)

    # Save visualizations
    sequence_output_dir = OUTPUT_DIR / sequence.name
    sequence_output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(sequence_output_dir / f"{frame_name}_bbox.png"), vis_bbox)
    cv2.imwrite(str(sequence_output_dir / f"{frame_name}_template.png"), template)
    save_heatmap(
        ncc_display,
        frame_bgr,
        "Normalized cross-correlation",
        sequence_output_dir / f"{frame_name}_ncc_overlay.png",
    )
    save_heatmap(
        hsv_prob,
        frame_bgr,
        "HSV backprojection",
        sequence_output_dir / f"{frame_name}_hsv_overlay.png",
    )
    save_heatmap(
        fused,
        frame_bgr,
        "Fused NCC + colour",
        sequence_output_dir / f"{frame_name}_fused_overlay.png",
    )

    # Print analysis results
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(ncc)
    _, fused_max_val, _, fused_max_loc = cv2.minMaxLoc(fused)
    
    print(f"Sequence: {sequence.name}")
    print(f"Frame: {frame_name}")
    print(f"Template extracted from frame {sequence.template_frame_index} at bbox {sequence.template_bbox}")
    print(f"Template size: {template.shape[1]} x {template.shape[0]}")
    print(f"NCC display method: {'Zero-padding' if args.use_padding else 'Interpolation'}")
    print(f"Raw NCC max score: {max_val:.4f} at location (x={max_loc[0]}, y={max_loc[1]})")
    print(f"Fused max score: {fused_max_val:.4f} at location (x={fused_max_loc[0]}, y={fused_max_loc[1]})")
    print(f"Detected bbox in current frame: {detected_top_left} -> {detected_bottom_right}")

    fused_flat = fused.flatten()
    topk_idx = np.argsort(fused_flat)[-10:][::-1]
    width = fused.shape[1]
    print("Top 5 fused scores (score, x, y):")
    for idx in topk_idx[:5]:
        y, x = divmod(idx, width)
        print(f"  {fused_flat[idx]:.4f}, {x}, {y}")

    print(f"Artifacts saved to: {sequence_output_dir}")


if __name__ == "__main__":
    main()
