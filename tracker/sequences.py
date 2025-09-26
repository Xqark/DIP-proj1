from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np

from .colour_detection import ColourDetectorConfig
from .utils import extract_patch


@dataclass(frozen=True)
class SequenceConfig:
    name: str
    directory: Path
    colour_config: ColourDetectorConfig
    default_frame_index: int
    template_bbox: Tuple[int, int, int, int]
    template_frame_index: int = 0


def list_frame_paths(sequence: SequenceConfig) -> list[Path]:
    return sorted(sequence.directory.glob("*.jpg"))


def load_frame(sequence: SequenceConfig, frame_index: int | None) -> tuple[np.ndarray, str]:
    frame_paths = list_frame_paths(sequence)
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {sequence.directory}")

    index = sequence.default_frame_index if frame_index is None else frame_index
    if index < 0 or index >= len(frame_paths):
        raise IndexError(
            f"Frame index {index} out of range for sequence '{sequence.name}' (available: 0..{len(frame_paths) - 1})"
        )

    frame_path = frame_paths[index]
    frame = cv2.imread(str(frame_path))
    if frame is None:
        raise RuntimeError(f"Failed to load frame: {frame_path}")
    return frame, frame_path.name


def extract_template_from_sequence(sequence: SequenceConfig) -> np.ndarray:
    template_frame, _ = load_frame(sequence, sequence.template_frame_index)
    x0, y0, x1, y1 = sequence.template_bbox
    return extract_patch(template_frame, (x0, y0), (x1, y1))


def build_default_sequence_configs(project_root: Path | None = None) -> Dict[str, SequenceConfig]:
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent
    sequences_root = project_root / "sequences"
    return {
        "blue": SequenceConfig(
            name="blue",
            directory=sequences_root / "blue",
            colour_config=ColourDetectorConfig(
                hsv_lower=(100, 80, 70),
                hsv_upper=(135, 255, 255),
                roi_y_fraction=(0.25, 0.65),
                min_area_frac=0.001,
                max_area_frac=0.03,
                target_hue=120,
                median_kernel_size=5,
                open_kernel_size=5,
                close_kernel_size=13,
                aspect_prior=1.5,
                aspect_sigma=0.5,
                margin_ratio=0.05,
            ),
            default_frame_index=0,
            template_bbox=(670, 256, 840, 390),
            template_frame_index=0,
        ),
        "red": SequenceConfig(
            name="red",
            directory=sequences_root / "red",
            colour_config=ColourDetectorConfig(
                hsv_lower=(0, 50, 50),
                hsv_upper=(20, 255, 255),
                extra_ranges=(((160, 70, 60), (180, 255, 255)),),
                roi_y_fraction=(0.2, 0.75),
                roi_x_fraction=(0.2, 0.85),
                min_area_frac=0.001,
                max_area_frac=0.04,
                target_hue=0,
                median_kernel_size=5,
                open_kernel_size=5,
                close_kernel_size=11,
                aspect_prior=1.4,
                aspect_sigma=0.4,
                margin_ratio=0.05,
            ),
            default_frame_index=0,
            template_bbox=(795, 275, 990, 413),
            template_frame_index=0,
        ),
    }


__all__ = [
    "SequenceConfig",
    "build_default_sequence_configs",
    "extract_template_from_sequence",
    "list_frame_paths",
    "load_frame",
]
