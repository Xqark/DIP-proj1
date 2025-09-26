"""Reusable NCC + colour fusion tracking utilities."""

from .colour_detection import ColourDetectorConfig, build_hsv_mask, detect_colour_bbox
from .fusion import (
    compute_hsv_backprojection,
    compute_ncc_response,
    detect_bbox_from_response,
    resize_response_for_display,
)
from .ncc_colour_tracker import (
    NCCColourTracker,
    SearchWindowConfig,
    TemplateUpdateConfig,
    TrackerConfig,
    TrackingResult,
)
from .sequences import (
    SequenceConfig,
    build_default_sequence_configs,
    extract_template_from_sequence,
    list_frame_paths,
    load_frame,
)
from .types import BoundingBox
from .utils import extract_patch

__all__ = [
    "BoundingBox",
    "ColourDetectorConfig",
    "NCCColourTracker",
    "SearchWindowConfig",
    "SequenceConfig",
    "TemplateUpdateConfig",
    "TrackerConfig",
    "TrackingResult",
    "build_default_sequence_configs",
    "build_hsv_mask",
    "compute_hsv_backprojection",
    "compute_ncc_response",
    "detect_bbox_from_response",
    "detect_colour_bbox",
    "extract_patch",
    "extract_template_from_sequence",
    "list_frame_paths",
    "load_frame",
    "resize_response_for_display",
]
