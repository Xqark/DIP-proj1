"""Reusable NCC + colour fusion tracking utilities."""

from .colour_detection import ColourDetectorConfig, detect_colour_bbox
from .ncc_colour_tracker import (
    SearchWindowConfig,
    TemplateUpdateConfig,
    TrackerConfig,
    TrackingResult,
    NCCColourTracker,
    extract_patch,
)

__all__ = [
    "ColourDetectorConfig",
    "detect_colour_bbox",
    "SearchWindowConfig",
    "TemplateUpdateConfig",
    "TrackerConfig",
    "TrackingResult",
    "NCCColourTracker",
    "extract_patch",
]
