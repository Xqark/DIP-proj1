from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tracker import (  # noqa: E402
    NCCColourTracker,
    SequenceConfig,
    TrackerConfig,
    build_default_sequence_configs,
    compute_hsv_backprojection,
    compute_ncc_response,
    detect_bbox_from_response,
    extract_template_from_sequence,
    load_frame,
    resize_response_for_display,
)


@dataclass(frozen=True)
class SequenceRunConfig:
    sequence: SequenceConfig
    initial_frame_index: int
    draw_color: Tuple[int, int, int]
    label: str

    @property
    def name(self) -> str:
        return self.sequence.name


@dataclass(slots=True)
class FrameLog:
    frame_index: int
    frame_name: str
    x0: int
    y0: int
    x1: int
    y1: int
    ncc_score: float
    fused_score: float


def annotate_frame(
    frame_bgr: np.ndarray,
    bbox: Tuple[Tuple[int, int], Tuple[int, int]],
    colour: Tuple[int, int, int],
    text: str,
) -> np.ndarray:
    vis = frame_bgr.copy()
    cv2.rectangle(vis, bbox[0], bbox[1], colour, thickness=3)
    cv2.putText(
        vis,
        text,
        (bbox[0][0], max(bbox[0][1] - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        colour,
        2,
        cv2.LINE_AA,
    )
    return vis


def run_sequence(
    config: SequenceRunConfig,
    tracker_config: TrackerConfig,
    *,
    output_dir: Path,
    write_video: bool,
    video_fps: float,
) -> None:
    sequence = config.sequence
    frame_paths = sorted(sequence.directory.glob("*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {sequence.directory}")

    if config.initial_frame_index >= len(frame_paths):
        raise IndexError(
            f"Initial frame index {config.initial_frame_index} out of range for {config.name} sequence"
        )

    initial_frame, initial_name = load_frame(sequence, config.initial_frame_index)

    template_bgr = extract_template_from_sequence(sequence)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)

    initial_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    ncc = compute_ncc_response(initial_gray, template_gray)
    ncc_display = resize_response_for_display(ncc, initial_gray.shape)
    ncc_display = cv2.normalize(ncc_display, None, 0.0, 1.0, cv2.NORM_MINMAX)

    hsv_prob = compute_hsv_backprojection(initial_frame, template_bgr, sequence.colour_config)
    fused = cv2.normalize(ncc_display * hsv_prob, None, 0.0, 1.0, cv2.NORM_MINMAX)

    initial_top_left, initial_bottom_right = detect_bbox_from_response(fused, template_gray.shape)
    initial_bbox = (initial_top_left, initial_bottom_right)

    _, max_val, _, _ = cv2.minMaxLoc(ncc)
    _, fused_max_val, _, _ = cv2.minMaxLoc(fused)

    tracker = NCCColourTracker(
        initial_frame,
        initial_bbox,
        tracker_config,
        colour_config=sequence.colour_config,
    )

    logs: List[FrameLog] = []

    output_dir.mkdir(parents=True, exist_ok=True)

    video_writer: cv2.VideoWriter | None = None
    if write_video:
        video_path = output_dir / f"{config.name}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width = initial_frame.shape[:2]
        video_writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (width, height))
        if not video_writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {video_path}")

    initial_log = FrameLog(
        frame_index=config.initial_frame_index,
        frame_name=initial_name,
        x0=initial_top_left[0],
        y0=initial_top_left[1],
        x1=initial_bottom_right[0],
        y1=initial_bottom_right[1],
        ncc_score=float(max_val),
        fused_score=float(fused_max_val),
    )
    logs.append(initial_log)

    overlay_text = f"{config.label} fused={initial_log.fused_score:.3f}"
    annotated_initial = annotate_frame(initial_frame, initial_bbox, config.draw_color, overlay_text)
    if video_writer is not None:
        video_writer.write(annotated_initial)

    for frame_idx in range(config.initial_frame_index + 1, len(frame_paths)):
        frame, frame_name = load_frame(sequence, frame_idx)
        result = tracker.track(frame)

        log = FrameLog(
            frame_index=frame_idx,
            frame_name=frame_name,
            x0=result.top_left[0],
            y0=result.top_left[1],
            x1=result.bottom_right[0],
            y1=result.bottom_right[1],
            ncc_score=result.ncc_score,
            fused_score=result.fused_score,
        )
        logs.append(log)

        overlay_text = f"{config.label} fused={result.fused_score:.3f}"
        annotated = annotate_frame(frame, (result.top_left, result.bottom_right), config.draw_color, overlay_text)
        if video_writer is not None:
            video_writer.write(annotated)

    if video_writer is not None:
        video_writer.release()

    csv_path = output_dir / f"{config.name}_track_log.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["frame_index", "frame_name", "x0", "y0", "x1", "y1", "ncc_score", "fused_score"])
        for entry in logs:
            writer.writerow(
                [
                    entry.frame_index,
                    entry.frame_name,
                    entry.x0,
                    entry.y0,
                    entry.x1,
                    entry.y1,
                    f"{entry.ncc_score:.6f}",
                    f"{entry.fused_score:.6f}",
                ]
            )

    fused_scores = np.array([entry.fused_score for entry in logs], dtype=float)
    if fused_scores.size:
        print(
            f"Sequence {config.name}: frames={len(logs)}, fused min={float(fused_scores.min()):.3f}, "
            f"median={float(np.median(fused_scores)):.3f}, max={float(fused_scores.max()):.3f}"
        )


def build_sequence_configs(project_root: Path) -> dict[str, SequenceRunConfig]:
    base_sequences = build_default_sequence_configs(project_root)
    blue = base_sequences["blue"]
    red = base_sequences["red"]

    return {
        "blue": SequenceRunConfig(
            sequence=blue,
            initial_frame_index=90,
            draw_color=(0, 0, 255),
            label="Blue car",
        ),
        "red": SequenceRunConfig(
            sequence=red,
            initial_frame_index=red.default_frame_index,
            draw_color=(0, 255, 255),
            label="Red car",
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate NCC + colour tracker on vehicle sequences.")
    parser.add_argument(
        "--sequence",
        choices=["blue", "red", "both"],
        default="both",
        help="Select which sequence to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis") / "evaluation",
        help="Directory where evaluation artifacts are stored.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=15.0,
        help="Frame rate for the annotated video output.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable annotated video generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    sequence_configs = build_sequence_configs(project_root)

    if args.sequence == "both":
        selected = [sequence_configs["blue"], sequence_configs["red"]]
    else:
        selected = [sequence_configs[args.sequence]]

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tracker_config = TrackerConfig()

    for cfg in selected:
        run_sequence(
            cfg,
            tracker_config,
            output_dir=output_dir / cfg.name,
            write_video=not args.no_video,
            video_fps=args.video_fps,
        )


if __name__ == "__main__":
    main()
