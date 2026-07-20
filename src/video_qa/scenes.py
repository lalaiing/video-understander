from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
from scenedetect import ContentDetector, SceneManager, open_video


def create_video_id(video_path: Path) -> str:
    """Create a repeatable short identifier from the absolute file path."""
    resolved = str(video_path.resolve()).encode("utf-8")
    return hashlib.sha256(resolved).hexdigest()[:12]


def extract_keyframe(
    capture: cv2.VideoCapture,
    timestamp_ms: int,
    output_path: Path,
) -> None:
    """Extract a frame at the requested timestamp."""
    capture.set(cv2.CAP_PROP_POS_MSEC, timestamp_ms)

    success, frame = capture.read()
    if not success or frame is None:
        raise RuntimeError(
            f"Could not read frame at {timestamp_ms} ms."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not cv2.imwrite(str(output_path), frame):
        raise RuntimeError(f"Could not write keyframe to {output_path}.")


def detect_video_scenes(
    video_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    video_id = create_video_id(video_path)
    video_output_dir = output_root / video_id
    keyframe_dir = video_output_dir / "keyframes"

    video_output_dir.mkdir(parents=True, exist_ok=True)
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    video = open_video(str(video_path))

    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(
        video=video,
        show_progress=True,
    )

    # start_in_scene=True ensures that a video with no detected cuts
    # still produces one scene covering the entire video.
    scene_pairs = scene_manager.get_scene_list(start_in_scene=True)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video_path}")

    scenes: list[dict[str, Any]] = []

    try:
        for scene_number, (start_time, end_time) in enumerate(scene_pairs):
            start_ms = round(start_time.get_seconds() * 1000)
            end_ms = round(end_time.get_seconds() * 1000)
            midpoint_ms = (start_ms + end_ms) // 2

            keyframe_name = f"scene_{scene_number:04d}.jpg"
            keyframe_path = keyframe_dir / keyframe_name

            extract_keyframe(
                capture=capture,
                timestamp_ms=midpoint_ms,
                output_path=keyframe_path,
            )

            scenes.append(
                {
                    "scene_id": f"{video_id}:{scene_number:04d}",
                    "scene_number": scene_number,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "duration_ms": end_ms - start_ms,
                    "keyframe_ms": midpoint_ms,
                    "keyframe_path": str(keyframe_path),
                }
            )
    finally:
        capture.release()

    manifest = {
        "video_id": video_id,
        "file_path": str(video_path.resolve()),
        "scene_count": len(scenes),
        "scenes": scenes,
    }

    manifest_path = video_output_dir / "scenes.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detect video scenes and extract keyframes."
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to the input MP4.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed"),
        help="Directory for processed video data.",
    )

    args = parser.parse_args()

    manifest = detect_video_scenes(
        video_path=args.video,
        output_root=args.output,
    )

    print(f"Video ID: {manifest['video_id']}")
    print(f"Scenes detected: {manifest['scene_count']}")
    print(
        "Manifest: "
        f"{args.output / manifest['video_id'] / 'scenes.json'}"
    )