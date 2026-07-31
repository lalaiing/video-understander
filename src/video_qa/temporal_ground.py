from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Literal

import cv2
from google import genai
from google.genai import types
from pydantic import BaseModel, Field


DEFAULT_MODEL = "gemini-3.1-flash-lite"


class GroundingSelection(BaseModel):
    answerable: bool

    support_type: Literal[
        "visual",
        "transcript",
        "both",
        "none",
    ]

    requested_subject_present: bool = Field(
        description=(
            "Whether the exact requested subject or actor is "
            "directly supported by the supplied evidence."
        )
    )

    requested_object_present: bool = Field(
        description=(
            "Whether the exact requested object category is "
            "directly supported. A related or metaphorical object "
            "does not count."
        )
    )

    requested_relation_present: bool = Field(
        description=(
            "Whether the exact requested action or relationship "
            "is directly supported."
        )
    )

    exact_claim_supported: bool = Field(
        description=(
            "True only when the complete proposition in the "
            "question is directly supported without substituting "
            "entities or object categories."
        )
    )

    start_frame_index: int | None = None
    end_frame_index: int | None = None
    start_word_index: int | None = None
    end_word_index: int | None = None

    confidence: Literal["high", "medium", "low"]
    evidence: str


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
            ensure_ascii=False,
        )


def normalize_path(path_value: str, base_directory: Path) -> Path:
    path = Path(path_value)

    if path.is_absolute():
        return path.resolve()

    return (base_directory / path).resolve()


def recursively_find_string(
    payload: Any,
    candidate_keys: set[str],
) -> str | None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized_key = key.lower().strip()

            if (
                normalized_key in candidate_keys
                and isinstance(value, str)
                and value.strip()
            ):
                return value

        for value in payload.values():
            result = recursively_find_string(
                value,
                candidate_keys,
            )

            if result:
                return result

    elif isinstance(payload, list):
        for value in payload:
            result = recursively_find_string(
                value,
                candidate_keys,
            )

            if result:
                return result

    return None


def resolve_video_path(
    index_metadata_path: Path,
    evidence_payload: Any,
    explicit_video_path: Path | None,
) -> Path:
    if explicit_video_path is not None:
        resolved = explicit_video_path.resolve()

        if not resolved.exists():
            raise FileNotFoundError(
                f"Video does not exist: {resolved}"
            )

        return resolved

    metadata = load_json(index_metadata_path)

    candidate_keys = {
        "video_path",
        "file_path",
        "source_video",
        "source_path",
        "input_video",
    }

    video_value = recursively_find_string(
        evidence_payload,
        candidate_keys,
    )

    if not video_value:
        video_value = recursively_find_string(
            metadata,
            candidate_keys,
        )

    if video_value:
        candidates = [
            normalize_path(
                video_value,
                index_metadata_path.parent,
            ),
            normalize_path(
                video_value,
                Path.cwd(),
            ),
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

    manifest_value = recursively_find_string(
        metadata,
        {
            "manifest",
            "manifest_path",
            "source_manifest",
            "scenes_manifest",
        },
    )

    if manifest_value:
        manifest_candidates = [
            normalize_path(
                manifest_value,
                index_metadata_path.parent,
            ),
            normalize_path(
                manifest_value,
                Path.cwd(),
            ),
        ]

        for manifest_path in manifest_candidates:
            if not manifest_path.exists():
                continue

            manifest = load_json(manifest_path)

            video_value = recursively_find_string(
                manifest,
                candidate_keys,
            )

            if video_value:
                video_candidates = [
                    normalize_path(
                        video_value,
                        manifest_path.parent,
                    ),
                    normalize_path(
                        video_value,
                        Path.cwd(),
                    ),
                ]

                for candidate in video_candidates:
                    if candidate.exists():
                        return candidate

    raise FileNotFoundError(
        "Could not resolve the source video. "
        "Supply --video explicitly."
    )


def extract_scene_list(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        candidates = payload
    elif isinstance(payload, dict):
        candidates = []

        for key in (
            "evidence_scenes",
            "retrieved_scenes",
            "scenes",
            "evidence",
        ):
            value = payload.get(key)

            if isinstance(value, list):
                candidates = value
                break
    else:
        candidates = []

    valid_scenes: list[dict[str, Any]] = []

    for scene in candidates:
        if not isinstance(scene, dict):
            continue

        if (
            scene.get("start_ms") is None
            or scene.get("end_ms") is None
        ):
            continue

        valid_scenes.append(scene)

    if not valid_scenes:
        raise ValueError(
            "No evidence scenes with start_ms and end_ms "
            "were found."
        )

    return valid_scenes


def scene_priority(scene: dict[str, Any], fallback: int) -> tuple:
    for key in (
        "retrieval_rank",
        "fused_rank",
        "rank",
    ):
        value = scene.get(key)

        if isinstance(value, int):
            return (0, value)

    score = scene.get("fused_score")

    if isinstance(score, (int, float)):
        return (1, -float(score))

    return (2, fallback)


def deduplicate_scenes(
    scenes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_number: dict[int, dict[str, Any]] = {}

    for index, scene in enumerate(scenes):
        scene_number = int(
            scene.get("scene_number", index)
        )

        existing = by_number.get(scene_number)

        if existing is None:
            by_number[scene_number] = scene
            continue

        if scene_priority(scene, index) < scene_priority(
            existing,
            index,
        ):
            by_number[scene_number] = scene

    return sorted(
        by_number.values(),
        key=lambda item: (
            int(item["start_ms"]),
            int(item["end_ms"]),
        ),
    )


def merge_adjacent_scenes(
    scenes: list[dict[str, Any]],
    maximum_gap_ms: int = 1_000,
    maximum_window_ms: int = 35_000,
) -> list[dict[str, Any]]:
    scenes = deduplicate_scenes(scenes)

    windows: list[dict[str, Any]] = []

    for scene in scenes:
        scene_start = int(scene["start_ms"])
        scene_end = int(scene["end_ms"])
        scene_number = int(
            scene.get("scene_number", len(windows))
        )

        if not windows:
            windows.append(
                {
                    "start_ms": scene_start,
                    "end_ms": scene_end,
                    "scene_numbers": [scene_number],
                    "scenes": [scene],
                }
            )
            continue

        previous = windows[-1]
        proposed_duration = (
            max(previous["end_ms"], scene_end)
            - previous["start_ms"]
        )

        is_adjacent = (
            scene_start
            <= previous["end_ms"] + maximum_gap_ms
        )

        if (
            is_adjacent
            and proposed_duration <= maximum_window_ms
        ):
            previous["end_ms"] = max(
                previous["end_ms"],
                scene_end,
            )
            previous["scene_numbers"].append(scene_number)
            previous["scenes"].append(scene)
        else:
            windows.append(
                {
                    "start_ms": scene_start,
                    "end_ms": scene_end,
                    "scene_numbers": [scene_number],
                    "scenes": [scene],
                }
            )

    return windows


def evenly_spaced_timestamps(
    start_ms: int,
    end_ms: int,
    count: int,
) -> list[int]:
    if end_ms <= start_ms:
        return [start_ms]

    safe_start = start_ms + min(
        100,
        max(0, (end_ms - start_ms) // 10),
    )
    safe_end = end_ms - min(
        100,
        max(0, (end_ms - start_ms) // 10),
    )

    if safe_end <= safe_start:
        return [start_ms]

    count = max(2, count)

    timestamps: list[int] = []

    for index in range(count):
        ratio = index / (count - 1)
        value = round(
            safe_start
            + ratio * (safe_end - safe_start)
        )
        timestamps.append(int(value))

    return sorted(set(timestamps))


def extract_frame(
    capture: cv2.VideoCapture,
    timestamp_ms: int,
) -> bytes:
    capture.set(
        cv2.CAP_PROP_POS_MSEC,
        float(timestamp_ms),
    )

    success, frame = capture.read()

    if not success or frame is None:
        raise RuntimeError(
            f"Could not extract frame at {timestamp_ms} ms."
        )

    success, encoded = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 88],
    )

    if not success:
        raise RuntimeError(
            f"Could not encode frame at {timestamp_ms} ms."
        )

    return encoded.tobytes()


def collect_transcript_words(
    scenes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    collected: list[dict[str, Any]] = []

    for scene in scenes:
        scene_number = scene.get("scene_number")
        words = scene.get("transcript_words")

        if not isinstance(words, list):
            continue

        for word in words:
            if not isinstance(word, dict):
                continue

            text = str(word.get("text", "")).strip()

            if not text:
                continue

            start_ms = word.get("start_ms")
            end_ms = word.get("end_ms")

            if start_ms is None or end_ms is None:
                continue

            collected.append(
                {
                    "index": len(collected),
                    "text": text,
                    "start_ms": int(start_ms),
                    "end_ms": int(end_ms),
                    "scene_number": scene_number,
                }
            )

    return sorted(
        collected,
        key=lambda item: (
            item["start_ms"],
            item["end_ms"],
        ),
    )


def build_grounding_prompt(
    question: str,
    frame_timestamps: list[int],
    transcript_words: list[dict[str, Any]],
) -> str:
    frame_lines = [
        f"FRAME {index}: supplied as the next image"
        for index in range(len(frame_timestamps))
    ]

    if transcript_words:
        word_lines = [
            (
                f"WORD {word['index']}: "
                f"{word['text']!r}"
            )
            for word in transcript_words
        ]
    else:
        word_lines = ["No timestamped transcript words supplied."]

    return f"""
Question:
{question}

You are performing temporal grounding over a short candidate
video window.

Your job is not to answer using outside knowledge. Determine
whether the supplied frames or timestamped transcript words
contain direct evidence for the question.

FRAME INDEXES
{chr(10).join(frame_lines)}

TRANSCRIPT WORD INDEXES
{chr(10).join(word_lines)}

Rules:
1. Select only frame indexes and word indexes that were supplied.
2. Never invent timestamps. The program converts indexes into
   timestamps after your response.
3. For a visible action, use support_type="visual" and select the
   first and last frames that directly show the action.
4. For spoken dialogue, use support_type="transcript" and select
   the first and last words that directly support the answer.
5. Use support_type="both" only when both visual and spoken
   evidence are necessary.
6. If the evidence is insufficient or ambiguous, set
   answerable=false and support_type="none".
7. A frame containing a person does not establish that person's
   name unless the transcript, on-screen text, or surrounding
   evidence establishes the identity.
8. Do not infer actions that occur between widely separated
   frames unless the supplied sequence supports that progression.
9. For questions asking whether one entity performs an action on
   another entity, verify all three independently:
   a. the requested subject is directly visible,
   b. the requested object is directly visible,
   c. the requested action or relationship is directly visible.
10. Do not substitute a related but different category. For example, a person
    is not a vehicle. A rope, platform, suspended person, airborne
    object, or object located above the ground is not automatically
    a flying vehicle.
11. The requested object must visibly match the noun category in
    the question. For example, a flying vehicle should visibly be
    a machine designed for transportation, such as an aircraft,
    helicopter, drone, spacecraft, or other identifiable vehicle.
12. Do not reinterpret objects metaphorically or functionally to
    make the answer true.
13. If the subject is visible and the action is visible, but the
    requested object is missing or belongs to another category,
    set answerable=false and support_type="none".
14. For yes/no questions, answerable=true means that the exact
    proposition in the question is directly supported. Partial
    matches are insufficient.

Before deciding, internally check:

- Requested subject:
- Requested object:
- Requested action:
- Exact object-category match:
- Exact full proposition supported:

If any required component is absent, answerable must be false.
""".strip()


def validate_index_range(
    start_index: int | None,
    end_index: int | None,
    item_count: int,
) -> tuple[int, int] | None:
    if (
        start_index is None
        or end_index is None
        or item_count <= 0
    ):
        return None

    start = max(
        0,
        min(int(start_index), item_count - 1),
    )
    end = max(
        0,
        min(int(end_index), item_count - 1),
    )

    if start > end:
        start, end = end, start

    return start, end


def request_grounding(
    client: genai.Client,
    model: str,
    question: str,
    frame_records: list[dict[str, Any]],
    transcript_words: list[dict[str, Any]],
) -> GroundingSelection:
    prompt = build_grounding_prompt(
        question=question,
        frame_timestamps=[
            item["timestamp_ms"]
            for item in frame_records
        ],
        transcript_words=transcript_words,
    )

    parts: list[types.Part] = [
        types.Part.from_text(text=prompt)
    ]

    for index, frame in enumerate(frame_records):
        parts.append(
            types.Part.from_text(
                text=f"FRAME {index}"
            )
        )
        parts.append(
            types.Part.from_bytes(
                data=frame["bytes"],
                mime_type="image/jpeg",
            )
        )

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=parts,
            )
        ],
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=GroundingSelection,
        ),
    )

    if response.parsed is None:
        raise RuntimeError(
            "Gemini did not return parsed grounding output."
        )

    selection = response.parsed

    if selection.answerable and not selection.exact_claim_supported:
        selection.answerable = False
        selection.support_type = "none"

    if (
        selection.answerable
        and not selection.requested_subject_present
    ):
        selection.answerable = False
        selection.support_type = "none"

    if (
        selection.answerable
        and not selection.requested_object_present
    ):
        selection.answerable = False
        selection.support_type = "none"

    if (
        selection.answerable
        and not selection.requested_relation_present
    ):
        selection.answerable = False
        selection.support_type = "none"

    return selection


def selection_to_timestamps(
    selection: GroundingSelection,
    frame_records: list[dict[str, Any]],
    transcript_words: list[dict[str, Any]],
    candidate_start_ms: int,
    candidate_end_ms: int,
    padding_ms: int = 400,
) -> tuple[int, int] | None:
    if not selection.answerable:
        return None

    starts: list[int] = []
    ends: list[int] = []

    if selection.support_type in {"visual", "both"}:
        frame_range = validate_index_range(
            selection.start_frame_index,
            selection.end_frame_index,
            len(frame_records),
        )

        if frame_range:
            start_index, end_index = frame_range

            starts.append(
                int(
                    frame_records[start_index][
                        "timestamp_ms"
                    ]
                )
            )
            ends.append(
                int(
                    frame_records[end_index][
                        "timestamp_ms"
                    ]
                )
            )

    if selection.support_type in {"transcript", "both"}:
        word_range = validate_index_range(
            selection.start_word_index,
            selection.end_word_index,
            len(transcript_words),
        )

        if word_range:
            start_index, end_index = word_range

            starts.append(
                int(
                    transcript_words[start_index][
                        "start_ms"
                    ]
                )
            )
            ends.append(
                int(
                    transcript_words[end_index][
                        "end_ms"
                    ]
                )
            )

    if not starts or not ends:
        return None

    start_ms = max(
        candidate_start_ms,
        min(starts) - padding_ms,
    )
    end_ms = min(
        candidate_end_ms,
        max(ends) + padding_ms,
    )

    if end_ms <= start_ms:
        end_ms = min(
            candidate_end_ms,
            start_ms + 1_000,
        )

    return int(start_ms), int(end_ms)


def create_frame_records(
    capture: cv2.VideoCapture,
    timestamps: list[int],
    preview_directory: Path,
    prefix: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    preview_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    for index, timestamp_ms in enumerate(timestamps):
        image_bytes = extract_frame(
            capture,
            timestamp_ms,
        )

        preview_path = (
            preview_directory
            / f"{prefix}_{index:02d}_{timestamp_ms}.jpg"
        )

        preview_path.write_bytes(image_bytes)

        records.append(
            {
                "index": index,
                "timestamp_ms": timestamp_ms,
                "path": str(preview_path),
                "bytes": image_bytes,
            }
        )

    return records


def run_fine_grounding(
    client: genai.Client,
    model: str,
    question: str,
    capture: cv2.VideoCapture,
    coarse_selection: GroundingSelection,
    coarse_frames: list[dict[str, Any]],
    transcript_words: list[dict[str, Any]],
    candidate_start_ms: int,
    candidate_end_ms: int,
    preview_directory: Path,
    fine_step_ms: int,
    maximum_fine_frames: int,
) -> tuple[
    GroundingSelection,
    list[dict[str, Any]],
]:
    if coarse_selection.support_type not in {
        "visual",
        "both",
    }:
        return coarse_selection, coarse_frames

    selected_range = validate_index_range(
        coarse_selection.start_frame_index,
        coarse_selection.end_frame_index,
        len(coarse_frames),
    )

    if selected_range is None:
        return coarse_selection, coarse_frames

    start_index, end_index = selected_range

    left_index = max(0, start_index - 1)
    right_index = min(
        len(coarse_frames) - 1,
        end_index + 1,
    )

    fine_start_ms = max(
        candidate_start_ms,
        int(
            coarse_frames[left_index]["timestamp_ms"]
        ),
    )
    fine_end_ms = min(
        candidate_end_ms,
        int(
            coarse_frames[right_index]["timestamp_ms"]
        ),
    )

    if fine_end_ms <= fine_start_ms:
        return coarse_selection, coarse_frames

    requested_count = (
        math.ceil(
            (fine_end_ms - fine_start_ms)
            / max(1, fine_step_ms)
        )
        + 1
    )

    frame_count = max(
        3,
        min(maximum_fine_frames, requested_count),
    )

    fine_timestamps = evenly_spaced_timestamps(
        fine_start_ms,
        fine_end_ms,
        frame_count,
    )

    fine_frames = create_frame_records(
        capture=capture,
        timestamps=fine_timestamps,
        preview_directory=preview_directory,
        prefix="fine",
    )

    fine_selection = request_grounding(
        client=client,
        model=model,
        question=question,
        frame_records=fine_frames,
        transcript_words=transcript_words,
    )

    if not fine_selection.answerable:
        return coarse_selection, coarse_frames

    return fine_selection, fine_frames


def ground_scenes(
    question: str,
    video_path: Path,
    scenes: list[dict[str, Any]],
    model: str = DEFAULT_MODEL,
    output_directory: Path = Path("grounding_output"),
    maximum_windows: int = 3,
    coarse_frame_count: int = 10,
    fine_step_ms: int = 300,
    maximum_fine_frames: int = 18,
) -> dict[str, Any]:
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    windows = merge_adjacent_scenes(scenes)
    windows = windows[:maximum_windows]

    if not windows:
        raise ValueError("No candidate windows were created.")

    client = genai.Client()

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    query_hash = hashlib.sha1(
        question.encode("utf-8")
    ).hexdigest()[:10]

    run_directory = (
        output_directory
        / query_hash
    )

    results: list[dict[str, Any]] = []

    try:
        for window_index, window in enumerate(windows):
            candidate_start_ms = int(
                window["start_ms"]
            )
            candidate_end_ms = int(
                window["end_ms"]
            )

            window_directory = (
                run_directory
                / f"window_{window_index:02d}"
            )

            coarse_timestamps = evenly_spaced_timestamps(
                candidate_start_ms,
                candidate_end_ms,
                coarse_frame_count,
            )

            coarse_frames = create_frame_records(
                capture=capture,
                timestamps=coarse_timestamps,
                preview_directory=window_directory,
                prefix="coarse",
            )

            transcript_words = collect_transcript_words(
                window["scenes"]
            )

            coarse_selection = request_grounding(
                client=client,
                model=model,
                question=question,
                frame_records=coarse_frames,
                transcript_words=transcript_words,
            )

            final_selection, final_frames = (
                run_fine_grounding(
                    client=client,
                    model=model,
                    question=question,
                    capture=capture,
                    coarse_selection=coarse_selection,
                    coarse_frames=coarse_frames,
                    transcript_words=transcript_words,
                    candidate_start_ms=(
                        candidate_start_ms
                    ),
                    candidate_end_ms=candidate_end_ms,
                    preview_directory=window_directory,
                    fine_step_ms=fine_step_ms,
                    maximum_fine_frames=(
                        maximum_fine_frames
                    ),
                )
            )

            refined = selection_to_timestamps(
                selection=final_selection,
                frame_records=final_frames,
                transcript_words=transcript_words,
                candidate_start_ms=(
                    candidate_start_ms
                ),
                candidate_end_ms=candidate_end_ms,
            )

            selected_frame_paths: list[str] = []

            frame_range = validate_index_range(
                final_selection.start_frame_index,
                final_selection.end_frame_index,
                len(final_frames),
            )

            if frame_range:
                start_index, end_index = frame_range

                selected_frame_paths = [
                    str(final_frames[index]["path"])
                    for index in range(
                        start_index,
                        end_index + 1,
                    )
                ]

            result = {
                "window_index": window_index,
                "scene_numbers": (
                    window["scene_numbers"]
                ),
                "candidate_start_ms": (
                    candidate_start_ms
                ),
                "candidate_end_ms": (
                    candidate_end_ms
                ),
                "answerable": (
                    final_selection.answerable
                ),
                "support_type": (
                    final_selection.support_type
                ),
                "confidence": (
                    final_selection.confidence
                ),
                "evidence": final_selection.evidence,
                "grounded_start_ms": (
                    refined[0]
                    if refined
                    else None
                ),
                "grounded_end_ms": (
                    refined[1]
                    if refined
                    else None
                ),
                "selected_frame_paths": (
                    selected_frame_paths
                ),
            }

            results.append(result)

    finally:
        capture.release()

    payload = {
        "question": question,
        "model": model,
        "video_path": str(video_path),
        "windows": results,
    }

    output_path = (
        run_directory
        / "last_grounding.json"
    )

    save_json(output_path, payload)

    payload["output_path"] = str(output_path)

    return payload


def format_ms(milliseconds: int | None) -> str:
    if milliseconds is None:
        return "--:--.---"

    total_seconds = milliseconds / 1_000
    minutes = int(total_seconds // 60)
    seconds = total_seconds - minutes * 60

    return f"{minutes:02d}:{seconds:06.3f}"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refine retrieved scene windows into grounded "
            "timestamp ranges."
        )
    )

    parser.add_argument(
        "question",
        help="Natural-language video question.",
    )

    parser.add_argument(
        "--evidence",
        type=Path,
        default=Path("last_evidence_context.json"),
        help=(
            "Milestone 5 evidence-context JSON. "
            "Default: last_evidence_context.json"
        ),
    )

    parser.add_argument(
        "--index-metadata",
        type=Path,
        required=True,
        help="Path to index_metadata.json.",
    )

    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help=(
            "Optional explicit video path when it cannot be "
            "resolved from metadata."
        ),
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
    )

    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("grounding_output"),
    )

    parser.add_argument(
        "--maximum-windows",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--coarse-frames",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--fine-step-ms",
        type=int,
        default=300,
    )

    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    evidence_path = args.evidence.resolve()
    index_metadata_path = (
        args.index_metadata.resolve()
    )

    if not evidence_path.exists():
        raise FileNotFoundError(
            f"Evidence file not found: {evidence_path}"
        )

    if not index_metadata_path.exists():
        raise FileNotFoundError(
            "Index metadata not found: "
            f"{index_metadata_path}"
        )

    evidence_payload = load_json(evidence_path)
    scenes = extract_scene_list(evidence_payload)

    video_path = resolve_video_path(
        index_metadata_path=index_metadata_path,
        evidence_payload=evidence_payload,
        explicit_video_path=args.video,
    )

    result = ground_scenes(
        question=args.question,
        video_path=video_path,
        scenes=scenes,
        model=args.model,
        output_directory=(
            args.output_directory.resolve()
        ),
        maximum_windows=args.maximum_windows,
        coarse_frame_count=args.coarse_frames,
        fine_step_ms=args.fine_step_ms,
    )

    print()
    print("[complete] Temporal grounding completed")
    print(f"[complete] Model: {result['model']}")
    print(f"[complete] Video: {result['video_path']}")

    for window in result["windows"]:
        print()
        print(
            f"[window {window['window_index']}] "
            f"Scenes {window['scene_numbers']}"
        )

        print(
            "[candidate] "
            f"{format_ms(window['candidate_start_ms'])}"
            " --> "
            f"{format_ms(window['candidate_end_ms'])}"
        )

        if window["answerable"]:
            print(
                "[grounded]  "
                f"{format_ms(window['grounded_start_ms'])}"
                " --> "
                f"{format_ms(window['grounded_end_ms'])}"
            )
        else:
            print("[grounded]  insufficient evidence")

        print(
            f"[support]   {window['support_type']}"
        )
        print(
            f"[confidence] {window['confidence']}"
        )
        print(
            f"[evidence]  {window['evidence']}"
        )

    print()
    print(
        f"[saved] {result['output_path']}"
    )

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())