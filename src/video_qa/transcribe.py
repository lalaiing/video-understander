from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel


def seconds_to_ms(seconds: float) -> int:
    """Convert seconds to integer milliseconds."""
    return round(seconds * 1000)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def transcribe_video(
    video_path: Path,
    model_name: str,
    device: str,
    compute_type: str,
    language: str | None,
) -> dict[str, Any]:
    """
    Transcribe a video and return segment-level and word-level timestamps.
    """

    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    print(f"[model] Loading Whisper model: {model_name}")
    print(f"[model] Device: {device}")
    print(f"[model] Compute type: {compute_type}")

    model = WhisperModel(
        model_name,
        device=device,
        compute_type=compute_type,
    )

    print(f"[asr] Transcribing: {video_path}")


    segments_generator, info = model.transcribe(
        str(video_path),
        language=language,
        beam_size=5,
        word_timestamps=True,

        # Remove long non-speech regions, but keep padding around
        # short dialogue mixed with effects and music.
        vad_filter=False,
        vad_parameters={
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 400,
        },

        # Prevent one recognized sentence from repeatedly seeding
        # later transcription windows.
        condition_on_previous_text=True,

        # Help reject hallucinated speech across long silent gaps.
        hallucination_silence_threshold=2.0,

        # hotwords=(
        #     "Thom Celia Barley Vivacissimo "
        #     "robot hand robotics memory override"
        # ),
    )

    segments: list[dict[str, Any]] = []
    all_words: list[dict[str, Any]] = []

    # faster-whisper returns a generator. Iterating over it performs
    # the actual transcription.
    for segment in segments_generator:
        segment_words: list[dict[str, Any]] = []

        for word in segment.words or []:
            word_data = {
                "start_ms": seconds_to_ms(word.start),
                "end_ms": seconds_to_ms(word.end),
                # Keep Whisper's original spacing. This helps reconstruct
                # sentences naturally using "".join(...).
                "text": word.word,
                "probability": round(float(word.probability), 4),
            }

            segment_words.append(word_data)
            all_words.append(word_data)

        segment_data = {
            "segment_id": int(segment.id),
            "start_ms": seconds_to_ms(segment.start),
            "end_ms": seconds_to_ms(segment.end),
            "text": segment.text.strip(),
            "average_log_probability": round(
                float(segment.avg_logprob),
                4,
            ),
            "no_speech_probability": round(
                float(segment.no_speech_prob),
                4,
            ),
            "words": segment_words,
        }

        segments.append(segment_data)

        start_display = format_timestamp(segment.start)
        end_display = format_timestamp(segment.end)

        print(
            f"[{start_display} --> {end_display}] "
            f"{segment.text.strip()}"
        )

    detected_language = getattr(info, "language", None)
    language_probability = getattr(
        info,
        "language_probability",
        None,
    )

    duration_ms = 0

    if all_words:
        duration_ms = all_words[-1]["end_ms"]
    elif segments:
        duration_ms = segments[-1]["end_ms"]

    result = {
        "video_path": str(video_path.resolve()),
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "detected_language": detected_language,
        "language_probability": (
            round(float(language_probability), 4)
            if language_probability is not None
            else None
        ),
        "duration_ms": duration_ms,
        "segment_count": len(segments),
        "word_count": len(all_words),
        "segments": segments,
        "words": all_words,
    }

    return result


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    milliseconds = round(seconds * 1000)

    hours = milliseconds // 3_600_000
    milliseconds %= 3_600_000

    minutes = milliseconds // 60_000
    milliseconds %= 60_000

    whole_seconds = milliseconds // 1000
    remaining_ms = milliseconds % 1000

    return (
        f"{hours:02d}:"
        f"{minutes:02d}:"
        f"{whole_seconds:02d}."
        f"{remaining_ms:03d}"
    )


def assign_words_to_scenes(
    manifest: dict[str, Any],
    words: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Assign each word to exactly one scene using the word's midpoint.

    Example:
        Word: 4.0s to 4.4s
        Midpoint: 4.2s

    The word is assigned to the scene containing 4.2s.
    """

    enriched_manifest = copy.deepcopy(manifest)
    scenes = enriched_manifest.get("scenes", [])

    if not scenes:
        raise ValueError("The scene manifest contains no scenes.")

    scenes.sort(key=lambda scene: scene["start_ms"])

    words_per_scene: dict[str, list[dict[str, Any]]] = {
        scene["scene_id"]: []
        for scene in scenes
    }

    scene_index = 0
    unassigned_words: list[dict[str, Any]] = []

    for word in words:
        midpoint_ms = (
            word["start_ms"] + word["end_ms"]
        ) // 2

        # Move forward through the scenes as word timestamps increase.
        while (
            scene_index < len(scenes) - 1
            and midpoint_ms >= scenes[scene_index]["end_ms"]
        ):
            scene_index += 1

        current_scene = scenes[scene_index]

        is_last_scene = scene_index == len(scenes) - 1

        inside_scene = (
            current_scene["start_ms"]
            <= midpoint_ms
            < current_scene["end_ms"]
        )

        # Include the final boundary in the final scene.
        if is_last_scene:
            inside_scene = (
                current_scene["start_ms"]
                <= midpoint_ms
                <= current_scene["end_ms"]
            )

        if inside_scene:
            words_per_scene[current_scene["scene_id"]].append(
                word
            )
        else:
            unassigned_words.append(word)

    scenes_with_speech = 0

    for scene in scenes:
        scene_words = words_per_scene[scene["scene_id"]]

        transcript_text = "".join(
            word["text"] for word in scene_words
        ).strip()

        scene["transcript"] = transcript_text
        scene["transcript_words"] = scene_words
        scene["word_count"] = len(scene_words)
        scene["has_speech"] = bool(scene_words)

        if scene_words:
            scenes_with_speech += 1

            scene["speech_start_ms"] = scene_words[0][
                "start_ms"
            ]
            scene["speech_end_ms"] = scene_words[-1][
                "end_ms"
            ]
        else:
            scene["speech_start_ms"] = None
            scene["speech_end_ms"] = None

    enriched_manifest["transcription_summary"] = {
        "total_words": len(words),
        "assigned_words": len(words) - len(unassigned_words),
        "unassigned_words": len(unassigned_words),
        "scenes_with_speech": scenes_with_speech,
        "scenes_without_speech": (
            len(scenes) - scenes_with_speech
        ),
    }

    if unassigned_words:
        enriched_manifest["unassigned_transcript_words"] = (
            unassigned_words
        )

    return enriched_manifest
   

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe a video and attach timestamped words "
            "to detected scenes."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to the scenes.json file.",
    )

    parser.add_argument(
        "--model",
        default="small",
        help=(
            "Whisper model name, for example: "
            "tiny, base, small, medium, or large-v3."
        ),
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Inference device.",
    )

    parser.add_argument(
        "--compute-type",
        default="int8",
        help=(
            "CTranslate2 compute type. "
            "Use int8 for CPU or float16 for CUDA."
        ),
    )

    parser.add_argument(
        "--language",
        default=None,
        help=(
            "Optional language code such as en or id. "
            "Leave blank for automatic detection."
        ),
    )

    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)

    raw_video_path = manifest.get("file_path")

    if not raw_video_path:
        raise ValueError(
            "The scene manifest has no 'file_path' field."
        )

    video_path = Path(raw_video_path)

    if not video_path.is_absolute():
        video_path = (
            manifest_path.parent / video_path
        ).resolve()

    transcription = transcribe_video(
        video_path=video_path,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )

    output_directory = manifest_path.parent

    transcription_path = (
        output_directory / "transcription.json"
    )

    enriched_manifest_path = (
        output_directory / "scenes_with_transcript.json"
    )

    save_json(
        transcription_path,
        transcription,
    )

    enriched_manifest = assign_words_to_scenes(
        manifest=manifest,
        words=transcription["words"],
    )

    enriched_manifest["transcription"] = {
        "model": transcription["model"],
        "detected_language": (
            transcription["detected_language"]
        ),
        "language_probability": (
            transcription["language_probability"]
        ),
        "transcription_file": str(transcription_path),
    }

    save_json(
        enriched_manifest_path,
        enriched_manifest,
    )

    summary = enriched_manifest["transcription_summary"]

    print()
    print("[complete] Transcription completed")
    print(
        f"[complete] Language: "
        f"{transcription['detected_language']}"
    )
    print(
        f"[complete] Words: "
        f"{transcription['word_count']}"
    )
    print(
        f"[complete] Scenes with speech: "
        f"{summary['scenes_with_speech']}"
    )
    print(
        f"[complete] Unassigned words: "
        f"{summary['unassigned_words']}"
    )
    print(
        f"[output] {transcription_path}"
    )
    print(
        f"[output] {enriched_manifest_path}"
    )