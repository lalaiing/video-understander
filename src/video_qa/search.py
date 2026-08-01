from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer


STREAM_NAMES = (
    "caption",
    "transcript",
    "keyframe",
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = path.with_suffix(path.suffix + ".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    temporary_path.replace(path)


def resolve_existing_path(
    raw_path: str,
    reference_directory: Path,
) -> Path:
    candidate = Path(raw_path)

    possible_paths = [
        candidate,
        Path.cwd() / candidate,
        reference_directory / candidate,
    ]

    for possible_path in possible_paths:
        resolved = possible_path.resolve()

        if resolved.exists():
            return resolved

    attempted = "\n".join(
        f"  - {path.resolve()}"
        for path in possible_paths
    )

    raise FileNotFoundError(
        f"Could not locate: {raw_path}\n"
        f"Attempted:\n{attempted}"
    )


def format_timestamp(milliseconds: int) -> str:
    total_seconds, remaining_ms = divmod(milliseconds, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    if hours:
        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}."
            f"{remaining_ms:03d}"
        )

    return (
        f"{minutes:02d}:"
        f"{seconds:02d}."
        f"{remaining_ms:03d}"
    )


def create_query_vectors(
    question: str,
    text_model: SentenceTransformer,
    frame_model: SentenceTransformer,
) -> dict[str, list[float]]:
    """
    Create two query embeddings:

    1. A semantic text vector for caption/transcript search.
    2. A CLIP text vector for keyframe image search.
    """

    text_vector = text_model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    visual_vector = frame_model.encode(
        question,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return {
        "caption": text_vector.tolist(),
        "transcript": text_vector.tolist(),
        "keyframe": visual_vector.tolist(),
    }


def query_vector_stream(
    client: QdrantClient,
    collection_name: str,
    vector_name: str,
    query_vector: list[float],
    limit: int,
) -> list[Any]:
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        using=vector_name,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    return list(response.points)


def is_visual_location_question(question: str) -> bool:
    """Return True for questions asking where a visible event occurs."""
    normalized = " ".join(question.lower().split())

    location_markers = (
        "where",
        "location",
        "located",
        "standing",
        "setting",
        "place",
        "room",
        "building",
    )

    return any(
        marker in normalized
        for marker in location_markers
    )


def choose_stream_weights(question: str) -> dict[str, float]:
    normalized = " ".join(question.lower().split())

    location_markers = (
        "where",
        "location",
        "located",
        "standing",
        "setting",
        "place",
        "room",
        "building",
    )

    fear_markers = (
        "frighten",
        "frightened",
        "afraid",
        "scared",
        "fear",
        "freaked out",
        "nightmare",
    )

    dialogue_markers = (
        "conversation",
        "say",
        "says",
        "said",
        "when does",
        "tell",
        "tells",
        "explains",
        "speak",
        "speaks",
        "talk",
        "discuss",
        "mention",
        "ask",
        "quote",
    )

    if is_visual_location_question(question):
        # Named visual-location questions need caption semantics to
        # establish the event/people before CLIP judges the setting.
        # Overweighting CLIP tends to retrieve generic rooms containing
        # arbitrary people.
        return {
            "caption": 1.8,
            "transcript": 0.8,
            "keyframe": 1.0,
        }

    if any(marker in normalized for marker in fear_markers):
        return {
            "caption": 0.6,
            "transcript": 1.9,
            "keyframe": 0.3,
        }

    if any(marker in normalized for marker in dialogue_markers):
        return {
            "caption": 0.4,
            "transcript": 2.2,
            "keyframe": 0.2,
        }

    return {
        "caption": 1.0,
        "transcript": 1.0,
        "keyframe": 1.0,
    }


def reciprocal_rank_fusion(
    stream_results: dict[str, list[Any]],
    rrf_k: int,
    stream_weights: dict[str, float],
) -> list[dict[str, Any]]:
    """
    Similarity-aware weighted Reciprocal Rank Fusion.

    Scores are normalized independently within each retrieval
    stream before being combined. This prevents weak results
    appearing in multiple streams from automatically beating a
    strong result from one stream.
    """

    fused_by_scene: dict[str, dict[str, Any]] = {}

    for stream_name, hits in stream_results.items():
        if not hits:
            continue

        raw_scores = [
            float(hit.score)
            for hit in hits
        ]

        minimum_score = min(raw_scores)
        maximum_score = max(raw_scores)
        score_range = maximum_score - minimum_score

        stream_weight = stream_weights.get(
            stream_name,
            1.0,
        )

        for rank, hit in enumerate(hits, start=1):
            payload = dict(hit.payload or {})

            scene_id = payload.get(
                "scene_id",
                str(hit.id),
            )

            raw_score = float(hit.score)

            if score_range > 1e-9:
                normalized_score = (
                    raw_score - minimum_score
                ) / score_range
            else:
                normalized_score = 1.0

            contribution = (
                stream_weight
                * normalized_score
                / (rrf_k + rank)
            )

            if scene_id not in fused_by_scene:
                fused_by_scene[scene_id] = {
                    "scene_id": scene_id,
                    "point_id": str(hit.id),
                    "rrf_score": 0.0,
                    "ranks": {},
                    "similarities": {},
                    "normalized_similarities": {},
                    "contributions": {},
                    "payload": payload,
                }

            fused_result = fused_by_scene[scene_id]

            fused_result["rrf_score"] += contribution

            fused_result["ranks"][stream_name] = rank
            fused_result["similarities"][stream_name] = (
                raw_score
            )

            fused_result[
                "normalized_similarities"
            ][stream_name] = normalized_score

            fused_result["contributions"][
                stream_name
            ] = contribution

    fused_results = list(fused_by_scene.values())

    fused_results.sort(
        key=lambda result: (
            -result["rrf_score"],
            min(result["ranks"].values()),
            result["payload"].get(
                "scene_number",
                1_000_000,
            ),
        )
    )

    return fused_results


def display_stream_results(
    stream_results: dict[str, list[Any]],
) -> None:
    print()
    print("=" * 72)
    print("INDIVIDUAL VECTOR RESULTS")
    print("=" * 72)

    for stream_name in STREAM_NAMES:
        print()
        print(f"[{stream_name.upper()}]")

        hits = stream_results.get(stream_name, [])

        for rank, hit in enumerate(hits, start=1):
            payload = hit.payload or {}

            scene_number = payload.get(
                "scene_number",
                "?",
            )

            start_ms = payload.get("start_ms", 0)
            end_ms = payload.get("end_ms", 0)

            print(
                f"  {rank:>2}. "
                f"Scene {scene_number} "
                f"[{format_timestamp(start_ms)}"
                f" --> {format_timestamp(end_ms)}] "
                f"similarity={float(hit.score):.4f}"
            )


def display_fused_results(
    question: str,
    fused_results: list[dict[str, Any]],
    top_k: int,
) -> None:
    print()
    print("=" * 72)
    print("FUSED RESULTS")
    print("=" * 72)
    print(f"Question: {question}")
    print()

    for result_position, result in enumerate(
        fused_results[:top_k],
        start=1,
    ):
        payload = result["payload"]

        start_ms = int(payload.get("start_ms", 0))
        end_ms = int(payload.get("end_ms", 0))

        caption = (
            payload.get("caption", "").strip()
            or "[No visual caption]"
        )

        transcript = (
            payload.get("transcript", "").strip()
            or "[No spoken dialogue]"
        )

        keyframe_path = payload.get(
            "keyframe_path",
            "[No keyframe path]",
        )

        rank_description = ", ".join(
            f"{stream}=#{result['ranks'][stream]}"
            for stream in STREAM_NAMES
            if stream in result["ranks"]
        )

        similarity_description = ", ".join(
            f"{stream}="
            f"{result['similarities'][stream]:.4f}"
            for stream in STREAM_NAMES
            if stream in result["similarities"]
        )

        print(
            f"{result_position}. "
            f"Scene {payload.get('scene_number', '?')} "
            f"[{format_timestamp(start_ms)}"
            f" --> {format_timestamp(end_ms)}]"
        )

        print(
            f"   RRF score: "
            f"{result['rrf_score']:.6f}"
        )

        print(f"   Ranks: {rank_description}")
        print(f"   Similarities: {similarity_description}")
        print(f"   Caption: {caption}")
        print(f"   Transcript: {transcript}")
        print(f"   Keyframe: {keyframe_path}")
        print()


def build_serializable_output(
    question: str,
    fused_results: list[dict[str, Any]],
    top_k: int,
    per_stream: int,
    rrf_k: int,
    index_metadata_path: Path,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []

    for position, result in enumerate(
        fused_results[:top_k],
        start=1,
    ):
        payload = result["payload"]

        results.append(
            {
                "result_position": position,
                "scene_id": result["scene_id"],
                "point_id": result["point_id"],
                "rrf_score": round(
                    result["rrf_score"],
                    8,
                ),
                "ranks": result["ranks"],
                "similarities": {
                    stream: round(score, 6)
                    for stream, score
                    in result["similarities"].items()
                },
                "video_id": payload.get("video_id"),
                "scene_number": payload.get(
                    "scene_number"
                ),
                "start_ms": payload.get("start_ms"),
                "end_ms": payload.get("end_ms"),
                "keyframe_ms": payload.get(
                    "keyframe_ms"
                ),
                "caption": payload.get("caption", ""),
                "transcript": payload.get(
                    "transcript",
                    "",
                ),
                "keyframe_path": payload.get(
                    "keyframe_path",
                ),
            }
        )

    return {
        "question": question,
        "retrieval": {
            "method": "reciprocal_rank_fusion",
            "per_stream_limit": per_stream,
            "top_k": top_k,
            "rrf_k": rrf_k,
            "streams": list(STREAM_NAMES),
        },
        "index_metadata_path": str(
            index_metadata_path
        ),
        "results": results,
    }


def open_keyframe(
    result: dict[str, Any],
    reference_directory: Path,
) -> None:
    raw_keyframe_path = result["payload"].get(
        "keyframe_path"
    )

    if not raw_keyframe_path:
        raise ValueError(
            "The selected result has no keyframe path."
        )

    keyframe_path = resolve_existing_path(
        raw_path=raw_keyframe_path,
        reference_directory=reference_directory,
    )

    print(f"[open] {keyframe_path}")

    os.startfile(keyframe_path)  # Windows only


def preview_video_scene(
    result: dict[str, Any],
    metadata_directory: Path,
) -> None:
    ffplay_path = shutil.which("ffplay")

    if not ffplay_path:
        raise RuntimeError(
            "ffplay was not found in PATH. "
            "Install a full FFmpeg build or skip --preview-rank."
        )

    manifest_path = (
        metadata_directory / "scenes_with_captions.json"
    )

    manifest = load_json(manifest_path)

    raw_video_path = manifest.get("file_path")

    if not raw_video_path:
        raise ValueError(
            "The caption manifest contains no file_path."
        )

    video_path = resolve_existing_path(
        raw_path=raw_video_path,
        reference_directory=metadata_directory,
    )

    payload = result["payload"]

    start_ms = int(payload["start_ms"])
    end_ms = int(payload["end_ms"])

    start_seconds = max(0, start_ms / 1000)
    duration_seconds = max(
        0.1,
        (end_ms - start_ms) / 1000,
    )

    print(
        f"[preview] {video_path}\n"
        f"[preview] "
        f"{format_timestamp(start_ms)}"
        f" --> {format_timestamp(end_ms)}"
    )

    subprocess.run(
        [
            ffplay_path,
            "-loglevel",
            "error",
            "-ss",
            f"{start_seconds:.3f}",
            "-i",
            str(video_path),
            "-t",
            f"{duration_seconds:.3f}",
            "-autoexit",
            "-window_title",
            (
                f"Scene "
                f"{payload.get('scene_number', '?')}"
            ),
        ],
        check=False,
    )


def validate_rank(
    requested_rank: int,
    result_count: int,
    argument_name: str,
) -> int:
    if requested_rank < 1 or requested_rank > result_count:
        raise ValueError(
            f"{argument_name} must be between "
            f"1 and {result_count}."
        )

    return requested_rank - 1


def apply_exact_text_bonus(
    question: str,
    fused_results: list[dict[str, Any]],
    bonus: float = 0.015,
) -> None:
    normalized_question = " ".join(
        question.lower().split()
    )

    if not normalized_question:
        return

    for result in fused_results:
        payload = result["payload"]

        transcript = " ".join(
            payload.get("transcript", "").lower().split()
        )

        caption = " ".join(
            payload.get("caption", "").lower().split()
        )

        if normalized_question in transcript:
            result["rrf_score"] += bonus
            result["exact_transcript_match"] = True

        elif normalized_question in caption:
            result["rrf_score"] += bonus / 2
            result["exact_caption_match"] = True

    fused_results.sort(
        key=lambda result: -result["rrf_score"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Search caption, transcript, and keyframe vectors "
            "and combine them with Reciprocal Rank Fusion."
        )
    )

    parser.add_argument(
        "question",
        help="Natural-language video search question.",
    )

    parser.add_argument(
        "--index-metadata",
        required=True,
        type=Path,
        help="Path to index_metadata.json.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Embedding inference device.",
    )

    parser.add_argument(
        "--per-stream",
        type=int,
        default=10,
        help="Number of candidates retrieved per vector.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of fused scenes to display.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF smoothing constant.",
    )

    parser.add_argument(
        "--show-streams",
        action="store_true",
        help="Print each vector's results before fusion.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )

    parser.add_argument(
        "--open-keyframe-rank",
        type=int,
        default=None,
        help=(
            "Open the keyframe for a fused result. "
            "Use 1 for the best result."
        ),
    )

    parser.add_argument(
        "--preview-rank",
        type=int,
        default=None,
        help=(
            "Play the video scene with ffplay. "
            "Use 1 for the best result."
        ),
    )

    args = parser.parse_args()

    if args.per_stream < 1:
        raise ValueError("--per-stream must be at least 1.")

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1.")

    if args.rrf_k < 1:
        raise ValueError("--rrf-k must be at least 1.")

    metadata_path = args.index_metadata.resolve()
    metadata = load_json(metadata_path)

    qdrant_path = resolve_existing_path(
        raw_path=metadata["qdrant_path"],
        reference_directory=metadata_path.parent,
    )

    collection_name = metadata["collection_name"]
    text_model_name = metadata["text_model"]
    frame_model_name = metadata["frame_model"]

    print(f"[question] {args.question}")
    print(f"[collection] {collection_name}")
    print(f"[qdrant] {qdrant_path}")
    print(f"[text model] {text_model_name}")
    print(f"[frame model] {frame_model_name}")
    print(f"[device] {args.device}")
    print()

    print("[model] Loading text embedding model...")

    text_model = SentenceTransformer(
        text_model_name,
        device=args.device,
    )

    print("[model] Loading CLIP embedding model...")

    frame_model = SentenceTransformer(
        frame_model_name,
        device=args.device,
    )

    print("[embed] Encoding question...")

    query_vectors = create_query_vectors(
        question=args.question,
        text_model=text_model,
        frame_model=frame_model,
    )

    client = QdrantClient(path=str(qdrant_path))

    try:
        if not client.collection_exists(collection_name):
            raise RuntimeError(
                f"Qdrant collection does not exist: "
                f"{collection_name}"
            )

        stream_results: dict[str, list[Any]] = {}

        for stream_name in STREAM_NAMES:
            print(f"[query] Searching {stream_name}...")

            stream_results[stream_name] = (
                query_vector_stream(
                    client=client,
                    collection_name=collection_name,
                    vector_name=stream_name,
                    query_vector=query_vectors[stream_name],
                    limit=args.per_stream,
                )
            )

    finally:
        # Release the local Qdrant storage lock before opening
        # a preview or another Qdrant process.
        client.close()

    stream_weights = choose_stream_weights(
        args.question
    )

    print(
        "[fusion weights] "
        + ", ".join(
            f"{name}={weight}"
            for name, weight in stream_weights.items()
        )
    )

    fused_results = reciprocal_rank_fusion(
        stream_results=stream_results,
        rrf_k=args.rrf_k,
        stream_weights=stream_weights,
    )

    apply_exact_text_bonus(
        question=args.question,
        fused_results=fused_results,
    )

    if args.show_streams:
        display_stream_results(stream_results)

    display_fused_results(
        question=args.question,
        fused_results=fused_results,
        top_k=args.top_k,
    )

    output_data = build_serializable_output(
        question=args.question,
        fused_results=fused_results,
        top_k=args.top_k,
        per_stream=args.per_stream,
        rrf_k=args.rrf_k,
        index_metadata_path=metadata_path,
    )

    output_path = (
        args.output.resolve()
        if args.output is not None
        else metadata_path.parent / "last_search.json"
    )

    save_json(output_path, output_data)

    print(f"[output] {output_path}")

    displayed_results = fused_results[:args.top_k]

    if (
        args.open_keyframe_rank is not None
        and displayed_results
    ):
        selected_index = validate_rank(
            requested_rank=args.open_keyframe_rank,
            result_count=len(displayed_results),
            argument_name="--open-keyframe-rank",
        )

        open_keyframe(
            result=displayed_results[selected_index],
            reference_directory=metadata_path.parent,
        )

    if (
        args.preview_rank is not None
        and displayed_results
    ):
        selected_index = validate_rank(
            requested_rank=args.preview_rank,
            result_count=len(displayed_results),
            argument_name="--preview-rank",
        )

        preview_video_scene(
            result=displayed_results[selected_index],
            metadata_directory=metadata_path.parent,
        )