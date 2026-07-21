from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Any

from PIL import Image
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer


DEFAULT_TEXT_MODEL = (
    "sentence-transformers/all-MiniLM-L6-v2"
)

DEFAULT_FRAME_MODEL = (
    "sentence-transformers/clip-ViT-B-32"
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def resolve_existing_path(
    raw_path: str,
    manifest_path: Path,
) -> Path:
    candidate = Path(raw_path)

    possible_paths = [
        candidate,
        Path.cwd() / candidate,
        manifest_path.parent / candidate,
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
        f"Could not locate file: {raw_path}\n"
        f"Attempted:\n{attempted}"
    )


def create_point_id(scene_id: str) -> str:
    """
    Create a deterministic UUID.

    Reindexing the same scene updates the existing Qdrant point
    rather than creating a duplicate.
    """
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"video-scene:{scene_id}",
        )
    )


def get_embedding_dimension(
    model: SentenceTransformer,
) -> int:
    dimension = model.get_sentence_embedding_dimension()

    if dimension is not None:
        return int(dimension)

    test_embedding = model.encode(
        ["dimension test"],
        convert_to_numpy=True,
    )

    return int(test_embedding.shape[1])


def load_rgb_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


def create_collection(
    client: QdrantClient,
    collection_name: str,
    text_dimension: int,
    frame_dimension: int,
    recreate: bool,
) -> None:
    collection_exists = client.collection_exists(
        collection_name=collection_name
    )

    if collection_exists and recreate:
        print(
            f"[qdrant] Deleting existing collection: "
            f"{collection_name}"
        )

        client.delete_collection(
            collection_name=collection_name
        )

        collection_exists = False

    if collection_exists:
        print(
            f"[qdrant] Reusing existing collection: "
            f"{collection_name}"
        )
        return

    print(
        f"[qdrant] Creating collection: {collection_name}"
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "caption": models.VectorParams(
                size=text_dimension,
                distance=models.Distance.COSINE,
            ),
            "transcript": models.VectorParams(
                size=text_dimension,
                distance=models.Distance.COSINE,
            ),
            "keyframe": models.VectorParams(
                size=frame_dimension,
                distance=models.Distance.COSINE,
            ),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Create caption, transcript, and keyframe "
            "embeddings and write them to Qdrant."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to scenes_with_captions.json.",
    )

    parser.add_argument(
        "--qdrant-path",
        type=Path,
        default=Path("data/qdrant"),
        help="Local persistent Qdrant directory.",
    )

    parser.add_argument(
        "--collection",
        default="video_scenes",
        help="Qdrant collection name.",
    )

    parser.add_argument(
        "--text-model",
        default=DEFAULT_TEXT_MODEL,
        help="Embedding model for captions and transcripts.",
    )

    parser.add_argument(
        "--frame-model",
        default=DEFAULT_FRAME_MODEL,
        help="CLIP model for keyframes.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Embedding device.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of scenes processed per batch.",
    )

    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection.",
    )

    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)

    scenes = manifest.get("scenes", [])

    if not scenes:
        raise ValueError("The manifest contains no scenes.")

    missing_captions = [
        scene["scene_id"]
        for scene in scenes
        if not scene.get("caption")
    ]

    if missing_captions:
        example_ids = ", ".join(missing_captions[:5])

        raise ValueError(
            f"{len(missing_captions)} scenes have no caption. "
            f"Examples: {example_ids}. "
            "Finish caption_scenes.py first."
        )

    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")

    print(f"[input] Manifest: {manifest_path}")
    print(f"[input] Scenes: {len(scenes)}")
    print(f"[device] {args.device}")
    print()

    print(f"[model] Loading text model: {args.text_model}")

    text_model = SentenceTransformer(
        args.text_model,
        device=args.device,
    )

    print(
        f"[model] Loading frame model: "
        f"{args.frame_model}"
    )

    frame_model = SentenceTransformer(
        args.frame_model,
        device=args.device,
    )

    text_dimension = get_embedding_dimension(text_model)
    frame_dimension = get_embedding_dimension(frame_model)

    print(f"[dimension] Caption: {text_dimension}")
    print(f"[dimension] Transcript: {text_dimension}")
    print(f"[dimension] Keyframe: {frame_dimension}")
    print()

    qdrant_path = args.qdrant_path.resolve()
    qdrant_path.mkdir(parents=True, exist_ok=True)

    print(f"[qdrant] Local path: {qdrant_path}")

    client = QdrantClient(path=str(qdrant_path))

    try:
        create_collection(
            client=client,
            collection_name=args.collection,
            text_dimension=text_dimension,
            frame_dimension=frame_dimension,
            recreate=args.recreate,
        )

        total_scenes = len(scenes)

        for batch_start in range(
            0,
            total_scenes,
            args.batch_size,
        ):
            batch_end = min(
                batch_start + args.batch_size,
                total_scenes,
            )

            scene_batch = scenes[batch_start:batch_end]

            print(
                f"[embed] Scenes "
                f"{batch_start + 1}-{batch_end}"
                f"/{total_scenes}"
            )

            captions = [
                scene["caption"]
                for scene in scene_batch
            ]

            transcripts = [
                (
                    scene.get("transcript", "").strip()
                    or "No spoken dialogue is present "
                       "in this scene."
                )
                for scene in scene_batch
            ]

            keyframe_paths = [
                resolve_existing_path(
                    raw_path=scene["keyframe_path"],
                    manifest_path=manifest_path,
                )
                for scene in scene_batch
            ]

            images = [
                load_rgb_image(keyframe_path)
                for keyframe_path in keyframe_paths
            ]

            caption_vectors = text_model.encode(
                captions,
                batch_size=args.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            transcript_vectors = text_model.encode(
                transcripts,
                batch_size=args.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            keyframe_vectors = frame_model.encode(
                images,
                batch_size=args.batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            points: list[models.PointStruct] = []

            for index, scene in enumerate(scene_batch):
                keyframe_path = keyframe_paths[index]

                payload = {
                    "video_id": manifest["video_id"],
                    "scene_id": scene["scene_id"],
                    "scene_number": scene["scene_number"],
                    "start_ms": scene["start_ms"],
                    "end_ms": scene["end_ms"],
                    "duration_ms": scene["duration_ms"],
                    "keyframe_ms": scene["keyframe_ms"],
                    "keyframe_path": str(keyframe_path),
                    "caption": scene["caption"],
                    "transcript": (
                        scene.get("transcript", "")
                    ),
                    "has_speech": scene.get(
                        "has_speech",
                        False,
                    ),
                    "word_count": scene.get(
                        "word_count",
                        0,
                    ),
                }

                point = models.PointStruct(
                    id=create_point_id(scene["scene_id"]),
                    vector={
                        "caption": (
                            caption_vectors[index].tolist()
                        ),
                        "transcript": (
                            transcript_vectors[index].tolist()
                        ),
                        "keyframe": (
                            keyframe_vectors[index].tolist()
                        ),
                    },
                    payload=payload,
                )

                points.append(point)

            client.upsert(
                collection_name=args.collection,
                points=points,
                wait=True,
            )

            print(
                f"[qdrant] Upserted {len(points)} points"
            )

        count_result = client.count(
            collection_name=args.collection,
            exact=True,
        )

        collection_info = client.get_collection(
            collection_name=args.collection
        )

        metadata = {
            "collection_name": args.collection,
            "qdrant_path": str(qdrant_path),
            "indexed_point_count": count_result.count,
            "expected_scene_count": len(scenes),
            "text_model": args.text_model,
            "frame_model": args.frame_model,
            "vectors": {
                "caption": {
                    "dimension": text_dimension,
                    "distance": "cosine",
                },
                "transcript": {
                    "dimension": text_dimension,
                    "distance": "cosine",
                },
                "keyframe": {
                    "dimension": frame_dimension,
                    "distance": "cosine",
                },
            },
            "collection_status": str(
                collection_info.status
            ),
        }

        metadata_path = (
            manifest_path.parent / "index_metadata.json"
        )

        save_json(metadata_path, metadata)

        print()
        print("[complete] Index construction completed")
        print(
            f"[complete] Stored points: "
            f"{count_result.count}"
        )
        print(
            f"[complete] Expected scenes: "
            f"{len(scenes)}"
        )
        print(
            f"[complete] Collection status: "
            f"{collection_info.status}"
        )
        print(f"[output] {metadata_path}")

        if count_result.count != len(scenes):
            print(
                "[warning] Qdrant point count does not "
                "match the scene count."
            )

    finally:
        client.close()