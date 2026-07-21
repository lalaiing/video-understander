from __future__ import annotations

import argparse
import json
from json.tool import main
import mimetypes
import os
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from google import genai
from google.genai import types


CAPTION_PROMPT = """
Describe this video keyframe in one or two concise, factual sentences.

Include:
- visible people;
- visible objects;
- the setting;
- any clearly visible action.

Rules:
- Describe only what is visibly supported.
- Do not invent names, identities, intentions, causes, or unseen events.
- Do not mention that this is an image or keyframe.
- Return only the caption, without headings or bullet points.
""".strip()


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
    manifest_path: Path,
) -> Path:
    """
    Resolve a path stored in the manifest.

    Earlier milestones may have stored either:
    - an absolute Windows path; or
    - a path relative to the project root.
    """

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
        f"Could not locate keyframe: {raw_path}\n"
        f"Attempted:\n{attempted}"
    )


def get_mime_type(image_path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(image_path.name)

    if mime_type is None:
        return "image/jpeg"

    return mime_type


def generate_caption(
    client: genai.Client,
    image_path: Path,
    model_name: str,
    max_retries: int,
) -> str:
    image_bytes = image_path.read_bytes()
    mime_type = get_mime_type(image_path)

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type=mime_type,
                    ),
                    CAPTION_PROMPT,
                ],
            )

            caption = (response.text or "").strip()

            if not caption:
                raise RuntimeError(
                    "Gemini returned an empty caption."
                )

            # Keep the result convenient for embedding and display.
            caption = " ".join(caption.split())

            return caption

        except Exception as error:
            last_error = error

            if attempt == max_retries:
                break

            wait_seconds = min(2**attempt, 30)

            print(
                f"[retry] Attempt {attempt} failed: {error}"
            )
            print(
                f"[retry] Waiting {wait_seconds} seconds..."
            )

            time.sleep(wait_seconds)

    raise RuntimeError(
        f"Captioning failed after {max_retries} attempts "
        f"for {image_path}: {last_error}"
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Generate a visual caption for every scene keyframe."
        )
    )

    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to scenes_with_transcript.json.",
    )

    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite",
        help="Gemini model used for image captioning.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Maximum number of new API requests. "
            "Useful for testing on a few scenes."
        ),
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between successful API requests.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Maximum attempts for each keyframe.",
    )

    args = parser.parse_args()

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found. "
            "Create a .env file in the project root."
        )

    manifest_path = args.manifest.resolve()
    manifest = load_json(manifest_path)

    scenes = manifest.get("scenes", [])

    if not scenes:
        raise ValueError("The manifest contains no scenes.")

    output_path = (
        manifest_path.parent / "scenes_with_captions.json"
    )

    cache_path = (
        manifest_path.parent / "captions_cache.json"
    )

    if cache_path.exists():
        cache = load_json(cache_path)
    else:
        cache = {}

    client = genai.Client(api_key=api_key)

    new_request_count = 0

    print(f"[input] Scenes: {len(scenes)}")
    print(f"[model] {args.model}")
    print(f"[cache] {cache_path}")
    print()

    for scene_position, scene in enumerate(scenes, start=1):
        scene_id = scene["scene_id"]

        # Preserve captions generated in previous runs,
        # even when switching to another model.
        if scene.get("caption"):
            print(
                f"[existing] {scene_position}/{len(scenes)} "
                f"{scene_id}: {scene['caption']}"
            )
            continue

        cached_result = cache.get(scene_id)

        if (
            cached_result
            and cached_result.get("model") == args.model
            and cached_result.get("caption")
        ):
            scene["caption"] = cached_result["caption"]
            scene["caption_model"] = args.model

            print(
                f"[cached] {scene_position}/{len(scenes)} "
                f"{scene_id}: {scene['caption']}"
            )

            continue

        if (
            args.limit is not None
            and new_request_count >= args.limit
        ):
            print(
                "[limit] Reached the maximum number "
                "of new requests."
            )
            break

        raw_keyframe_path = scene.get("keyframe_path")

        if not raw_keyframe_path:
            raise ValueError(
                f"Scene {scene_id} has no keyframe_path."
            )

        keyframe_path = resolve_existing_path(
            raw_path=raw_keyframe_path,
            manifest_path=manifest_path,
        )

        print(
            f"[caption] {scene_position}/{len(scenes)} "
            f"{scene_id}"
        )

        caption = generate_caption(
            client=client,
            image_path=keyframe_path,
            model_name=args.model,
            max_retries=args.max_retries,
        )

        scene["caption"] = caption
        scene["caption_model"] = args.model

        cache[scene_id] = {
            "caption": caption,
            "model": args.model,
            "keyframe_path": str(keyframe_path),
        }

        new_request_count += 1

        print(f"          {caption}")

        # Checkpoint after every scene. If the process stops,
        # already completed captions do not need to be requested again.
        save_json(cache_path, cache)

        manifest["captioning"] = {
            "model": args.model,
            "completed_scene_count": sum(
                1
                for current_scene in scenes
                if current_scene.get("caption")
            ),
            "total_scene_count": len(scenes),
            "cache_file": str(cache_path),
        }

        save_json(output_path, manifest)

        if args.delay > 0:
            time.sleep(args.delay)

    completed_count = sum(
        1
        for scene in scenes
        if scene.get("caption")
    )

    manifest["captioning"] = {
        "model": args.model,
        "completed_scene_count": completed_count,
        "total_scene_count": len(scenes),
        "is_complete": completed_count == len(scenes),
        "cache_file": str(cache_path),
    }

    save_json(cache_path, cache)
    save_json(output_path, manifest)

    print()
    print("[complete] Captioning pass finished")
    print(
        f"[complete] Captioned scenes: "
        f"{completed_count}/{len(scenes)}"
    )
    print(f"[complete] New API requests: {new_request_count}")
    print(f"[output] {output_path}")

    if completed_count < len(scenes):
        print(
            "[next] Run the same command without --limit "
            "to caption the remaining scenes."
        )