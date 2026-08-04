from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from search import (
    STREAM_NAMES,
    apply_exact_text_bonus,
    choose_stream_weights,
    create_query_vectors,
    is_visual_location_question,
    query_vector_stream,
    reciprocal_rank_fusion,
    resolve_existing_path,
)


class ModelCitation(BaseModel):
    scene_number: int = Field(
        description=(
            "Scene number supporting this answer. "
            "It must be one of the supplied evidence scenes."
        )
    )
    evidence: str = Field(
        description=(
            "A concise description of what this scene proves."
        )
    )


class ModelAnswer(BaseModel):
    answerable: bool = Field(
        description=(
            "True only when the supplied scene evidence supports "
            "a reliable answer."
        )
    )
    answer: str = Field(
        description=(
            "Direct answer to the user's question. "
            "Do not include invented details."
        )
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence based only on the supplied evidence."
    )
    citations: list[ModelCitation] = Field(
        default_factory=list,
        description=(
            "Scenes supporting the answer. Empty when the question "
            "cannot be answered."
        ),
    )
    limitations: list[str] = Field(
        default_factory=list,
        description=(
            "Important evidence limitations, ambiguities, or reasons "
            "the answer may be incomplete."
        ),
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


def get_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "image/jpeg"


def retrieve_scenes(
    question: str,
    metadata: dict[str, Any],
    metadata_path: Path,
    device: str,
    per_stream: int,
    top_retrieved: int,
    rrf_k: int,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    qdrant_path = resolve_existing_path(
        raw_path=metadata["qdrant_path"],
        reference_directory=metadata_path.parent,
    )

    collection_name = metadata["collection_name"]
    text_model_name = metadata["text_model"]
    frame_model_name = metadata["frame_model"]

    print(f"[text model] {text_model_name}")
    print(f"[frame model] {frame_model_name}")
    print(f"[qdrant] {qdrant_path}")

    print("[model] Loading text embedding model...")

    text_model = SentenceTransformer(
        text_model_name,
        device=device,
    )

    print("[model] Loading CLIP embedding model...")

    frame_model = SentenceTransformer(
        frame_model_name,
        device=device,
    )

    print("[embed] Encoding question...")

    query_vectors = create_query_vectors(
        question=question,
        text_model=text_model,
        frame_model=frame_model,
    )

    client = QdrantClient(path=str(qdrant_path))

    try:
        stream_results: dict[str, list[Any]] = {}

        for stream_name in STREAM_NAMES:
            print(f"[query] Searching {stream_name}...")

            stream_results[stream_name] = query_vector_stream(
                client=client,
                collection_name=collection_name,
                vector_name=stream_name,
                query_vector=query_vectors[stream_name],
                limit=per_stream,
            )
    finally:
        client.close()

    stream_weights = choose_stream_weights(question)

    print(
        "[fusion weights] "
        + ", ".join(
            f"{name}={weight}"
            for name, weight in stream_weights.items()
        )
    )

    fused_results = reciprocal_rank_fusion(
        stream_results=stream_results,
        rrf_k=rrf_k,
        stream_weights=stream_weights,
    )

    apply_exact_text_bonus(
        question=question,
        fused_results=fused_results,
    )

    return fused_results[:top_retrieved], stream_weights


def find_caption_manifest(
    metadata_path: Path,
) -> Path:
    expected_path = (
        metadata_path.parent / "scenes_with_captions.json"
    )

    if expected_path.exists():
        return expected_path

    candidates = list(
        metadata_path.parent.glob(
            "scenes_with_captions*.json"
        )
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not find scenes_with_captions.json beside "
            "index_metadata.json."
        )

    return candidates[0]


def select_evidence_scenes(
    retrieved_results: list[dict[str, Any]],
    manifest: dict[str, Any],
    neighbor_radius: int,
    max_evidence: int,
) -> list[dict[str, Any]]:
    scenes = manifest.get("scenes", [])

    scenes_by_number = {
        int(scene["scene_number"]): scene
        for scene in scenes
    }

    retrieval_rank_by_number = {
        int(result["payload"]["scene_number"]): rank
        for rank, result in enumerate(
            retrieved_results,
            start=1,
        )
    }

    selected_numbers: list[int] = []

    def add_scene_number(scene_number: int) -> None:
        if (
            scene_number in scenes_by_number
            and scene_number not in selected_numbers
            and len(selected_numbers) < max_evidence
        ):
            selected_numbers.append(scene_number)

    # Always preserve the directly retrieved scenes first.
    for result in retrieved_results:
        add_scene_number(
            int(result["payload"]["scene_number"])
        )

    # Then add immediate temporal context.
    for result in retrieved_results:
        center = int(result["payload"]["scene_number"])

        for distance in range(1, neighbor_radius + 1):
            add_scene_number(center - distance)
            add_scene_number(center + distance)

    evidence_scenes: list[dict[str, Any]] = []

    for scene_number in sorted(selected_numbers):
        scene = dict(scenes_by_number[scene_number])

        scene["retrieval_rank"] = (
            retrieval_rank_by_number.get(scene_number)
        )

        scene["is_direct_retrieval"] = (
            scene_number in retrieval_rank_by_number
        )

        evidence_scenes.append(scene)

    return evidence_scenes


def build_evidence_text(
    question: str,
    evidence_scenes: list[dict[str, Any]],
) -> str:
    blocks: list[str] = []

    for scene in evidence_scenes:
        scene_number = int(scene["scene_number"])
        start_ms = int(scene["start_ms"])
        end_ms = int(scene["end_ms"])

        caption = (
            scene.get("caption", "").strip()
            or "[No visual caption]"
        )

        transcript = (
            scene.get("transcript", "").strip()
            or "[No spoken dialogue]"
        )

        source_type = (
            f"direct retrieval rank "
            f"{scene['retrieval_rank']}"
            if scene.get("is_direct_retrieval")
            else "neighboring temporal context"
        )

        blocks.append(
            "\n".join(
                [
                    f"EVIDENCE SCENE {scene_number}",
                    (
                        f"Time: {format_timestamp(start_ms)} "
                        f"to {format_timestamp(end_ms)}"
                    ),
                    f"Selection: {source_type}",
                    f"Visual caption: {caption}",
                    f"Transcript: {transcript}",
                ]
            )
        )

    return (
        f"USER QUESTION:\n{question}\n\n"
        "AVAILABLE VIDEO EVIDENCE:\n\n"
        + "\n\n".join(blocks)
    )


def build_multimodal_contents(
    prompt_text: str,
    evidence_scenes: list[dict[str, Any]],
    manifest_path: Path,
    include_images: bool,
) -> list[Any]:
    contents: list[Any] = [prompt_text]

    if not include_images:
        return contents

    for scene in evidence_scenes:
        raw_keyframe_path = scene.get("keyframe_path")

        if not raw_keyframe_path:
            continue

        try:
            keyframe_path = resolve_existing_path(
                raw_path=raw_keyframe_path,
                reference_directory=manifest_path.parent,
            )
        except FileNotFoundError as error:
            print(f"[warning] {error}")
            continue

        scene_label = (
            f"Keyframe for evidence scene "
            f"{scene['scene_number']}"
        )

        contents.append(scene_label)

        contents.append(
            types.Part.from_bytes(
                data=keyframe_path.read_bytes(),
                mime_type=get_mime_type(keyframe_path),
            )
        )

    return contents


def validate_model_answer(
    model_answer: ModelAnswer,
    evidence_scenes: list[dict[str, Any]],
) -> tuple[ModelAnswer, list[dict[str, Any]]]:
    evidence_by_number = {
        int(scene["scene_number"]): scene
        for scene in evidence_scenes
    }

    valid_model_citations: list[ModelCitation] = []
    output_citations: list[dict[str, Any]] = []

    seen_numbers: set[int] = set()

    for citation in model_answer.citations:
        scene_number = int(citation.scene_number)

        if scene_number not in evidence_by_number:
            print(
                f"[warning] Removing unsupported citation "
                f"to scene {scene_number}."
            )
            continue

        if scene_number in seen_numbers:
            continue

        seen_numbers.add(scene_number)
        valid_model_citations.append(citation)

        scene = evidence_by_number[scene_number]

        output_citations.append(
            {
                "scene_number": scene_number,
                "start_ms": int(scene["start_ms"]),
                "end_ms": int(scene["end_ms"]),
                "start_timestamp": format_timestamp(
                    int(scene["start_ms"])
                ),
                "end_timestamp": format_timestamp(
                    int(scene["end_ms"])
                ),
                "evidence": citation.evidence,
                "caption": scene.get("caption", ""),
                "transcript": scene.get(
                    "transcript",
                    "",
                ),
                "keyframe_path": scene.get(
                    "keyframe_path",
                ),
            }
        )

    model_answer.citations = valid_model_citations

    if model_answer.answerable and not output_citations:
        model_answer.answerable = False
        model_answer.answer = (
            "The retrieved scenes do not provide enough "
            "verifiable evidence to answer reliably."
        )
        model_answer.confidence = "low"

        model_answer.limitations.append(
            "The model did not return a valid citation "
            "to any supplied evidence scene."
        )

    if not model_answer.answerable:
        model_answer.citations = []
        output_citations = []

    return model_answer, output_citations


def calibrate_confidence(
    model_answer: ModelAnswer,
) -> ModelAnswer:
    uncertainty_phrases = (
        "does not explicitly",
        "not explicitly",
        "imply",
        "implied",
        "unclear",
        "ambiguous",
        "cannot confirm",
    )

    limitation_text = " ".join(
        model_answer.limitations
    ).lower()

    if any(
        phrase in limitation_text
        for phrase in uncertainty_phrases
    ):
        if model_answer.confidence == "high":
            model_answer.confidence = "low"

    return model_answer


def display_evidence(
    evidence_scenes: list[dict[str, Any]],
) -> None:
    print()
    print("=" * 72)
    print("EVIDENCE SENT TO ANSWERER")
    print("=" * 72)

    for scene in evidence_scenes:
        start_ms = int(scene["start_ms"])
        end_ms = int(scene["end_ms"])

        source = (
            f"retrieved #{scene['retrieval_rank']}"
            if scene.get("is_direct_retrieval")
            else "neighbor"
        )

        print(
            f"Scene {scene['scene_number']} "
            f"[{format_timestamp(start_ms)}"
            f" --> {format_timestamp(end_ms)}] "
            f"({source})"
        )

        print(
            "  Caption: "
            + (
                scene.get("caption", "").strip()
                or "[No visual caption]"
            )
        )

        print(
            "  Transcript: "
            + (
                scene.get("transcript", "").strip()
                or "[No spoken dialogue]"
            )
        )

        print()


def display_answer(
    answer_data: dict[str, Any],
) -> None:
    print()
    print("=" * 72)
    print("GROUNDED ANSWER")
    print("=" * 72)

    print(f"Answerable: {answer_data['answerable']}")
    print(f"Confidence: {answer_data['confidence']}")
    print()
    print(answer_data["answer"])

    citations = answer_data.get("citations", [])

    if citations:
        print()
        print("Citations:")

        for citation in citations:
            print(
                f"- Scene {citation['scene_number']} "
                f"[{citation['start_timestamp']}"
                f" --> {citation['end_timestamp']}]: "
                f"{citation['evidence']}"
            )

    limitations = answer_data.get("limitations", [])

    if limitations:
        print()
        print("Limitations:")

        for limitation in limitations:
            print(f"- {limitation}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrieve video evidence and produce a grounded "
            "answer with timestamp citations."
        )
    )

    parser.add_argument(
        "question",
        help="Natural-language question about the video.",
    )

    parser.add_argument(
        "--index-metadata",
        required=True,
        type=Path,
        help="Path to index_metadata.json.",
    )

    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite",
        help="Gemini model used for answer synthesis.",
    )

    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Embedding model device.",
    )

    parser.add_argument(
        "--per-stream",
        type=int,
        default=10,
        help="Candidates retrieved from each vector stream.",
    )

    parser.add_argument(
        "--top-retrieved",
        type=int,
        default=5,
        help="Top fused scenes used as primary evidence.",
    )

    parser.add_argument(
        "--neighbor-radius",
        type=int,
        default=1,
        help="Neighboring scenes added around each result.",
    )

    parser.add_argument(
        "--max-evidence",
        type=int,
        default=7,
        help="Maximum scenes passed to the answer model.",
    )

    parser.add_argument(
        "--rrf-k",
        type=int,
        default=60,
        help="RRF smoothing constant.",
    )

    parser.add_argument(
        "--text-only",
        action="store_true",
        help=(
            "Do not send keyframe images. Useful for dialogue "
            "questions and lower multimodal usage."
        ),
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Retrieve and print evidence without calling Gemini."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path.",
    )

    args = parser.parse_args()

    if args.top_retrieved < 1:
        raise ValueError("--top-retrieved must be at least 1.")

    if args.neighbor_radius < 0:
        raise ValueError("--neighbor-radius cannot be negative.")

    if args.max_evidence < args.top_retrieved:
        raise ValueError(
            "--max-evidence must be at least --top-retrieved."
        )

    metadata_path = args.index_metadata.resolve()
    metadata = load_json(metadata_path)

    print(f"[question] {args.question}")
    print(f"[answer model] {args.model}")
    print(f"[device] {args.device}")

    retrieved_results, stream_weights = retrieve_scenes(
        question=args.question,
        metadata=metadata,
        metadata_path=metadata_path,
        device=args.device,
        per_stream=args.per_stream,
        top_retrieved=args.top_retrieved,
        rrf_k=args.rrf_k,
    )

    if not retrieved_results:
        raise RuntimeError("Retrieval returned no scenes.")

    manifest_path = find_caption_manifest(metadata_path)
    manifest = load_json(manifest_path)

    effective_max_evidence = args.max_evidence

    if is_visual_location_question(args.question):
        # A location answer needs the frames surrounding every primary
        # retrieval. Otherwise a relevant lower-ranked scene can survive
        # retrieval while its two-person/context frames are dropped.
        complete_neighborhood_budget = (
            args.top_retrieved
            * (1 + 2 * args.neighbor_radius)
        )
        effective_max_evidence = max(
            effective_max_evidence,
            complete_neighborhood_budget,
        )

        if effective_max_evidence != args.max_evidence:
            print()
            print(
                "[evidence budget] "
                f"expanded {args.max_evidence} -> "
                f"{effective_max_evidence} for visual location"
            )

    evidence_scenes = select_evidence_scenes(
        retrieved_results=retrieved_results,
        manifest=manifest,
        neighbor_radius=args.neighbor_radius,
        max_evidence=effective_max_evidence,
    )

    display_evidence(evidence_scenes)

    evidence_context_path = (
        metadata_path.parent / "last_evidence_context.json"
    )

    save_json(
        evidence_context_path,
        {
            "question": args.question,
            "retrieval_weights": stream_weights,
            "retrieved_results": [
                {
                    "rank": rank,
                    "scene_number": result[
                        "payload"
                    ].get("scene_number"),
                    "rrf_score": result["rrf_score"],
                    "ranks": result["ranks"],
                    "similarities": result[
                        "similarities"
                    ],
                }
                for rank, result in enumerate(
                    retrieved_results,
                    start=1,
                )
            ],
            "evidence_scenes": evidence_scenes,
        },
    )

    print(
        f"[evidence output] {evidence_context_path}"
    )

    if args.dry_run:
        print(
            "[dry run] Evidence retrieval completed. "
            "Gemini was not called."
        )
        return

    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found in .env."
        )

    evidence_text = build_evidence_text(
        question=args.question,
        evidence_scenes=evidence_scenes,
    )

    system_instruction = """
You are the grounded answer component of a video question-answering system.

Use only the supplied scene evidence. Do not rely on outside knowledge,
even when you recognize the film, people, setting, or story.

Evidence rules:
1. Captions may contain visual-model mistakes.
2. Transcripts may contain speech-recognition mistakes.
3. A single keyframe does not prove an entire action, count, sequence,
   cause, intention, or temporal relationship.
4. Prefer evidence confirmed by more than one source, such as transcript
   plus image or neighboring scenes.
5. Cite only scene numbers that appear in the supplied evidence.
6. If the evidence is incomplete, ambiguous, or does not prove the
   answer, set answerable=false.
7. Never invent timestamps, names, actions, dialogue, or scene numbers.
8. For counting and before/after questions, be especially conservative.
9. Do not identify a visible person as a named character unless the
   transcript, on-screen text, or neighboring evidence establishes that
   identity. A caption saying "a man" or "a woman" does not prove that
   the person is a named character.
10. For questions naming multiple characters, do not claim that the
    visible people are those characters unless the supplied evidence
    supports both identities.
""".strip()


    if is_visual_location_question(args.question):
        system_instruction += """

    Additional rules for visual-loca tion questions:

    11. Evaluate consecutive, temporally adjacent scenes as one local
        evidence group. Do not mix identities or settings between
        unrelated scene groups.

    12. The names of all requested characters do not need to be spoken
        inside the same scene. One participant may be identified through
        local transcript or on-screen text, while the other may be
        established through continuous adjacent frames showing the same
        ongoing two-person interaction.

    13. For a question asking where named people are standing, determine:
        a. whether a local adjacent scene group anchors the conversation,
        b. whether the frames show the participants together,
        c. what setting is directly visible.

    14. Do not transfer character identities between nonadjacent scene
        groups. Evidence from an earlier room cannot establish that people
        in a later outdoor scene are the same characters, or vice versa.

    15. When a local group contains a named conversation anchor, adjacent
        frames show the two-person interaction, and the setting is visible,
        answer with the directly visible setting and cite that local group.
    """

    user_prompt = (
        evidence_text
        + "\n\n"
        + """
TASK:
Answer the user question using only the evidence above.

When answerable:
- Give a concise direct answer.
- Include every scene that materially supports the answer.
- State uncertainty in limitations when necessary.

When not answerable:
- Set answerable to false.
- Explain briefly what evidence is missing.
- Return no citations.
""".strip()
    )

    if is_visual_location_question(args.question):
        user_prompt += """

    LOCATION-QUESTION PROCEDURE:

    - Examine each consecutive scene group independently.
    - Prefer the group where transcript or on-screen text anchors the
    requested conversation.
    - Use adjacent frames in that same group to establish the interaction.
    - Describe only the directly visible location.
    - Do not require both character names to be spoken.
    - Do not borrow identity evidence from unrelated scene groups.
    """

    contents = build_multimodal_contents(
        prompt_text=user_prompt,
        evidence_scenes=evidence_scenes,
        manifest_path=manifest_path,
        include_images=not args.text_only,
    )

    print()
    print(
        f"[synth] Calling {args.model} with "
        f"{len(evidence_scenes)} evidence scenes..."
    )

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=args.model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ModelAnswer,
            temperature=0.1,
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    model_answer = ModelAnswer.model_validate_json(
        response.text
    )

    model_answer, citations = validate_model_answer(
        model_answer=model_answer,
        evidence_scenes=evidence_scenes,
    )

    # Downgrade confidence when the model's own limitations
    # acknowledge ambiguity or unsupported inference.
    model_answer = calibrate_confidence(model_answer)

    output_data = {
        "question": args.question,
        "answerable": model_answer.answerable,
        "answer": model_answer.answer,
        "confidence": model_answer.confidence,
        "citations": citations,
        "limitations": model_answer.limitations,
        "model": args.model,
        "text_only": args.text_only,
        "retrieval": {
            "top_retrieved": args.top_retrieved,
            "neighbor_radius": args.neighbor_radius,
            "max_evidence": effective_max_evidence,
            "requested_max_evidence": args.max_evidence,
            "stream_weights": stream_weights,
        },
        "evidence_context_path": str(
            evidence_context_path
        ),
    }

    output_path = (
        args.output.resolve()
        if args.output is not None
        else metadata_path.parent / "last_answer.json"
    )

    save_json(output_path, output_data)

    display_answer(output_data)

    print()
    print(f"[output] {output_path}")

def is_visual_location_question(
    question: str,
) -> bool:
    normalized = " ".join(
        question.lower().split()
    )

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

if __name__ == "__main__":
    main()