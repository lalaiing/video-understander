from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary_path.replace(path)


def normalize_text(value: str) -> str:
    normalized = value.lower().replace("’", "'")
    return re.sub(r"\s+", " ", normalized).strip()


def timestamp_iou(
    predicted_start_ms: int,
    predicted_end_ms: int,
    expected_start_ms: int,
    expected_end_ms: int,
) -> float:
    intersection_start = max(predicted_start_ms, expected_start_ms)
    intersection_end = min(predicted_end_ms, expected_end_ms)
    intersection = max(0, intersection_end - intersection_start)
    union_start = min(predicted_start_ms, expected_start_ms)
    union_end = max(predicted_end_ms, expected_end_ms)
    union = max(0, union_end - union_start)
    return intersection / union if union > 0 else 0.0


def check_answer_text(
    answer: str,
    specification: dict[str, Any] | None,
) -> tuple[bool, dict[str, Any]]:
    if not specification:
        return True, {"checked": False}

    normalized_answer = normalize_text(answer)
    any_terms = [
        normalize_text(str(term))
        for term in specification.get("any_terms", [])
        if str(term).strip()
    ]
    all_groups = [
        [
            normalize_text(str(term))
            for term in group
            if str(term).strip()
        ]
        for group in specification.get("all_groups", [])
        if isinstance(group, list)
    ]
    forbidden_terms = [
        normalize_text(str(term))
        for term in specification.get("forbidden_terms", [])
        if str(term).strip()
    ]

    any_terms_pass = (
        True
        if not any_terms
        else any(term in normalized_answer for term in any_terms)
    )

    group_results: list[dict[str, Any]] = []
    for group in all_groups:
        group_pass = (
            True
            if not group
            else any(term in normalized_answer for term in group)
        )
        group_results.append({"terms": group, "pass": group_pass})

    all_groups_pass = all(item["pass"] for item in group_results)
    matched_forbidden_terms = [
        term for term in forbidden_terms if term in normalized_answer
    ]
    forbidden_terms_pass = not matched_forbidden_terms
    passed = any_terms_pass and all_groups_pass and forbidden_terms_pass

    return passed, {
        "checked": True,
        "any_terms": any_terms,
        "any_terms_pass": any_terms_pass,
        "all_group_results": group_results,
        "all_groups_pass": all_groups_pass,
        "forbidden_terms": forbidden_terms,
        "matched_forbidden_terms": matched_forbidden_terms,
        "forbidden_terms_pass": forbidden_terms_pass,
    }


def run_command(
    command: list[str],
    working_directory: Path,
    log_path: Path,
) -> None:
    print()
    print("[run] " + subprocess.list2cmdline(command))
    completed = subprocess.run(
        command,
        cwd=working_directory,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="")
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed.returncode}. "
            f"See {log_path}"
        )


def build_grounding_output_path(
    output_directory: Path,
    question: str,
) -> Path:
    query_hash = hashlib.sha1(
        question.encode("utf-8")
    ).hexdigest()[:10]
    return output_directory / query_hash / "last_grounding.json"


def score_grounding(
    grounding_payload: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    windows = grounding_payload.get("windows", [])
    answerable_windows = [
        window
        for window in windows
        if (
            window.get("answerable") is True
            and window.get("grounded_start_ms") is not None
            and window.get("grounded_end_ms") is not None
        )
    ]

    temporal_specification = case.get("temporal")
    result: dict[str, Any] = {
        "answerable_window_count": len(answerable_windows),
        "answerable_windows": [
            {
                "window_index": window.get("window_index"),
                "scene_numbers": window.get("scene_numbers", []),
                "grounded_start_ms": window.get("grounded_start_ms"),
                "grounded_end_ms": window.get("grounded_end_ms"),
                "support_type": window.get("support_type"),
                "confidence": window.get("confidence"),
                "evidence": window.get("evidence"),
            }
            for window in answerable_windows
        ],
        "top1_iou": None,
        "oracle_iou": None,
        "top1_window_index": None,
        "oracle_window_index": None,
        "temporal_pass": True,
        "window_count_pass": True,
    }

    expected_answerable = bool(case["expected_answerable"])
    maximum_answerable_windows = case.get("max_answerable_windows")
    if temporal_specification:
        maximum_answerable_windows = temporal_specification.get(
            "max_answerable_windows",
            maximum_answerable_windows,
        )

    if maximum_answerable_windows is not None:
        result["window_count_pass"] = (
            len(answerable_windows) <= int(maximum_answerable_windows)
        )

    if not expected_answerable:
        result["temporal_pass"] = len(answerable_windows) == 0
        return result

    if not temporal_specification:
        result["temporal_pass"] = len(answerable_windows) >= 1
        return result

    expected_start_ms = int(temporal_specification["start_ms"])
    expected_end_ms = int(temporal_specification["end_ms"])
    minimum_top1_iou = float(
        temporal_specification.get("min_top1_iou", 0.3)
    )
    minimum_oracle_iou = float(
        temporal_specification.get(
            "min_oracle_iou",
            minimum_top1_iou,
        )
    )

    scored_windows: list[dict[str, Any]] = []
    for window in answerable_windows:
        score = timestamp_iou(
            int(window["grounded_start_ms"]),
            int(window["grounded_end_ms"]),
            expected_start_ms,
            expected_end_ms,
        )
        scored_windows.append({"window": window, "iou": score})

    if not scored_windows:
        result.update(
            {
                "temporal_pass": False,
                "expected_start_ms": expected_start_ms,
                "expected_end_ms": expected_end_ms,
            }
        )
        return result

    top1 = scored_windows[0]
    oracle = max(scored_windows, key=lambda item: item["iou"])

    result.update(
        {
            "expected_start_ms": expected_start_ms,
            "expected_end_ms": expected_end_ms,
            "minimum_top1_iou": minimum_top1_iou,
            "minimum_oracle_iou": minimum_oracle_iou,
            "top1_iou": round(top1["iou"], 6),
            "oracle_iou": round(oracle["iou"], 6),
            "top1_window_index": top1["window"].get("window_index"),
            "oracle_window_index": oracle["window"].get("window_index"),
        }
    )
    result["temporal_pass"] = (
        top1["iou"] >= minimum_top1_iou
        and oracle["iou"] >= minimum_oracle_iou
    )
    return result


def make_csv_row(result: dict[str, Any]) -> dict[str, Any]:
    answer = result.get("answer", {})
    grounding = result.get("grounding", {})
    return {
        "id": result["id"],
        "category": result["category"],
        "question": result["question"],
        "expected_answerable": result["expected_answerable"],
        "predicted_answerable": answer.get("predicted_answerable"),
        "answerability_pass": answer.get("answerability_pass"),
        "answer_text_pass": answer.get("answer_text_pass"),
        "answerable_window_count": grounding.get(
            "answerable_window_count"
        ),
        "window_count_pass": grounding.get("window_count_pass"),
        "top1_iou": grounding.get("top1_iou"),
        "oracle_iou": grounding.get("oracle_iou"),
        "temporal_pass": grounding.get("temporal_pass"),
        "case_pass": result.get("case_pass"),
        "answer": answer.get("answer", ""),
        "error": result.get("error", ""),
    }


def summarize_results(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    total = len(results)

    def pass_count(section: str | None, key: str) -> int:
        count = 0
        for result in results:
            value = (
                result.get(key)
                if section is None
                else result.get(section, {}).get(key)
            )
            if value is True:
                count += 1
        return count

    top1_ious = [
        float(result["grounding"]["top1_iou"])
        for result in results
        if result.get("grounding", {}).get("top1_iou") is not None
    ]
    oracle_ious = [
        float(result["grounding"]["oracle_iou"])
        for result in results
        if result.get("grounding", {}).get("oracle_iou") is not None
    ]

    category_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        category_groups[result["category"]].append(result)

    categories: dict[str, Any] = {}
    for category, category_results in sorted(category_groups.items()):
        passed = sum(
            result.get("case_pass") is True
            for result in category_results
        )
        categories[category] = {
            "total": len(category_results),
            "passed": passed,
            "pass_rate": passed / len(category_results),
        }

    cases_passed = pass_count(None, "case_pass")
    answerability_passed = pass_count("answer", "answerability_pass")
    answer_text_passed = pass_count("answer", "answer_text_pass")
    temporal_passed = pass_count("grounding", "temporal_pass")
    window_count_passed = pass_count("grounding", "window_count_pass")

    return {
        "total_cases": total,
        "cases_passed": cases_passed,
        "case_pass_rate": cases_passed / total if total else 0.0,
        "answerability_accuracy": (
            answerability_passed / total if total else 0.0
        ),
        "answer_text_accuracy": (
            answer_text_passed / total if total else 0.0
        ),
        "temporal_pass_rate": (
            temporal_passed / total if total else 0.0
        ),
        "window_count_pass_rate": (
            window_count_passed / total if total else 0.0
        ),
        "mean_top1_iou": (
            sum(top1_ious) / len(top1_ious) if top1_ious else None
        ),
        "mean_oracle_iou": (
            sum(oracle_ious) / len(oracle_ious)
            if oracle_ious
            else None
        ),
        "categories": categories,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print()
    print("=" * 72)
    print("MILESTONE 7 EVALUATION SUMMARY")
    print("=" * 72)
    print(
        f"Cases passed: {summary['cases_passed']}/"
        f"{summary['total_cases']} "
        f"({summary['case_pass_rate']:.1%})"
    )
    print(
        f"Answerability accuracy: "
        f"{summary['answerability_accuracy']:.1%}"
    )
    print(
        f"Answer text accuracy: "
        f"{summary['answer_text_accuracy']:.1%}"
    )
    print(
        f"Temporal pass rate: "
        f"{summary['temporal_pass_rate']:.1%}"
    )
    print(
        f"Grounding-window discipline: "
        f"{summary['window_count_pass_rate']:.1%}"
    )
    if summary["mean_top1_iou"] is not None:
        print(
            f"Mean top-1 temporal IoU: "
            f"{summary['mean_top1_iou']:.4f}"
        )
    if summary["mean_oracle_iou"] is not None:
        print(
            f"Mean oracle temporal IoU: "
            f"{summary['mean_oracle_iou']:.4f}"
        )
    print()
    print("Per category:")
    for category, item in summary["categories"].items():
        print(
            f"- {category}: {item['passed']}/{item['total']} "
            f"({item['pass_rate']:.1%})"
        )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate answer synthesis and temporal grounding "
            "over a labeled video-QA set."
        )
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("eval_cases.json"),
    )
    parser.add_argument(
        "--index-metadata",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--answer-script",
        type=Path,
        default=Path("src/video_qa/answer.py"),
    )
    parser.add_argument(
        "--ground-script",
        type=Path,
        default=Path("src/video_qa/temporal_ground.py"),
    )
    parser.add_argument(
        "--model",
        default="gemini-3.1-flash-lite",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("eval_output"),
    )
    parser.add_argument(
        "--maximum-windows",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run only this case ID. Repeat for multiple cases.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--text-only-answer",
        action="store_true",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
    )
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    project_directory = Path.cwd().resolve()

    cases_path = args.cases.resolve()
    metadata_path = args.index_metadata.resolve()
    video_path = args.video.resolve()
    answer_script = args.answer_script.resolve()
    ground_script = args.ground_script.resolve()
    output_directory = args.output_directory.resolve()

    for required_path in (
        cases_path,
        metadata_path,
        video_path,
        answer_script,
        ground_script,
    ):
        if not required_path.exists():
            raise FileNotFoundError(
                f"Required path does not exist: {required_path}"
            )

    cases = load_json(cases_path)
    if not isinstance(cases, list):
        raise ValueError("The cases file must contain a JSON list.")

    if args.case_id:
        requested_ids = set(args.case_id)
        cases = [
            case for case in cases if str(case.get("id")) in requested_ids
        ]
        found_ids = {str(case.get("id")) for case in cases}
        missing_ids = requested_ids - found_ids
        if missing_ids:
            raise ValueError(
                "Unknown case IDs: " + ", ".join(sorted(missing_ids))
            )

    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1.")
        cases = cases[: args.limit]

    if not cases:
        raise ValueError("No evaluation cases selected.")

    output_directory.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []

    for position, case in enumerate(cases, start=1):
        case_id = str(case["id"])
        question = str(case["question"])
        category = str(case.get("category", "uncategorized"))
        expected_answerable = bool(case["expected_answerable"])

        print()
        print("#" * 72)
        print(f"CASE {position}/{len(cases)}: {case_id}")
        print("#" * 72)
        print(question)

        case_directory = output_directory / case_id
        case_directory.mkdir(parents=True, exist_ok=True)

        answer_path = case_directory / "answer.json"
        answer_log_path = case_directory / "answer.log"
        evidence_copy_path = case_directory / "evidence_context.json"
        grounding_directory = case_directory / "grounding"
        grounding_path = build_grounding_output_path(
            grounding_directory,
            question,
        )
        grounding_log_path = case_directory / "grounding.log"

        case_result: dict[str, Any] = {
            "id": case_id,
            "category": category,
            "question": question,
            "expected_answerable": expected_answerable,
            "case_pass": False,
        }

        try:
            can_reuse = (
                args.skip_existing
                and answer_path.exists()
                and grounding_path.exists()
            )

            if not can_reuse:
                answer_command = [
                    sys.executable,
                    str(answer_script),
                    question,
                    "--index-metadata",
                    str(metadata_path),
                    "--model",
                    args.model,
                    "--output",
                    str(answer_path),
                ]
                if args.text_only_answer:
                    answer_command.append("--text-only")

                run_command(
                    answer_command,
                    project_directory,
                    answer_log_path,
                )

                answer_payload = load_json(answer_path)
                evidence_source_path = Path(
                    answer_payload["evidence_context_path"]
                )
                if not evidence_source_path.is_absolute():
                    evidence_source_path = (
                        project_directory / evidence_source_path
                    ).resolve()
                if not evidence_source_path.exists():
                    raise FileNotFoundError(
                        "Evidence context does not exist: "
                        f"{evidence_source_path}"
                    )
                shutil.copy2(evidence_source_path, evidence_copy_path)

                grounding_command = [
                    sys.executable,
                    str(ground_script),
                    question,
                    "--evidence",
                    str(evidence_copy_path),
                    "--index-metadata",
                    str(metadata_path),
                    "--model",
                    args.model,
                    "--video",
                    str(video_path),
                    "--output-directory",
                    str(grounding_directory),
                    "--maximum-windows",
                    str(args.maximum_windows),
                ]
                run_command(
                    grounding_command,
                    project_directory,
                    grounding_log_path,
                )

            answer_payload = load_json(answer_path)
            grounding_payload = load_json(grounding_path)

            predicted_answerable = bool(
                answer_payload.get("answerable")
            )
            answerability_pass = (
                predicted_answerable == expected_answerable
            )

            if expected_answerable:
                answer_text_pass, text_details = check_answer_text(
                    str(answer_payload.get("answer", "")),
                    case.get("answer_check"),
                )
            else:
                answer_text_pass = not predicted_answerable
                text_details = {
                    "checked": False,
                    "negative_case": True,
                }

            grounding_score = score_grounding(
                grounding_payload,
                case,
            )
            case_pass = (
                answerability_pass
                and answer_text_pass
                and grounding_score["temporal_pass"]
                and grounding_score["window_count_pass"]
            )

            case_result.update(
                {
                    "answer": {
                        "predicted_answerable": predicted_answerable,
                        "answerability_pass": answerability_pass,
                        "answer_text_pass": answer_text_pass,
                        "answer": answer_payload.get("answer", ""),
                        "confidence": answer_payload.get("confidence"),
                        "citations": answer_payload.get("citations", []),
                        "limitations": answer_payload.get(
                            "limitations",
                            [],
                        ),
                        "text_check": text_details,
                    },
                    "grounding": grounding_score,
                    "case_pass": case_pass,
                    "artifacts": {
                        "answer_json": str(answer_path),
                        "answer_log": str(answer_log_path),
                        "evidence_json": str(evidence_copy_path),
                        "grounding_json": str(grounding_path),
                        "grounding_log": str(grounding_log_path),
                    },
                }
            )

            print()
            print(f"[case result] {'PASS' if case_pass else 'FAIL'}")
            print(f"[answerability] {answerability_pass}")
            print(f"[answer text] {answer_text_pass}")
            print(f"[temporal] {grounding_score['temporal_pass']}")
            print(
                f"[answerable windows] "
                f"{grounding_score['answerable_window_count']}"
            )
            if grounding_score.get("top1_iou") is not None:
                print(
                    f"[top-1 IoU] "
                    f"{grounding_score['top1_iou']:.4f}"
                )
            if grounding_score.get("oracle_iou") is not None:
                print(
                    f"[oracle IoU] "
                    f"{grounding_score['oracle_iou']:.4f}"
                )

        except Exception as error:
            case_result["error"] = str(error)
            print()
            print(f"[case error] {error}")

        save_json(case_directory / "case_result.json", case_result)
        results.append(case_result)

    summary = summarize_results(results)
    report = {
        "model": args.model,
        "cases_path": str(cases_path),
        "index_metadata_path": str(metadata_path),
        "video_path": str(video_path),
        "summary": summary,
        "results": results,
    }

    report_path = output_directory / "evaluation_report.json"
    save_json(report_path, report)

    csv_path = output_directory / "evaluation_report.csv"
    csv_rows = [make_csv_row(result) for result in results]
    with csv_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(csv_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print_summary(summary)
    print()
    print(f"[report] {report_path}")
    print(f"[csv]    {csv_path}")

    if (
        args.fail_on_regression
        and summary["cases_passed"] != summary["total_cases"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())