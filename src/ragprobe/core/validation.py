"""Validation helpers for v0.2 retrieval artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field

from ragprobe.core.models import RetrievalResult, TestSet


class ValidationError(ValueError):
    """Raised when a user supplied artifact does not match the v0.2 schema."""


@dataclass
class ValidationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate_testset(testset: TestSet) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()

    if not testset.cases:
        errors.append("testset must contain at least one case")

    for index, case in enumerate(testset.cases):
        if not case.id:
            errors.append(f"case[{index}] is missing id")
        elif case.id in seen_ids:
            errors.append(f"duplicate case id {case.id!r}")
        seen_ids.add(case.id)

        if not case.query:
            errors.append(f"case[{index}] is missing query")
        if not case.expected_chunks:
            errors.append(f"case[{index}] must include expected_chunks")
        if not case.hard_negatives:
            warnings.append(f"case[{index}] has no hard_negatives; FPR coverage will be limited")
        quality = case.metadata.get("quality", {})
        if isinstance(quality, dict):
            if quality.get("filter_passed") is False:
                warnings.append(f"case[{index}] failed generated quality filter")
            if float(quality.get("score", 1.0)) < 0.6:
                warnings.append(
                    f"case[{index}] has low generated quality score: {quality.get('score')}"
                )
            for warning in quality.get("warnings", []):
                warnings.append(f"case[{index}] quality warning: {warning}")

    chunks = testset.metadata.get("chunks")
    if chunks is None:
        warnings.append(
            "testset.metadata.chunks is missing; content fallback matching will be unavailable"
        )
    elif not isinstance(chunks, dict):
        errors.append("testset.metadata.chunks must be an object mapping chunk_id to content")
    else:
        known_chunks = set(chunks)
        for index, case in enumerate(testset.cases):
            missing_expected = [
                chunk_id for chunk_id in case.expected_chunks if chunk_id not in known_chunks
            ]
            if missing_expected:
                warnings.append(
                    f"case[{index}] expected_chunks are not present in metadata.chunks: "
                    f"{', '.join(missing_expected)}"
                )
            missing_negatives = [
                item.chunk_id for item in case.hard_negatives if item.chunk_id not in known_chunks
            ]
            if missing_negatives:
                warnings.append(
                    f"case[{index}] hard_negatives are not present in metadata.chunks: "
                    f"{', '.join(missing_negatives)}"
                )

    quality_summary = testset.metadata.get("quality_summary", {})
    if isinstance(quality_summary, dict):
        coverage = float(quality_summary.get("hard_negative_coverage", 1.0))
        average_score = float(quality_summary.get("average_quality_score", 1.0))
        if coverage < 0.8:
            warnings.append(
                f"hard negative coverage is {coverage:.1%}; generated testset may be weak"
            )
        if average_score < 0.7:
            warnings.append(
                f"average generated quality score is {average_score:.3f}; review before CI use"
            )

    return ValidationReport(valid=not errors, errors=errors, warnings=warnings)


def validate_results(testset: TestSet, results: list[RetrievalResult]) -> None:
    report = validate_results_report(testset, results)
    if not report.valid:
        raise ValidationError("; ".join(report.errors))


def validate_results_report(testset: TestSet, results: list[RetrievalResult]) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    case_ids = {case.id for case in testset.cases}
    seen_ids: set[str] = set()

    for index, result in enumerate(results):
        if not result.test_case_id:
            errors.append(f"result[{index}] is missing test_case_id")
        elif result.test_case_id not in case_ids:
            errors.append(f"result[{index}] references unknown case {result.test_case_id!r}")
        elif result.test_case_id in seen_ids:
            errors.append(f"duplicate result for case {result.test_case_id!r}")
        seen_ids.add(result.test_case_id)

        if not isinstance(result.retrieved, list):
            errors.append(f"result[{index}].retrieved must be a list")
            continue
        for chunk_index, chunk in enumerate(result.retrieved):
            if not chunk.content:
                errors.append(
                    f"result[{index}].retrieved[{chunk_index}] is missing required content"
                )
            if chunk.score is not None and not isinstance(chunk.score, int | float):
                errors.append(
                    f"result[{index}].retrieved[{chunk_index}] score must be numeric"
                )
            if not chunk.chunk_id:
                warnings.append(
                    f"result[{index}].retrieved[{chunk_index}] has no chunk_id; "
                    "content fallback may be used"
                )

    missing = sorted(case_ids - seen_ids)
    if missing:
        warnings.append(f"results missing {len(missing)} test cases: {', '.join(missing[:5])}")

    return ValidationReport(valid=not errors, errors=errors, warnings=warnings)
