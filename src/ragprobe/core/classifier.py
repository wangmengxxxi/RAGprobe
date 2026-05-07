"""Rule-based failure pattern classification for v0.4 diagnostics."""

from __future__ import annotations

from collections import defaultdict

from ragprobe.core.models import FailureCase, FailurePattern, RetrievalResult


def classify_failure_patterns(
    failures: list[FailureCase],
    results: list[RetrievalResult],
    total_cases: int,
) -> list[FailurePattern]:
    if total_cases <= 0:
        return []

    grouped: dict[str, set[str]] = defaultdict(set)
    result_by_case = {result.test_case_id: result for result in results}

    for failure in failures:
        if failure.failure_type in {"miss", "both"}:
            if failure.false_positives:
                grouped["hard_negative_without_expected"].add(failure.test_case_id)
            else:
                grouped["retrieval_miss"].add(failure.test_case_id)

        if failure.correct_rank is not None and failure.correct_rank > 1:
            grouped["ranking_weakness"].add(failure.test_case_id)

        if _hard_negative_ranked_above_correct(failure):
            grouped["hard_negative_ranked_above_correct"].add(failure.test_case_id)

        result = result_by_case.get(failure.test_case_id)
        if result and _uses_low_confidence_matching(result):
            grouped["low_confidence_matching"].add(failure.test_case_id)

    patterns = [
        _build_pattern(pattern_type, sorted(case_ids), total_cases)
        for pattern_type, case_ids in grouped.items()
        if case_ids
    ]
    severity_order = {"high": 0, "medium": 1, "low": 2}
    patterns.sort(
        key=lambda item: (
            severity_order[item.severity],
            -item.affected_percentage,
            item.pattern_type,
        )
    )
    return patterns


def _hard_negative_ranked_above_correct(failure: FailureCase) -> bool:
    if failure.correct_rank is None or not failure.false_positives:
        return False
    false_positive_ranks = [
        failure.retrieved_ids.index(chunk_id) + 1
        for chunk_id in failure.false_positives
        if chunk_id in failure.retrieved_ids
    ]
    return bool(false_positive_ranks) and min(false_positive_ranks) < failure.correct_rank


def _uses_low_confidence_matching(result: RetrievalResult) -> bool:
    return any(
        chunk.metadata.get("ragprobe_match_method") == "content_fallback"
        or chunk.metadata.get("ragprobe_match_method") == "unmatched"
        for chunk in result.retrieved
    )


def _build_pattern(
    pattern_type: str,
    affected_cases: list[str],
    total_cases: int,
) -> FailurePattern:
    affected_percentage = len(affected_cases) / total_cases
    severity = _severity(affected_percentage)
    evidence = _evidence(pattern_type, affected_cases, affected_percentage)
    return FailurePattern(
        pattern_type=pattern_type,
        severity=severity,
        evidence=evidence,
        affected_cases=affected_cases,
        affected_percentage=affected_percentage,
    )


def _severity(affected_percentage: float) -> str:
    if affected_percentage >= 0.3:
        return "high"
    if affected_percentage >= 0.1:
        return "medium"
    return "low"


def _evidence(
    pattern_type: str,
    affected_cases: list[str],
    affected_percentage: float,
) -> str:
    case_text = ", ".join(affected_cases[:5])
    prefix = f"{len(affected_cases)} cases ({affected_percentage:.1%})"
    if pattern_type == "retrieval_miss":
        return f"{prefix} missed expected chunks: {case_text}."
    if pattern_type == "ranking_weakness":
        return f"{prefix} found expected chunks, but not at rank 1: {case_text}."
    if pattern_type == "hard_negative_ranked_above_correct":
        return f"{prefix} ranked hard negatives above expected chunks: {case_text}."
    if pattern_type == "hard_negative_without_expected":
        return f"{prefix} retrieved hard negatives while missing expected chunks: {case_text}."
    if pattern_type == "low_confidence_matching":
        return f"{prefix} relied on content fallback or unmatched retrieval output: {case_text}."
    return f"{prefix} matched pattern {pattern_type}: {case_text}."
