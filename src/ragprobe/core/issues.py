"""System issue detection for v0.4 evidence-backed diagnostics."""

from __future__ import annotations

from ragprobe.core.confusion import is_metadata_confusion_type
from ragprobe.core.models import FailurePattern, SystemIssue, TestSet


def detect_system_issues(
    *,
    hit_rate: float,
    mrr: float,
    fpr: float,
    failure_patterns: list[FailurePattern],
    confusion_distribution: dict[str, float],
    match_stats: dict[str, int],
    testset: TestSet,
) -> list[SystemIssue]:
    issues: list[SystemIssue] = []

    if fpr >= 0.3:
        issues.append(
            SystemIssue(
                issue_type="hard_negative_confusion",
                severity="high",
                evidence=f"FPR is {fpr:.2f}; hard negatives are frequently retrieved.",
                affected_percentage=fpr,
            )
        )

    ranking_pattern = _pattern(failure_patterns, "ranking_weakness")
    if (hit_rate >= 0.7 and mrr < 0.5) or ranking_pattern is not None:
        affected = (
            ranking_pattern.affected_percentage
            if ranking_pattern
            else max(hit_rate - mrr, 0.0)
        )
        issues.append(
            SystemIssue(
                issue_type="ranking_weakness",
                severity="high" if affected >= 0.3 else "medium",
                evidence=(
                    f"Hit rate is {hit_rate:.2f} while MRR is {mrr:.2f}; "
                    "expected chunks are present but often not early."
                ),
                affected_percentage=affected,
            )
        )

    miss_pattern = _pattern(failure_patterns, "retrieval_miss")
    hard_miss_pattern = _pattern(failure_patterns, "hard_negative_without_expected")
    if hit_rate < 0.7 or miss_pattern or hard_miss_pattern:
        affected = 1 - hit_rate
        issues.append(
            SystemIssue(
                issue_type="low_recall_coverage",
                severity="high" if hit_rate < 0.5 else "medium",
                evidence=f"Hit rate is {hit_rate:.2f}; expected chunks are missing in many cases.",
                affected_percentage=affected,
            )
        )

    metadata_confusions = {
        key: value
        for key, value in confusion_distribution.items()
        if is_metadata_confusion_type(key)
    }
    if metadata_confusions and max(metadata_confusions.values()) >= 0.3:
        label, value = max(metadata_confusions.items(), key=lambda item: item[1])
        issues.append(
            SystemIssue(
                issue_type="metadata_filter_needed",
                severity="medium",
                evidence=f"{label} accounts for {value:.1%} of hard negative confusions.",
                affected_percentage=value,
            )
        )

    total_matches = sum(match_stats.values())
    low_confidence = match_stats.get("content_fallback", 0) + match_stats.get("unmatched", 0)
    low_confidence_rate = low_confidence / total_matches if total_matches else 0.0
    if low_confidence_rate >= 0.2:
        issues.append(
            SystemIssue(
                issue_type="low_confidence_matching",
                severity="medium",
                evidence=(
                    f"{low_confidence_rate:.1%} of retrieved chunks used content fallback "
                    "or could not be matched; prefer stable chunk_id output."
                ),
                affected_percentage=low_confidence_rate,
            )
        )

    quality_gap = _testset_quality_gap(testset)
    if quality_gap:
        issues.append(quality_gap)

    return _dedupe_issues(issues)


def _pattern(patterns: list[FailurePattern], pattern_type: str) -> FailurePattern | None:
    return next((item for item in patterns if item.pattern_type == pattern_type), None)


def _testset_quality_gap(testset: TestSet) -> SystemIssue | None:
    if not testset.cases:
        return None
    cases_without_hard_negatives = [
        case.id for case in testset.cases if not case.hard_negatives
    ]
    chunks = testset.metadata.get("chunks")
    if cases_without_hard_negatives:
        affected = len(cases_without_hard_negatives) / len(testset.cases)
        return SystemIssue(
            issue_type="testset_quality_gap",
            severity="medium",
            evidence=(
                f"{len(cases_without_hard_negatives)} cases have no hard negatives; "
                "FPR coverage is limited."
            ),
            affected_percentage=affected,
        )
    if not isinstance(chunks, dict) or not chunks:
        return SystemIssue(
            issue_type="testset_quality_gap",
            severity="medium",
            evidence=(
                "testset.metadata.chunks is missing or empty; "
                "content-aware checks are limited."
            ),
            affected_percentage=1.0,
        )
    return None


def _dedupe_issues(issues: list[SystemIssue]) -> list[SystemIssue]:
    seen: set[str] = set()
    unique = []
    severity_order = {"high": 0, "medium": 1, "low": 2}
    for issue in sorted(
        issues,
        key=lambda item: (
            severity_order[item.severity],
            -item.affected_percentage,
            item.issue_type,
        ),
    ):
        if issue.issue_type in seen:
            continue
        seen.add(issue.issue_type)
        unique.append(issue)
    return unique
