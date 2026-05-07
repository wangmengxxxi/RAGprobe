"""Evidence-backed reference recommendations for v0.4 diagnostics."""

from __future__ import annotations

from ragprobe.core.models import FailurePattern, Recommendation, SystemIssue


def build_recommendations(
    issues: list[SystemIssue],
    failure_patterns: list[FailurePattern],
) -> list[Recommendation]:
    recommendations: list[Recommendation] = []
    issue_map = {issue.issue_type: issue for issue in issues}
    pattern_map = {pattern.pattern_type: pattern for pattern in failure_patterns}

    if "hard_negative_confusion" in issue_map:
        recommendations.append(
            Recommendation(
                priority=1,
                action=(
                    "Review hard negative-heavy cases and consider reranking "
                    "or stricter filters."
                ),
                evidence=issue_map["hard_negative_confusion"].evidence,
                effort="medium",
            )
        )

    if "hard_negative_ranked_above_correct" in pattern_map:
        recommendations.append(
            Recommendation(
                priority=2,
                action=(
                    "Prioritize ranking diagnostics for cases where wrong chunks "
                    "outrank expected chunks."
                ),
                evidence=pattern_map["hard_negative_ranked_above_correct"].evidence,
                effort="medium",
            )
        )

    if "ranking_weakness" in issue_map:
        recommendations.append(
            Recommendation(
                priority=3,
                action=(
                    "Inspect ranking features, reranker behavior, or score fusion "
                    "for low-rank hits."
                ),
                evidence=issue_map["ranking_weakness"].evidence,
                effort="medium",
            )
        )

    if "low_recall_coverage" in issue_map:
        recommendations.append(
            Recommendation(
                priority=4,
                action=(
                    "Review missed cases for query wording, index coverage, "
                    "chunking, or top-k coverage."
                ),
                evidence=issue_map["low_recall_coverage"].evidence,
                effort="medium",
            )
        )

    if "metadata_filter_needed" in issue_map:
        recommendations.append(
            Recommendation(
                priority=5,
                action=(
                    "Consider metadata-aware filtering for concentrated subject, "
                    "scope, or version confusions."
                ),
                evidence=issue_map["metadata_filter_needed"].evidence,
                effort="medium",
            )
        )

    if "low_confidence_matching" in issue_map:
        recommendations.append(
            Recommendation(
                priority=6,
                action=(
                    "Return stable chunk_id values from the retriever before "
                    "trusting fine-grained diagnosis."
                ),
                evidence=issue_map["low_confidence_matching"].evidence,
                effort="low",
            )
        )

    if "testset_quality_gap" in issue_map:
        recommendations.append(
            Recommendation(
                priority=7,
                action="Add hard negatives and metadata.chunks to strengthen regression coverage.",
                evidence=issue_map["testset_quality_gap"].evidence,
                effort="low",
            )
        )

    return _dedupe_recommendations(recommendations)


def _dedupe_recommendations(items: list[Recommendation]) -> list[Recommendation]:
    seen: set[str] = set()
    unique = []
    for item in sorted(items, key=lambda rec: rec.priority):
        if item.action in seen:
            continue
        seen.add(item.action)
        unique.append(item)
    return unique
