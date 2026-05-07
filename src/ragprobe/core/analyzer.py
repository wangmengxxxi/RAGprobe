"""Deterministic diagnostic analysis for offline retrieval results."""

from __future__ import annotations

from collections import Counter

from ragprobe.core.models import (
    DiagnosticReport,
    FailureCase,
    Recommendation,
    RetrievalResult,
    SystemIssue,
    TestCase,
    TestSet,
)


class DiagnosticAnalyzer:
    """Analyze offline retrieval results against a test set."""

    def __init__(self, precision_ks: tuple[int, ...] = (5, 10)) -> None:
        self.precision_ks = precision_ks

    def analyze(self, testset: TestSet, results: list[RetrievalResult]) -> DiagnosticReport:
        cases_by_id = {case.id: case for case in testset.cases}
        results_by_case = {result.test_case_id: result for result in results}
        total_cases = len(testset.cases)

        if total_cases == 0:
            return DiagnosticReport(metadata={"testset_name": testset.name, "total_cases": 0})

        hits = 0
        reciprocal_rank_sum = 0.0
        precision_sums = {k: 0.0 for k in self.precision_ks}
        total_hard_negatives = 0
        false_positive_hits = 0
        confusion_counts: Counter[str] = Counter()
        failure_cases: list[FailureCase] = []

        for case in testset.cases:
            result = results_by_case.get(case.id)
            if result is None:
                failure_cases.append(
                    FailureCase(
                        test_case_id=case.id,
                        failure_type="miss",
                        query=case.query,
                        difficulty=case.difficulty,
                    )
                )
                continue

            expected_ids = set(case.expected_chunks)
            hard_negative_map = {hn.chunk_id: hn for hn in case.hard_negatives}
            retrieved_ids = [chunk.chunk_id for chunk in result.retrieved if chunk.chunk_id]

            correct_rank = _find_first_rank(retrieved_ids, expected_ids)
            hit = correct_rank is not None
            if hit:
                hits += 1
                reciprocal_rank_sum += 1 / correct_rank

            false_positives = [chunk_id for chunk_id in retrieved_ids if chunk_id in hard_negative_map]
            total_hard_negatives += len(hard_negative_map)
            false_positive_hits += len(set(false_positives))
            for chunk_id in set(false_positives):
                confusion_counts[hard_negative_map[chunk_id].confusion_type] += 1

            for k in self.precision_ks:
                top_k_ids = retrieved_ids[:k]
                if top_k_ids:
                    precision_sums[k] += len([cid for cid in top_k_ids if cid in expected_ids]) / k

            if (not hit) or false_positives:
                failure_cases.append(_build_failure_case(case, retrieved_ids, correct_rank, false_positives))

        fpr = false_positive_hits / total_hard_negatives if total_hard_negatives else 0.0
        report = DiagnosticReport(
            hit_rate=hits / total_cases,
            mrr=reciprocal_rank_sum / total_cases,
            precision_at_k={k: precision_sums[k] / total_cases for k in self.precision_ks},
            fpr=fpr,
            failure_cases=_rank_failures(failure_cases),
            confusion_distribution=_distribution(confusion_counts),
            system_issues=_detect_basic_issues(fpr=fpr, hit_rate=hits / total_cases),
            recommendations=_basic_recommendations(fpr=fpr, hit_rate=hits / total_cases),
            metadata={
                "testset_name": testset.name,
                "total_cases": total_cases,
                "evaluated_cases": len(results_by_case),
                "hits": hits,
                "total_hard_negatives": total_hard_negatives,
                "false_positive_hits": false_positive_hits,
            },
        )
        return report


def _find_first_rank(retrieved_ids: list[str], expected_ids: set[str]) -> int | None:
    for index, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in expected_ids:
            return index
    return None


def _build_failure_case(
    case: TestCase,
    retrieved_ids: list[str],
    correct_rank: int | None,
    false_positives: list[str],
) -> FailureCase:
    if correct_rank is None and false_positives:
        failure_type = "both"
    elif correct_rank is None:
        failure_type = "miss"
    else:
        failure_type = "false_positive"

    hard_negative_map = {hn.chunk_id: hn for hn in case.hard_negatives}
    first_false_positive = false_positives[0] if false_positives else None
    hard_negative = hard_negative_map.get(first_false_positive or "")

    return FailureCase(
        test_case_id=case.id,
        failure_type=failure_type,
        query=case.query,
        confusion_type=hard_negative.confusion_type if hard_negative else None,
        false_positive_similarity=hard_negative.similarity_to_correct if hard_negative else None,
        correct_rank=correct_rank,
        false_positives=false_positives,
        retrieved_ids=retrieved_ids,
        difficulty=case.difficulty,
    )


def _rank_failures(failures: list[FailureCase]) -> list[FailureCase]:
    priority = {"both": 0, "miss": 1, "false_positive": 2}
    return sorted(
        failures,
        key=lambda item: (
            priority[item.failure_type],
            item.correct_rank if item.correct_rank is not None else 999,
            item.test_case_id,
        ),
    )


def _distribution(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}


def _detect_basic_issues(fpr: float, hit_rate: float) -> list[SystemIssue]:
    issues = []
    if fpr >= 0.3:
        issues.append(
            SystemIssue(
                issue_type="hard_negative_confusion",
                severity="high",
                evidence=f"FPR is {fpr:.2f}, indicating frequent hard negative retrieval.",
                affected_percentage=fpr,
            )
        )
    if hit_rate < 0.7:
        issues.append(
            SystemIssue(
                issue_type="low_hit_rate",
                severity="high" if hit_rate < 0.5 else "medium",
                evidence=f"Hit rate is {hit_rate:.2f}, below the default v0.1 target of 0.70.",
                affected_percentage=1 - hit_rate,
            )
        )
    return issues


def _basic_recommendations(fpr: float, hit_rate: float) -> list[Recommendation]:
    recommendations = []
    if fpr >= 0.3:
        recommendations.append(
            Recommendation(
                priority=1,
                action="Inspect hard negative cases and consider reranking or stricter metadata filters.",
                expected_impact="Reduce retrieval of similar but incorrect chunks.",
                effort="medium",
            )
        )
    if hit_rate < 0.7:
        recommendations.append(
            Recommendation(
                priority=2,
                action="Review missed cases for chunking, query wording, or top-k coverage problems.",
                expected_impact="Improve the rate of retrieving expected chunks.",
                effort="medium",
            )
        )
    return recommendations
