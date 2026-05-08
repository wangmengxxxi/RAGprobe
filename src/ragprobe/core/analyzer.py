"""Deterministic diagnostic analysis for offline retrieval results."""

from __future__ import annotations

from collections import Counter

from ragprobe.core.classifier import classify_failure_patterns
from ragprobe.core.issues import detect_system_issues
from ragprobe.core.matching import collect_match_stats
from ragprobe.core.models import (
    DiagnosticReport,
    FailureCase,
    MetricSignal,
    RetrievalResult,
    TestCase,
    TestSet,
)
from ragprobe.core.recommendations import build_recommendations
from ragprobe.core.schema import SCHEMA_DIAGNOSTIC_REPORT, schema_metadata
from ragprobe.core.validation import validate_results


class DiagnosticAnalyzer:
    """Analyze offline retrieval results against a test set."""

    def __init__(self, precision_ks: tuple[int, ...] = (5, 10)) -> None:
        self.precision_ks = precision_ks

    def analyze(self, testset: TestSet, results: list[RetrievalResult]) -> DiagnosticReport:
        validate_results(testset, results)
        results_by_case = {result.test_case_id: result for result in results}
        total_cases = len(testset.cases)

        if total_cases == 0:
            return DiagnosticReport(
                metadata={
                    **schema_metadata(SCHEMA_DIAGNOSTIC_REPORT),
                    "testset_name": testset.name,
                    "total_cases": 0,
                }
            )

        hits = 0
        reciprocal_rank_sum = 0.0
        precision_sums = {k: 0.0 for k in self.precision_ks}
        total_hard_negatives = 0
        false_positive_hits = 0
        confusion_counts: Counter[str] = Counter()
        failure_cases: list[FailureCase] = []
        match_stats = collect_match_stats(results)

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

            false_positives = [
                chunk_id for chunk_id in retrieved_ids if chunk_id in hard_negative_map
            ]
            total_hard_negatives += len(hard_negative_map)
            false_positive_hits += len(set(false_positives))
            for chunk_id in set(false_positives):
                confusion_counts[hard_negative_map[chunk_id].confusion_type] += 1

            for k in self.precision_ks:
                top_k_ids = retrieved_ids[:k]
                if top_k_ids:
                    precision_sums[k] += len([cid for cid in top_k_ids if cid in expected_ids]) / k

            if (not hit) or false_positives:
                failure_cases.append(
                    _build_failure_case(case, retrieved_ids, correct_rank, false_positives)
                )

        fpr = false_positive_hits / total_hard_negatives if total_hard_negatives else 0.0
        hit_rate = hits / total_cases
        mrr = reciprocal_rank_sum / total_cases
        ranked_failures = _rank_failures(failure_cases)
        confusion_distribution = _distribution(confusion_counts)
        failure_patterns = classify_failure_patterns(ranked_failures, results, total_cases)
        system_issues = detect_system_issues(
            hit_rate=hit_rate,
            mrr=mrr,
            fpr=fpr,
            failure_patterns=failure_patterns,
            confusion_distribution=confusion_distribution,
            match_stats=match_stats,
            testset=testset,
        )
        report = DiagnosticReport(
            hit_rate=hit_rate,
            mrr=mrr,
            precision_at_k={k: precision_sums[k] / total_cases for k in self.precision_ks},
            fpr=fpr,
            failure_cases=ranked_failures,
            failure_patterns=failure_patterns,
            confusion_distribution=confusion_distribution,
            system_issues=system_issues,
            metric_signals=_metric_signals(fpr=fpr, hit_rate=hit_rate, mrr=mrr),
            recommendations=build_recommendations(system_issues, failure_patterns),
            metadata={
                **schema_metadata(SCHEMA_DIAGNOSTIC_REPORT),
                "testset_name": testset.name,
                "total_cases": total_cases,
                "evaluated_cases": len(results_by_case),
                "hits": hits,
                "total_hard_negatives": total_hard_negatives,
                "false_positive_hits": false_positive_hits,
                "match_stats": match_stats,
                "low_confidence_match_rate": _low_confidence_match_rate(match_stats),
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


def _metric_signals(fpr: float, hit_rate: float, mrr: float) -> list[MetricSignal]:
    signals: list[MetricSignal] = []

    if hit_rate >= 0.7 and fpr >= 0.3:
        signals.append(
            MetricSignal(
                name="high_recall_high_confusion",
                severity="high",
                summary=(
                    "The retriever often finds expected chunks, but it also retrieves hard "
                    "negatives. This usually means recall is not the main bottleneck; ranking, "
                    "reranking, filtering, or chunk disambiguation may deserve attention."
                ),
                evidence=f"hit_rate={hit_rate:.3f}, fpr={fpr:.3f}",
            )
        )
    elif hit_rate < 0.7 and fpr < 0.3:
        signals.append(
            MetricSignal(
                name="low_recall_low_confusion",
                severity="medium",
                summary=(
                    "The retriever does not retrieve many expected chunks, but it is not "
                    "frequently pulling known hard negatives. This often points to recall "
                    "coverage, top-k, indexing, or query/document wording gaps."
                ),
                evidence=f"hit_rate={hit_rate:.3f}, fpr={fpr:.3f}",
            )
        )
    elif hit_rate < 0.7 and fpr >= 0.3:
        signals.append(
            MetricSignal(
                name="low_recall_high_confusion",
                severity="high",
                summary=(
                    "The retriever both misses expected chunks and retrieves confusing "
                    "negatives. This is a broad retrieval quality issue rather than a single "
                    "threshold tuning problem."
                ),
                evidence=f"hit_rate={hit_rate:.3f}, fpr={fpr:.3f}",
            )
        )

    if hit_rate >= 0.7 and mrr < 0.5:
        signals.append(
            MetricSignal(
                name="correct_but_ranked_low",
                severity="medium",
                summary=(
                    "Expected chunks are often present but not ranked early. This indicates "
                    "the candidate pool may be adequate while ranking quality is weak."
                ),
                evidence=f"hit_rate={hit_rate:.3f}, mrr={mrr:.3f}",
            )
        )

    if not signals:
        signals.append(
            MetricSignal(
                name="no_obvious_metric_warning",
                severity="low",
                summary=(
                    "No obvious metric-level warning was detected from hit rate, MRR, and FPR. "
                    "Review worst cases and confusion distribution for finer-grained issues."
                ),
                evidence=f"hit_rate={hit_rate:.3f}, mrr={mrr:.3f}, fpr={fpr:.3f}",
            )
        )

    return signals


def _low_confidence_match_rate(match_stats: dict[str, int]) -> float:
    total = sum(match_stats.values())
    if not total:
        return 0.0
    low_confidence = match_stats.get("content_fallback", 0) + match_stats.get("unmatched", 0)
    return low_confidence / total
