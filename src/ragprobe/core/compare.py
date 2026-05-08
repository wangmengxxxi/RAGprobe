"""Compare two diagnostic reports."""

from __future__ import annotations

from ragprobe.core.models import ComparisonReport, DiagnosticReport, MetricDelta
from ragprobe.core.schema import SCHEMA_COMPARISON_REPORT, schema_metadata


def compare_reports(before: DiagnosticReport, after: DiagnosticReport) -> ComparisonReport:
    """Compare two diagnostic reports."""
    deltas = [
        MetricDelta("hit_rate", before.hit_rate, after.hit_rate, after.hit_rate - before.hit_rate),
        MetricDelta("mrr", before.mrr, after.mrr, after.mrr - before.mrr),
        MetricDelta("fpr", before.fpr, after.fpr, after.fpr - before.fpr),
    ]
    before_failures = {case.test_case_id for case in before.failure_cases}
    after_failures = {case.test_case_id for case in after.failure_cases}
    return ComparisonReport(
        before=before,
        after=after,
        deltas=deltas,
        improved_cases=sorted(before_failures - after_failures),
        regressed_cases=sorted(after_failures - before_failures),
        metadata={
            **schema_metadata(SCHEMA_COMPARISON_REPORT),
            "before_total_cases": before.metadata.get("total_cases"),
            "after_total_cases": after.metadata.get("total_cases"),
        },
    )
