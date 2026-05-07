"""JSON artifact IO for v0.1 offline diagnostics."""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ragprobe.core.models import (
    ComparisonReport,
    DiagnosticReport,
    FailureCase,
    HardNegative,
    MetricDelta,
    Recommendation,
    RetrievalResult,
    RetrievedChunk,
    SystemIssue,
    TestCase,
    TestSet,
)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        json.dump(to_jsonable(data), file, ensure_ascii=False, indent=2)
        file.write("\n")


def to_jsonable(data: Any) -> Any:
    if is_dataclass(data):
        return to_jsonable(asdict(data))
    if isinstance(data, dict):
        return {str(key): to_jsonable(value) for key, value in data.items()}
    if isinstance(data, list):
        return [to_jsonable(item) for item in data]
    return data


def load_testset(path: str | Path) -> TestSet:
    payload = load_json(path)
    cases_payload = payload["cases"] if isinstance(payload, dict) else payload
    cases = []
    for case_data in cases_payload:
        hard_negatives = [
            HardNegative(
                chunk_id=item["chunk_id"],
                confusion_type=item["confusion_type"],
                similarity_to_correct=item.get("similarity_to_correct"),
                reason=item.get("reason", ""),
            )
            for item in case_data.get("hard_negatives", [])
        ]
        cases.append(
            TestCase(
                id=case_data["id"],
                query=case_data["query"],
                expected_chunks=list(case_data.get("expected_chunks", [])),
                hard_negatives=hard_negatives,
                difficulty=case_data.get("difficulty", "medium"),
                source_document=case_data.get("source_document", ""),
                metadata=dict(case_data.get("metadata", {})),
            )
        )
    return TestSet(
        cases=cases,
        name=payload.get("name", "") if isinstance(payload, dict) else "",
        metadata=dict(payload.get("metadata", {})) if isinstance(payload, dict) else {},
    )


def load_results(path: str | Path) -> list[RetrievalResult]:
    payload = load_json(path)
    rows = payload.get("results", payload) if isinstance(payload, dict) else payload
    results = []
    for row in rows:
        retrieved = [
            RetrievedChunk(
                content=item.get("content", ""),
                score=float(item.get("score", 0.0)),
                metadata=dict(item.get("metadata", {})),
                chunk_id=item.get("chunk_id"),
            )
            for item in row.get("retrieved", [])
        ]
        results.append(
            RetrievalResult(
                test_case_id=row["test_case_id"],
                query=row.get("query", ""),
                retrieved=retrieved,
                hit=row.get("hit"),
                correct_rank=row.get("correct_rank"),
                false_positives=list(row.get("false_positives", [])),
            )
        )
    return results


def diagnostic_report_from_dict(data: dict[str, Any]) -> DiagnosticReport:
    return DiagnosticReport(
        hit_rate=float(data.get("hit_rate", 0.0)),
        mrr=float(data.get("mrr", 0.0)),
        precision_at_k={int(k): float(v) for k, v in data.get("precision_at_k", {}).items()},
        fpr=float(data.get("fpr", 0.0)),
        failure_cases=[
            FailureCase(
                test_case_id=item["test_case_id"],
                failure_type=item["failure_type"],
                query=item.get("query", ""),
                confusion_type=item.get("confusion_type"),
                correct_chunk_similarity=item.get("correct_chunk_similarity"),
                false_positive_similarity=item.get("false_positive_similarity"),
                correct_rank=item.get("correct_rank"),
                false_positives=list(item.get("false_positives", [])),
                retrieved_ids=list(item.get("retrieved_ids", [])),
                difficulty=item.get("difficulty"),
            )
            for item in data.get("failure_cases", [])
        ],
        confusion_distribution={
            str(key): float(value) for key, value in data.get("confusion_distribution", {}).items()
        },
        system_issues=[
            SystemIssue(
                issue_type=item["issue_type"],
                severity=item["severity"],
                evidence=item.get("evidence", ""),
                affected_percentage=float(item.get("affected_percentage", 0.0)),
            )
            for item in data.get("system_issues", [])
        ],
        recommendations=[
            Recommendation(
                priority=int(item["priority"]),
                action=item["action"],
                expected_impact=item.get("expected_impact", ""),
                effort=item.get("effort", ""),
            )
            for item in data.get("recommendations", [])
        ],
        metadata=dict(data.get("metadata", {})),
    )


def load_report(path: str | Path) -> DiagnosticReport:
    return diagnostic_report_from_dict(load_json(path))


def comparison_report_from_dict(data: dict[str, Any]) -> ComparisonReport:
    return ComparisonReport(
        before=diagnostic_report_from_dict(data["before"]),
        after=diagnostic_report_from_dict(data["after"]),
        deltas=[
            MetricDelta(
                metric=item["metric"],
                before=float(item["before"]),
                after=float(item["after"]),
                delta=float(item["delta"]),
            )
            for item in data.get("deltas", [])
        ],
        improved_cases=list(data.get("improved_cases", [])),
        regressed_cases=list(data.get("regressed_cases", [])),
        metadata=dict(data.get("metadata", {})),
    )
