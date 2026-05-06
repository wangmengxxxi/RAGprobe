from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MethodMetrics:
    method: str
    query_count: int
    evidence_hit_rate: float
    false_positive_rate: float
    boundary_exclusion_rate: float | None
    source_grounding_rate: float | None

    def as_row(self) -> dict[str, Any]:
        return {
            "Method": self.method,
            "EHR": self.evidence_hit_rate,
            "FPR": self.false_positive_rate,
            "BER": self.boundary_exclusion_rate,
            "SGR": self.source_grounding_rate,
        }


def compute_metrics(
    method: str,
    query_results: list[dict[str, Any]],
    *,
    include_boundary: bool,
    include_source: bool,
) -> MethodMetrics:
    query_count = len(query_results)
    if query_count == 0:
        return MethodMetrics(method, 0, 0.0, 0.0, None, None)

    hit_sum = 0.0
    fp_sum = 0.0
    ber_sum = 0.0
    sgr_values: list[float] = []

    for item in query_results:
        expected = set(item["expected_evidence"])
        hard_negatives = set(item["hard_negatives"])
        selected = set(item["selected"])
        excluded = set(item.get("excluded", []))

        hit_sum += 1.0 if expected & selected else 0.0
        fp_sum += len(selected & hard_negatives) / max(len(hard_negatives), 1)
        if include_boundary:
            ber_sum += len(excluded & hard_negatives) / max(len(hard_negatives), 1)
        if include_source:
            sgr_values.extend(item.get("source_grounding_rates", []))

    return MethodMetrics(
        method=method,
        query_count=query_count,
        evidence_hit_rate=round(hit_sum / query_count, 4),
        false_positive_rate=round(fp_sum / query_count, 4),
        boundary_exclusion_rate=round(ber_sum / query_count, 4) if include_boundary else None,
        source_grounding_rate=round(sum(sgr_values) / len(sgr_values), 4) if sgr_values else None,
    )


def format_markdown_table(metrics: list[MethodMetrics]) -> str:
    rows = [metric.as_row() for metric in metrics]
    headers = ["Method", "EHR", "FPR", "BER", "SGR"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        values = [_format_value(row[header]) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
