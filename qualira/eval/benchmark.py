from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from qualira.backends.sqlite import SQLiteEvidenceBackend, load_evidence_file
from qualira.eval.baselines.naive_chunk import naive_topk
from qualira.eval.metrics import MethodMetrics, compute_metrics, format_markdown_table
from qualira.retrieval.executor import RetrievalExecutor


def load_queries(path: str | Path) -> list[dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["queries"] if isinstance(data, dict) else data


def run_benchmark(
    evidence_path: str | Path,
    queries_path: str | Path,
    *,
    db_path: str | Path = ":memory:",
    top_k: int = 5,
) -> tuple[list[MethodMetrics], list[dict[str, Any]], str]:
    units = load_evidence_file(evidence_path)
    queries = load_queries(queries_path)

    backend = SQLiteEvidenceBackend(db_path)
    backend.reset()
    backend.store_many(units)
    executor = RetrievalExecutor(backend)

    qualira_rows: list[dict[str, Any]] = []
    naive_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []

    for query_case in queries:
        query = query_case["query"]
        expected = query_case["expected_evidence"]
        hard_negatives = query_case["hard_negatives"]

        qualira_result = executor.execute(query, limit=top_k)
        qualira_selected = [item.unit.id for item in qualira_result.selected]
        qualira_excluded = [item.unit_id for item in qualira_result.excluded]
        qualira_rows.append(
            {
                "expected_evidence": expected,
                "hard_negatives": hard_negatives,
                "selected": qualira_selected,
                "excluded": qualira_excluded,
                "source_grounding_rates": [item.unit.claim_source_grounding_rate() for item in qualira_result.selected],
            }
        )

        naive_result = naive_topk(query, units, k=top_k)
        naive_rows.append(
            {
                "expected_evidence": expected,
                "hard_negatives": hard_negatives,
                "selected": [unit.id for unit, _score in naive_result],
                "excluded": [],
                "source_grounding_rates": [],
            }
        )

        details.append(
            {
                "query": query,
                "test_type": query_case.get("test_type"),
                "expected_evidence": expected,
                "hard_negatives": hard_negatives,
                "qualira_selected": qualira_selected,
                "qualira_excluded": qualira_excluded,
                "naive_selected": [unit.id for unit, _score in naive_result],
            }
        )

    metrics = [
        compute_metrics("naive_chunk_topk", naive_rows, include_boundary=False, include_source=False),
        compute_metrics("qualira", qualira_rows, include_boundary=True, include_source=True),
    ]
    return metrics, details, format_markdown_table(metrics)
