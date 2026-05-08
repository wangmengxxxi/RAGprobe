"""Stable schema metadata for RAGProbe v1 artifacts."""

from __future__ import annotations

RAGPROBE_VERSION = "1.0.0"

SCHEMA_CHUNKS = "ragprobe.chunks.v1"
SCHEMA_TESTSET = "ragprobe.testset.v1"
SCHEMA_RESULTS = "ragprobe.results.v1"
SCHEMA_DIAGNOSTIC_REPORT = "ragprobe.diagnostic_report.v1"
SCHEMA_COMPARISON_REPORT = "ragprobe.comparison_report.v1"
SCHEMA_EXPERIMENT_REPORT = "ragprobe.experiment_report.v1"
SCHEMA_AUDIT_REPORT = "ragprobe.audit_report.v1"
SCHEMA_REPAIR_PLAN = "ragprobe.repair_plan.v1"


def schema_metadata(schema_version: str) -> dict[str, str]:
    return {
        "schema_version": schema_version,
        "ragprobe_version": RAGPROBE_VERSION,
    }
