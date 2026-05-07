"""Diagnostic analyzer placeholder.

Phase 0 only defines the module boundary. Metric computation belongs to the
v0.1 Diagnostic Core implementation pass.
"""

from __future__ import annotations

from ragprobe.core.models import DiagnosticReport, RetrievalResult, TestSet


class DiagnosticAnalyzer:
    """Analyze offline retrieval results against a test set."""

    def analyze(self, testset: TestSet, results: list[RetrievalResult]) -> DiagnosticReport:
        raise NotImplementedError("Diagnostic analysis will be implemented in v0.1.")
