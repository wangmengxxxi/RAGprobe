"""Comparison module placeholder for v0.1."""

from __future__ import annotations

from ragprobe.core.models import DiagnosticReport


def compare_reports(before: DiagnosticReport, after: DiagnosticReport) -> DiagnosticReport:
    """Compare two diagnostic reports.

    The concrete delta model is intentionally deferred until v0.1.
    """
    raise NotImplementedError("Report comparison will be implemented in v0.1.")
