"""Threshold check module placeholder for v0.1."""

from __future__ import annotations

from dataclasses import dataclass

from ragprobe.core.models import DiagnosticReport


@dataclass
class CheckResult:
    passed: bool
    messages: list[str]


def check_thresholds(
    report: DiagnosticReport,
    min_hit_rate: float | None = None,
    max_fpr: float | None = None,
) -> CheckResult:
    raise NotImplementedError("Threshold checks will be implemented in v0.1.")
