"""Threshold checks for CI-style diagnostics."""

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
    messages = []
    passed = True

    if min_hit_rate is not None and report.hit_rate < min_hit_rate:
        passed = False
        messages.append(f"hit_rate {report.hit_rate:.3f} is below minimum {min_hit_rate:.3f}")

    if max_fpr is not None and report.fpr > max_fpr:
        passed = False
        messages.append(f"fpr {report.fpr:.3f} is above maximum {max_fpr:.3f}")

    if passed:
        messages.append("all thresholds passed")

    return CheckResult(passed=passed, messages=messages)
