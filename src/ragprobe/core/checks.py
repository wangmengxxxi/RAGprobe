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
    min_mrr: float | None = None,
    max_fpr: float | None = None,
    max_low_confidence_match_rate: float | None = None,
) -> CheckResult:
    messages = []
    passed = True

    if min_hit_rate is not None and report.hit_rate < min_hit_rate:
        passed = False
        messages.append(f"hit_rate {report.hit_rate:.3f} is below minimum {min_hit_rate:.3f}")

    if min_mrr is not None and report.mrr < min_mrr:
        passed = False
        messages.append(f"mrr {report.mrr:.3f} is below minimum {min_mrr:.3f}")

    if max_fpr is not None and report.fpr > max_fpr:
        passed = False
        messages.append(f"fpr {report.fpr:.3f} is above maximum {max_fpr:.3f}")

    low_confidence_rate = float(report.metadata.get("low_confidence_match_rate", 0.0))
    if (
        max_low_confidence_match_rate is not None
        and low_confidence_rate > max_low_confidence_match_rate
    ):
        passed = False
        messages.append(
            "low_confidence_match_rate "
            f"{low_confidence_rate:.3f} is above maximum "
            f"{max_low_confidence_match_rate:.3f}"
        )

    if passed:
        messages.append("all thresholds passed")

    return CheckResult(passed=passed, messages=messages)
