"""Terminal report renderers."""

from __future__ import annotations

from ragprobe.core.models import ComparisonReport, DiagnosticReport


def render_terminal(report: DiagnosticReport) -> str:
    lines = [
        "RAGProbe Diagnostic Report",
        "=" * 28,
        "",
        "Retrieval Quality",
        f"  Hit Rate: {report.hit_rate:.3f}",
        f"  MRR:      {report.mrr:.3f}",
    ]
    for k, value in sorted(report.precision_at_k.items()):
        lines.append(f"  Precision@{k}: {value:.3f}")
    lines.extend(["", "Hard Negative Resistance", f"  FPR: {report.fpr:.3f}", ""])

    lines.append("Confusion Distribution")
    if report.confusion_distribution:
        for key, value in report.confusion_distribution.items():
            lines.append(f"  {key}: {value:.1%}")
    else:
        lines.append("  none")

    lines.extend(["", "Worst Cases"])
    if report.failure_cases:
        for index, case in enumerate(report.failure_cases[:5], start=1):
            false_positives = ", ".join(case.false_positives) if case.false_positives else "none"
            rank = case.correct_rank if case.correct_rank is not None else "miss"
            lines.append(f"  {index}. {case.test_case_id} [{case.failure_type}]")
            lines.append(f"     Query: {case.query}")
            lines.append(f"     Correct rank: {rank}")
            lines.append(f"     False positives: {false_positives}")
    else:
        lines.append("  none")

    if report.recommendations:
        lines.extend(["", "Recommendations"])
        for item in sorted(report.recommendations, key=lambda rec: rec.priority):
            lines.append(f"  {item.priority}. {item.action}")

    return "\n".join(lines) + "\n"


def render_compare_terminal(report: ComparisonReport) -> str:
    lines = ["RAGProbe Compare Report", "=" * 23, "", "Metrics"]
    for delta in report.deltas:
        lines.append(
            f"  {delta.metric}: {delta.before:.3f} -> {delta.after:.3f} "
            f"({delta.delta:+.3f})"
        )
    lines.extend(["", "Case Changes"])
    lines.append(
        "  Improved: " + (", ".join(report.improved_cases) if report.improved_cases else "none")
    )
    lines.append(
        "  Regressed: " + (", ".join(report.regressed_cases) if report.regressed_cases else "none")
    )
    return "\n".join(lines) + "\n"
