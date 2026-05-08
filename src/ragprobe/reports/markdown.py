"""Markdown report renderers."""

from __future__ import annotations

from ragprobe.core.experiment import ExperimentReport
from ragprobe.core.models import ComparisonReport, DiagnosticReport


def render_markdown(report: DiagnosticReport) -> str:
    lines = [
        "# RAGProbe Diagnostic Report",
        "",
        "## Retrieval Quality",
        "",
        f"- Hit Rate: {report.hit_rate:.3f}",
        f"- MRR: {report.mrr:.3f}",
    ]
    for k, value in sorted(report.precision_at_k.items()):
        lines.append(f"- Precision@{k}: {value:.3f}")
    lines.extend(
        [
            "",
            "## Hard Negative Resistance",
            "",
            f"- FPR: {report.fpr:.3f}",
            "",
            "## Confusion Distribution",
            "",
        ]
    )
    if report.confusion_distribution:
        for key, value in report.confusion_distribution.items():
            lines.append(f"- {key}: {value:.1%}")
    else:
        lines.append("- none")

    if report.metric_signals:
        lines.extend(["", "## Metric Signals", ""])
        for signal in report.metric_signals:
            lines.extend(
                [
                    f"- **[{signal.severity.upper()}] {signal.name}**",
                    f"  - {signal.summary}",
                    f"  - Evidence: `{signal.evidence}`",
                ]
            )

    if report.failure_patterns:
        lines.extend(["", "## Failure Patterns", ""])
        for pattern in report.failure_patterns:
            lines.extend(
                [
                    f"- **[{pattern.severity.upper()}] {pattern.pattern_type}**",
                    f"  - Affected: {pattern.affected_percentage:.1%}",
                    f"  - Evidence: {pattern.evidence}",
                ]
            )

    lines.extend(["", "## Worst Cases", ""])
    if report.failure_cases:
        for index, case in enumerate(report.failure_cases[:5], start=1):
            false_positives = ", ".join(case.false_positives) if case.false_positives else "none"
            rank = case.correct_rank if case.correct_rank is not None else "miss"
            lines.extend(
                [
                    f"{index}. `{case.test_case_id}` ({case.failure_type})",
                    f"   - Query: {case.query}",
                    f"   - Correct rank: {rank}",
                    f"   - False positives: {false_positives}",
                ]
            )
    else:
        lines.append("- none")

    if report.system_issues:
        lines.extend(["", "## System Issues", ""])
        for issue in report.system_issues:
            lines.append(f"- [{issue.severity.upper()}] {issue.issue_type}: {issue.evidence}")

    if report.recommendations:
        lines.extend(["", "## Recommendations", ""])
        for item in sorted(report.recommendations, key=lambda rec: rec.priority):
            lines.append(f"{item.priority}. {item.action}")
            if item.evidence:
                lines.append(f"   - Evidence: {item.evidence}")
            lines.append(f"   - Scope: {item.scope}")

    match_stats = report.metadata.get("match_stats", {})
    if match_stats and (match_stats.get("content_fallback", 0) or match_stats.get("unmatched", 0)):
        lines.extend(["", "## Matching Notes", ""])
        if match_stats.get("content_fallback", 0):
            lines.append(
                f"- {match_stats['content_fallback']} retrieved chunks were matched "
                "by content fallback."
            )
        if match_stats.get("unmatched", 0):
            lines.append(f"- {match_stats['unmatched']} retrieved chunks could not be matched.")

    return "\n".join(lines) + "\n"


def render_compare_markdown(report: ComparisonReport) -> str:
    lines = ["# RAGProbe Compare Report", "", "## Metrics", ""]
    for delta in report.deltas:
        lines.append(
            f"- {delta.metric}: {delta.before:.3f} -> {delta.after:.3f} "
            f"({delta.delta:+.3f})"
        )
    lines.extend(["", "## Case Changes", ""])
    lines.append(
        "- Improved cases: "
        + (", ".join(report.improved_cases) if report.improved_cases else "none")
    )
    lines.append(
        "- Regressed cases: "
        + (", ".join(report.regressed_cases) if report.regressed_cases else "none")
    )
    return "\n".join(lines) + "\n"


def render_experiment_markdown(report: ExperimentReport) -> str:
    lines = [
        "# RAGProbe Experiment Report",
        "",
        f"- Name: {report.name}",
        f"- Baseline: {report.baseline}",
        f"- Testset: {report.metadata.get('testset_name', '')}",
        f"- Total cases: {report.metadata.get('total_cases', 0)}",
        "",
        "## Retriever Metrics",
        "",
        "| Retriever | Hit Rate | MRR | FPR | Failures |",
        "|---|---:|---:|---:|---:|",
    ]
    for run in report.runs:
        diagnostic = run.report
        lines.append(
            f"| {run.name} | {diagnostic.hit_rate:.3f} | {diagnostic.mrr:.3f} | "
            f"{diagnostic.fpr:.3f} | {len(diagnostic.failure_cases)} |"
        )

    if report.best_by_metric:
        lines.extend(["", "## Best By Metric", ""])
        lines.append(f"- Hit Rate: {report.best_by_metric.get('hit_rate', '')}")
        lines.append(f"- MRR: {report.best_by_metric.get('mrr', '')}")
        lines.append(f"- FPR: {report.best_by_metric.get('fpr', '')}")

    if report.comparisons:
        lines.extend(["", "## Baseline Comparisons", ""])
        for item in report.comparisons:
            lines.append(f"### {item.candidate} vs {item.baseline}")
            lines.append("")
            for delta in item.comparison.deltas:
                lines.append(
                    f"- {delta.metric}: {delta.before:.3f} -> {delta.after:.3f} "
                    f"({delta.delta:+.3f})"
                )
            lines.append(
                "- Improved cases: "
                + (
                    ", ".join(item.comparison.improved_cases)
                    if item.comparison.improved_cases
                    else "none"
                )
            )
            lines.append(
                "- Regressed cases: "
                + (
                    ", ".join(item.comparison.regressed_cases)
                    if item.comparison.regressed_cases
                    else "none"
                )
            )
            if item.failure_pattern_deltas:
                lines.append("- Failure pattern deltas:")
                for key, value in item.failure_pattern_deltas.items():
                    lines.append(f"  - {key}: {value:+d}")
            if item.confusion_deltas:
                lines.append("- Confusion distribution deltas:")
                for key, value in item.confusion_deltas.items():
                    lines.append(f"  - {key}: {value:+.1%}")
            lines.append("")

    lines.extend(["## Output Files", ""])
    result_files = report.metadata.get("result_files", {})
    report_files = report.metadata.get("report_files", {})
    for run in report.runs:
        lines.append(f"- {run.name} results: `{result_files.get(run.name, run.results_path)}`")
        lines.append(f"- {run.name} report: `{report_files.get(run.name, run.report_path)}`")

    return "\n".join(lines).rstrip() + "\n"
