from __future__ import annotations

from pathlib import Path

from ragprobe.core.analyzer import DiagnosticAnalyzer
from ragprobe.core.checks import check_thresholds
from ragprobe.core.matching import apply_content_fallback
from ragprobe.io.jsonl import load_results, load_testset
from ragprobe.reports.markdown import render_markdown
from ragprobe.reports.terminal import render_terminal

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_v04_analyzer_reports_failure_patterns_and_evidence() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")
    results = load_results(FIXTURES / "results_v1.json")

    report = DiagnosticAnalyzer().analyze(testset, results)

    pattern_types = {pattern.pattern_type for pattern in report.failure_patterns}
    issue_types = {issue.issue_type for issue in report.system_issues}

    assert "hard_negative_without_expected" in pattern_types
    assert "hard_negative_ranked_above_correct" in pattern_types
    assert "ranking_weakness" in pattern_types
    assert "hard_negative_confusion" in issue_types
    assert "low_recall_coverage" in issue_types
    assert report.recommendations
    assert all(item.evidence for item in report.recommendations)
    assert all(item.scope == "reference_only" for item in report.recommendations)


def test_v04_reports_render_failure_patterns_and_recommendation_evidence() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")
    results = load_results(FIXTURES / "results_v1.json")
    report = DiagnosticAnalyzer().analyze(testset, results)

    terminal = render_terminal(report)
    markdown = render_markdown(report)

    assert "Failure Patterns" in terminal
    assert "Evidence:" in terminal
    assert "Scope: reference_only" in terminal
    assert "## Failure Patterns" in markdown
    assert "Scope: reference_only" in markdown


def test_v04_check_thresholds_include_mrr_and_low_confidence_matching() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")
    results = load_results(FIXTURES / "results_without_ids.json")
    matched = apply_content_fallback(testset, results)
    report = DiagnosticAnalyzer().analyze(testset, matched)

    failed = check_thresholds(
        report,
        min_hit_rate=0.9,
        min_mrr=1.1,
        max_fpr=0.1,
        max_low_confidence_match_rate=0.2,
    )

    assert not failed.passed
    assert any(message.startswith("mrr ") for message in failed.messages)
    assert any(message.startswith("low_confidence_match_rate ") for message in failed.messages)
