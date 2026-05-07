from pathlib import Path

from ragprobe.core.analyzer import DiagnosticAnalyzer
from ragprobe.core.checks import check_thresholds
from ragprobe.core.compare import compare_reports
from ragprobe.io.jsonl import load_results, load_testset


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_analyzer_computes_v01_metrics() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")
    results = load_results(FIXTURES / "results_v1.json")

    report = DiagnosticAnalyzer().analyze(testset, results)

    assert report.hit_rate == 0.5
    assert report.mrr == 0.25
    assert report.fpr == 1.0
    assert report.precision_at_k[5] == 0.1
    assert report.confusion_distribution == {
        "event_confusion": 0.5,
        "subject_confusion": 0.5,
    }
    assert [case.test_case_id for case in report.failure_cases] == ["case_2", "case_1"]


def test_compare_reports_tracks_improved_cases() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")
    before = DiagnosticAnalyzer().analyze(testset, load_results(FIXTURES / "results_v1.json"))
    after = DiagnosticAnalyzer().analyze(testset, load_results(FIXTURES / "results_v2.json"))

    compare = compare_reports(before, after)

    assert compare.deltas[0].metric == "hit_rate"
    assert compare.deltas[0].delta == 0.5
    assert compare.deltas[2].metric == "fpr"
    assert compare.deltas[2].delta == -1.0
    assert compare.improved_cases == ["case_1", "case_2"]
    assert compare.regressed_cases == []


def test_threshold_checks_return_pass_or_fail() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")
    report = DiagnosticAnalyzer().analyze(testset, load_results(FIXTURES / "results_v1.json"))

    failed = check_thresholds(report, min_hit_rate=0.7, max_fpr=0.3)
    passed = check_thresholds(report, min_hit_rate=0.4, max_fpr=1.0)

    assert not failed.passed
    assert failed.messages == [
        "hit_rate 0.500 is below minimum 0.700",
        "fpr 1.000 is above maximum 0.300",
    ]
    assert passed.passed
