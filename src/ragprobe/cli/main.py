"""RAGProbe command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ragprobe.core.analyzer import DiagnosticAnalyzer
from ragprobe.core.checks import check_thresholds
from ragprobe.core.compare import compare_reports
from ragprobe.io.jsonl import load_report, load_results, load_testset, save_json
from ragprobe.reports.markdown import render_compare_markdown, render_markdown
from ragprobe.reports.terminal import render_compare_terminal, render_terminal


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = PACKAGE_ROOT / "data" / "contract"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragprobe")
    subparsers = parser.add_subparsers(dest="command")

    demo = subparsers.add_parser("demo", help="Run the bundled offline demo.")
    demo.add_argument("--output", required=False, help="Optional report output path.")
    demo.add_argument("--format", choices=["terminal", "markdown", "json"], default="terminal")

    diagnose = subparsers.add_parser("diagnose", help="Diagnose offline retrieval results.")
    diagnose.add_argument("--testset", required=True)
    diagnose.add_argument("--results", required=True)
    diagnose.add_argument("--output", required=False)
    diagnose.add_argument("--format", choices=["terminal", "markdown", "json"], default="terminal")

    compare = subparsers.add_parser("compare", help="Compare two retrieval result sets.")
    compare.add_argument("--testset", required=True)
    compare.add_argument("--before", required=True)
    compare.add_argument("--after", required=True)
    compare.add_argument("--output", required=False)
    compare.add_argument("--format", choices=["terminal", "markdown", "json"], default="terminal")

    check = subparsers.add_parser("check", help="Fail when diagnostic metrics cross thresholds.")
    check.add_argument("--report", required=False)
    check.add_argument("--results", required=False)
    check.add_argument("--testset", required=False)
    check.add_argument("--min-hit-rate", type=float, required=False)
    check.add_argument("--max-fpr", type=float, required=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "demo":
        return _run_demo(args)
    if args.command == "diagnose":
        return _run_diagnose(args)
    if args.command == "compare":
        return _run_compare(args)
    if args.command == "check":
        return _run_check(args)
    parser.error(f"unknown command: {args.command}")
    return 2


def _run_demo(args: argparse.Namespace) -> int:
    testset_path = DEMO_DIR / "testset.json"
    results_path = DEMO_DIR / "results_v1.json"
    report = _analyze(testset_path, results_path)
    _emit_report(report, args.format, args.output)
    return 0


def _run_diagnose(args: argparse.Namespace) -> int:
    report = _analyze(args.testset, args.results)
    _emit_report(report, args.format, args.output)
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    before = _analyze(args.testset, args.before)
    after = _analyze(args.testset, args.after)
    report = compare_reports(before, after)
    if args.format == "json":
        if args.output:
            save_json(report, args.output)
        else:
            print_json(report)
        return 0

    text = render_compare_markdown(report) if args.format == "markdown" else render_compare_terminal(report)
    _emit_text(text, args.output)
    return 0


def _run_check(args: argparse.Namespace) -> int:
    if args.report:
        report = load_report(args.report)
    else:
        if not args.testset or not args.results:
            raise SystemExit("check requires --report or both --testset and --results")
        report = _analyze(args.testset, args.results)

    result = check_thresholds(report, min_hit_rate=args.min_hit_rate, max_fpr=args.max_fpr)
    for message in result.messages:
        print(message)
    return 0 if result.passed else 1


def _analyze(testset_path: str | Path, results_path: str | Path):
    testset = load_testset(testset_path)
    results = load_results(results_path)
    return DiagnosticAnalyzer().analyze(testset, results)


def _emit_report(report, fmt: str, output: str | None) -> None:
    if fmt == "json":
        if output:
            save_json(report, output)
        else:
            print_json(report)
        return

    text = render_markdown(report) if fmt == "markdown" else render_terminal(report)
    _emit_text(text, output)


def _emit_text(text: str, output: str | None) -> None:
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)


def print_json(data) -> None:
    import json

    from ragprobe.io.jsonl import to_jsonable

    print(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
