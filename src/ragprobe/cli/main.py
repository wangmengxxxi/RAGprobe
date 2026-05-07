"""RAGProbe command line entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ragprobe.core.analyzer import DiagnosticAnalyzer
from ragprobe.core.checks import check_thresholds
from ragprobe.core.compare import compare_reports
from ragprobe.core.generator import (
    add_case,
    generate_testset_from_chunks,
    load_chunks,
    render_quality_report,
    sample_testset,
)
from ragprobe.core.matching import apply_content_fallback
from ragprobe.core.runner import (
    load_endpoint_config,
    run_endpoint,
    run_retriever_command,
    run_retriever_script,
)
from ragprobe.core.validation import validate_results_report, validate_testset
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

    run = subparsers.add_parser("run", help="Run a retriever against a testset.")
    run.add_argument("--testset", required=True)
    source = run.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--retriever",
        required=False,
        help="Python file exposing retrieve(query, top_k).",
    )
    source.add_argument(
        "--retriever-cmd",
        required=False,
        help="JSONL subprocess retriever command.",
    )
    source.add_argument("--endpoint", required=False, help="HTTP endpoint accepting POST JSON.")
    run.add_argument("--output", required=True)
    run.add_argument("--top-k", type=int, default=10)
    run.add_argument("--timeout", type=float, default=30.0)
    run.add_argument("--endpoint-config", required=False)
    run.add_argument("--batch-size", type=int, default=1)
    run.add_argument("--content-match-threshold", type=float, default=0.9)

    generate = subparsers.add_parser("generate", help="Generate a lightweight testset from chunks.")
    generate.add_argument("--chunks", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--num-cases", type=int, required=False)
    generate.add_argument("--hard-negative-top-k", type=int, default=1)
    generate.add_argument("--name", default="generated-testset")
    generate.add_argument("--mode", choices=["standard"], default="standard")
    generate.add_argument("--hn-strategy", choices=["lexical", "hybrid"], default="hybrid")
    generate.add_argument("--quality-report", required=False)

    add_case_cmd = subparsers.add_parser("add-case", help="Append a manual bad case to a testset.")
    add_case_cmd.add_argument("--testset", required=True)
    add_case_cmd.add_argument("--output", required=True)
    add_case_cmd.add_argument("--query", required=True)
    add_case_cmd.add_argument("--expected-chunk", required=True)
    add_case_cmd.add_argument("--hard-negative", action="append", default=[])
    add_case_cmd.add_argument("--confusion-type", default="manual")
    add_case_cmd.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="medium")
    add_case_cmd.add_argument("--source-document", default="")
    add_case_cmd.add_argument("--tag", action="append", default=[])
    add_case_cmd.add_argument("--case-id", required=False)

    sample = subparsers.add_parser("sample", help="Export testset cases for human review.")
    sample.add_argument("--testset", required=True)
    sample.add_argument("--output", required=True)
    sample.add_argument("--limit", type=int, default=10)

    validate = subparsers.add_parser(
        "validate",
        help="Validate testset and optional results schema.",
    )
    validate.add_argument("--testset", required=True)
    validate.add_argument("--results", required=False)

    export_queries = subparsers.add_parser(
        "export-queries",
        help="Export testset queries as JSONL.",
    )
    export_queries.add_argument("--testset", required=True)
    export_queries.add_argument("--output", required=True)
    export_queries.add_argument("--top-k", type=int, default=10)

    import_results = subparsers.add_parser("import-results", help="Import JSONL retrieval outputs.")
    import_results.add_argument("--queries", required=True)
    import_results.add_argument("--results", required=True)
    import_results.add_argument("--output", required=True)

    diagnose = subparsers.add_parser("diagnose", help="Diagnose offline retrieval results.")
    diagnose.add_argument("--testset", required=True)
    diagnose.add_argument("--results", required=True)
    diagnose.add_argument("--output", required=False)
    diagnose.add_argument("--format", choices=["terminal", "markdown", "json"], default="terminal")
    diagnose.add_argument("--content-match-threshold", type=float, default=0.9)

    compare = subparsers.add_parser("compare", help="Compare two retrieval result sets.")
    compare.add_argument("--testset", required=True)
    compare.add_argument("--before", required=True)
    compare.add_argument("--after", required=True)
    compare.add_argument("--output", required=False)
    compare.add_argument("--format", choices=["terminal", "markdown", "json"], default="terminal")
    compare.add_argument("--content-match-threshold", type=float, default=0.9)

    check = subparsers.add_parser("check", help="Fail when diagnostic metrics cross thresholds.")
    check.add_argument("--report", required=False)
    check.add_argument("--results", required=False)
    check.add_argument("--testset", required=False)
    check.add_argument("--min-hit-rate", type=float, required=False)
    check.add_argument("--min-mrr", type=float, required=False)
    check.add_argument("--max-fpr", type=float, required=False)
    check.add_argument("--max-low-confidence-match-rate", type=float, required=False)
    check.add_argument("--content-match-threshold", type=float, default=0.9)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "demo":
        return _run_demo(args)
    if args.command == "run":
        return _run_run(args)
    if args.command == "generate":
        return _run_generate(args)
    if args.command == "add-case":
        return _run_add_case(args)
    if args.command == "sample":
        return _run_sample(args)
    if args.command == "validate":
        return _run_validate(args)
    if args.command == "export-queries":
        return _run_export_queries(args)
    if args.command == "import-results":
        return _run_import_results(args)
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


def _run_run(args: argparse.Namespace) -> int:
    testset = load_testset(args.testset)
    if args.retriever:
        results = run_retriever_script(
            testset,
            args.retriever,
            top_k=args.top_k,
            content_fallback_threshold=args.content_match_threshold,
        )
    elif args.retriever_cmd:
        results = run_retriever_command(
            testset,
            args.retriever_cmd,
            top_k=args.top_k,
            timeout=args.timeout,
            content_fallback_threshold=args.content_match_threshold,
        )
    else:
        config = load_endpoint_config(args.endpoint_config)
        results = run_endpoint(
            testset,
            args.endpoint,
            top_k=args.top_k,
            timeout=config.timeout if args.endpoint_config else args.timeout,
            headers=config.headers,
            batch_size=config.batch_size if args.endpoint_config else args.batch_size,
            content_fallback_threshold=args.content_match_threshold,
        )
    save_json({"results": results}, args.output)
    return 0


def _run_generate(args: argparse.Namespace) -> int:
    chunks = load_chunks(args.chunks)
    testset = generate_testset_from_chunks(
        chunks,
        num_cases=args.num_cases,
        hard_negative_top_k=args.hard_negative_top_k,
        name=args.name,
        mode=args.mode,
        hn_strategy=args.hn_strategy,
    )
    save_json(testset, args.output)
    if args.quality_report:
        _emit_text(render_quality_report(testset), args.quality_report)
    return 0


def _run_add_case(args: argparse.Namespace) -> int:
    testset = load_testset(args.testset)
    updated = add_case(
        testset,
        query=args.query,
        expected_chunk=args.expected_chunk,
        hard_negative_ids=args.hard_negative,
        confusion_type=args.confusion_type,
        difficulty=args.difficulty,
        source_document=args.source_document,
        tags=args.tag,
        case_id=args.case_id,
    )
    save_json(updated, args.output)
    return 0


def _run_sample(args: argparse.Namespace) -> int:
    testset = load_testset(args.testset)
    rows = sample_testset(testset, limit=args.limit)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


def _run_export_queries(args: argparse.Namespace) -> int:
    testset = load_testset(args.testset)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as file:
        for case in testset.cases:
            row = {"id": case.id, "query": case.query, "top_k": args.top_k}
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return 0


def _run_import_results(args: argparse.Namespace) -> int:
    queries = _load_jsonl(args.queries)
    raw_results = _load_jsonl(args.results)
    if len(queries) != len(raw_results):
        raise SystemExit("queries and results JSONL files must have the same number of lines")

    results = []
    for query_row, result_row in zip(queries, raw_results, strict=True):
        if isinstance(result_row, list):
            retrieved = result_row
            test_case_id = query_row["id"]
            query = query_row["query"]
        else:
            retrieved = result_row.get("retrieved", result_row.get("results", result_row))
            test_case_id = result_row.get("test_case_id", query_row["id"])
            query = result_row.get("query", query_row["query"])
        if not isinstance(retrieved, list):
            raise SystemExit("each results JSONL line must be a list or contain retrieved/results")
        results.append(
            {
                "test_case_id": test_case_id,
                "query": query,
                "retrieved": retrieved,
            }
        )
    save_json({"results": results}, args.output)
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    testset = load_testset(args.testset)
    testset_report = validate_testset(testset)
    reports = [("testset", testset_report)]
    if args.results:
        results = load_results(args.results)
        results_report = validate_results_report(testset, results)
        reports.append(("results", results_report))

    ok = True
    for label, report in reports:
        if report.valid:
            print(f"{label}: valid")
        else:
            ok = False
            print(f"{label}: invalid")
        for warning in report.warnings:
            print(f"  warning: {warning}")
        for error in report.errors:
            print(f"  error: {error}")
    return 0 if ok else 1


def _run_diagnose(args: argparse.Namespace) -> int:
    report = _analyze(
        args.testset,
        args.results,
        content_match_threshold=args.content_match_threshold,
    )
    _emit_report(report, args.format, args.output)
    return 0


def _run_compare(args: argparse.Namespace) -> int:
    before = _analyze(
        args.testset,
        args.before,
        content_match_threshold=args.content_match_threshold,
    )
    after = _analyze(
        args.testset,
        args.after,
        content_match_threshold=args.content_match_threshold,
    )
    report = compare_reports(before, after)
    if args.format == "json":
        if args.output:
            save_json(report, args.output)
        else:
            print_json(report)
        return 0

    text = (
        render_compare_markdown(report)
        if args.format == "markdown"
        else render_compare_terminal(report)
    )
    _emit_text(text, args.output)
    return 0


def _run_check(args: argparse.Namespace) -> int:
    if args.report:
        report = load_report(args.report)
    else:
        if not args.testset or not args.results:
            raise SystemExit("check requires --report or both --testset and --results")
        report = _analyze(
            args.testset,
            args.results,
            content_match_threshold=args.content_match_threshold,
        )

    result = check_thresholds(
        report,
        min_hit_rate=args.min_hit_rate,
        min_mrr=args.min_mrr,
        max_fpr=args.max_fpr,
        max_low_confidence_match_rate=args.max_low_confidence_match_rate,
    )
    for message in result.messages:
        print(message)
    return 0 if result.passed else 1


def _analyze(
    testset_path: str | Path,
    results_path: str | Path,
    content_match_threshold: float = 0.9,
):
    testset = load_testset(testset_path)
    results = load_results(results_path)
    results = apply_content_fallback(testset, results, threshold=content_match_threshold)
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
    from ragprobe.io.jsonl import to_jsonable

    print(json.dumps(to_jsonable(data), ensure_ascii=False, indent=2))


def _load_jsonl(path: str | Path) -> list:
    rows = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
