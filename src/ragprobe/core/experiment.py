"""Multi-retriever experiment orchestration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragprobe.core.analyzer import DiagnosticAnalyzer
from ragprobe.core.compare import compare_reports
from ragprobe.core.matching import apply_content_fallback
from ragprobe.core.models import ComparisonReport, DiagnosticReport, RetrievalResult, TestSet
from ragprobe.core.runner import (
    load_endpoint_config,
    run_endpoint,
    run_retriever_command,
    run_retriever_script,
)
from ragprobe.io.jsonl import load_json, load_testset, save_json


@dataclass
class ExperimentRetrieverSpec:
    name: str
    retriever: str | None = None
    retriever_cmd: str | None = None
    endpoint: str | None = None
    endpoint_config: str | None = None
    top_k: int | None = None
    timeout: float | None = None
    batch_size: int | None = None
    content_match_threshold: float | None = None


@dataclass
class ExperimentRetrieverRun:
    name: str
    report: DiagnosticReport
    results: list[RetrievalResult] = field(default_factory=list)
    results_path: str = ""
    report_path: str = ""


@dataclass
class ExperimentComparison:
    baseline: str
    candidate: str
    comparison: ComparisonReport
    failure_pattern_deltas: dict[str, int] = field(default_factory=dict)
    confusion_deltas: dict[str, float] = field(default_factory=dict)


@dataclass
class ExperimentReport:
    name: str
    baseline: str
    runs: list[ExperimentRetrieverRun]
    comparisons: list[ExperimentComparison] = field(default_factory=list)
    best_by_metric: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def run_experiment(
    config: dict[str, Any] | str | Path,
    *,
    output_dir: str | Path | None = None,
) -> ExperimentReport:
    """Run several named retrievers against one testset and compare them."""
    payload, base_dir = _load_config(config)
    name = str(payload.get("name", "ragprobe-experiment"))
    testset = load_testset(_resolve_path(payload["testset"], base_dir))
    specs = _parse_retrievers(payload)
    baseline = str(payload.get("baseline") or specs[0].name)
    if baseline not in {spec.name for spec in specs}:
        raise ValueError(f"baseline retriever is not defined: {baseline}")

    if output_dir is not None:
        target_dir = Path(output_dir)
    else:
        target_dir = Path(payload.get("output_dir", "ragprobe_experiment"))
        if not target_dir.is_absolute():
            target_dir = base_dir / target_dir
    if not target_dir.is_absolute():
        target_dir = Path.cwd() / target_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    default_top_k = int(payload.get("top_k", 10))
    default_timeout = float(payload.get("timeout", 30.0))
    default_batch_size = int(payload.get("batch_size", 1))
    default_match_threshold = float(payload.get("content_match_threshold", 0.9))

    runs: list[ExperimentRetrieverRun] = []
    reports_by_name: dict[str, DiagnosticReport] = {}
    for spec in specs:
        results = _run_spec(
            testset,
            spec,
            base_dir=base_dir,
            default_top_k=default_top_k,
            default_timeout=default_timeout,
            default_batch_size=default_batch_size,
            default_match_threshold=default_match_threshold,
        )
        match_threshold = spec.content_match_threshold or default_match_threshold
        matched = apply_content_fallback(testset, results, threshold=match_threshold)
        report = DiagnosticAnalyzer().analyze(testset, matched)
        report.metadata["retriever_name"] = spec.name

        slug = _slug(spec.name)
        results_path = target_dir / f"{slug}.results.json"
        report_path = target_dir / f"{slug}.report.json"
        save_json({"results": matched}, results_path)
        save_json(report, report_path)

        run = ExperimentRetrieverRun(
            name=spec.name,
            report=report,
            results=matched,
            results_path=str(results_path),
            report_path=str(report_path),
        )
        runs.append(run)
        reports_by_name[spec.name] = report

    baseline_report = reports_by_name[baseline]
    comparisons = [
        ExperimentComparison(
            baseline=baseline,
            candidate=run.name,
            comparison=compare_reports(baseline_report, run.report),
            failure_pattern_deltas=_failure_pattern_deltas(baseline_report, run.report),
            confusion_deltas=_distribution_deltas(
                baseline_report.confusion_distribution,
                run.report.confusion_distribution,
            ),
        )
        for run in runs
        if run.name != baseline
    ]

    report = ExperimentReport(
        name=name,
        baseline=baseline,
        runs=runs,
        comparisons=comparisons,
        best_by_metric=_best_by_metric(runs),
        metadata={
            "testset_name": testset.name,
            "total_cases": len(testset.cases),
            "output_dir": str(target_dir),
            "result_files": {run.name: run.results_path for run in runs},
            "report_files": {run.name: run.report_path for run in runs},
        },
    )
    save_json(report, target_dir / "experiment_report.json")
    return report


def _load_config(config: dict[str, Any] | str | Path) -> tuple[dict[str, Any], Path]:
    if isinstance(config, (str, Path)):
        path = Path(config)
        return load_json(path), path.resolve().parent
    return config, Path.cwd()


def _parse_retrievers(payload: dict[str, Any]) -> list[ExperimentRetrieverSpec]:
    raw_specs = payload.get("retrievers", [])
    if not raw_specs:
        raise ValueError("experiment config requires at least one retriever")

    specs = []
    names = set()
    for raw in raw_specs:
        spec = ExperimentRetrieverSpec(
            name=str(raw["name"]),
            retriever=raw.get("retriever"),
            retriever_cmd=raw.get("retriever_cmd"),
            endpoint=raw.get("endpoint"),
            endpoint_config=raw.get("endpoint_config"),
            top_k=raw.get("top_k"),
            timeout=raw.get("timeout"),
            batch_size=raw.get("batch_size"),
            content_match_threshold=raw.get("content_match_threshold"),
        )
        if spec.name in names:
            raise ValueError(f"duplicate retriever name: {spec.name}")
        names.add(spec.name)
        source_count = sum(
            item is not None for item in [spec.retriever, spec.retriever_cmd, spec.endpoint]
        )
        if source_count != 1:
            raise ValueError(
                f"retriever '{spec.name}' must define exactly one of retriever, "
                "retriever_cmd, or endpoint"
            )
        specs.append(spec)
    return specs


def _run_spec(
    testset: TestSet,
    spec: ExperimentRetrieverSpec,
    *,
    base_dir: Path,
    default_top_k: int,
    default_timeout: float,
    default_batch_size: int,
    default_match_threshold: float,
) -> list[RetrievalResult]:
    top_k = int(spec.top_k or default_top_k)
    timeout = float(spec.timeout or default_timeout)
    batch_size = int(spec.batch_size or default_batch_size)
    match_threshold = float(spec.content_match_threshold or default_match_threshold)

    if spec.retriever:
        return run_retriever_script(
            testset,
            _resolve_path(spec.retriever, base_dir),
            top_k=top_k,
            content_fallback_threshold=match_threshold,
        )
    if spec.retriever_cmd:
        return run_retriever_command(
            testset,
            spec.retriever_cmd,
            top_k=top_k,
            timeout=timeout,
            content_fallback_threshold=match_threshold,
        )

    config = load_endpoint_config(
        _resolve_path(spec.endpoint_config, base_dir) if spec.endpoint_config else None
    )
    return run_endpoint(
        testset,
        spec.endpoint or "",
        top_k=top_k,
        timeout=config.timeout if spec.endpoint_config else timeout,
        headers=config.headers,
        batch_size=config.batch_size if spec.endpoint_config else batch_size,
        content_fallback_threshold=match_threshold,
    )


def _resolve_path(path: str | Path | None, base_dir: Path) -> Path:
    if path is None:
        raise ValueError("path is required")
    resolved = Path(path)
    return resolved if resolved.is_absolute() else base_dir / resolved


def _failure_pattern_deltas(
    baseline: DiagnosticReport,
    candidate: DiagnosticReport,
) -> dict[str, int]:
    before = _failure_pattern_counts(baseline)
    after = _failure_pattern_counts(candidate)
    keys = sorted(set(before) | set(after))
    return {key: after.get(key, 0) - before.get(key, 0) for key in keys}


def _failure_pattern_counts(report: DiagnosticReport) -> dict[str, int]:
    return {
        pattern.pattern_type: len(pattern.affected_cases)
        for pattern in report.failure_patterns
    }


def _distribution_deltas(
    baseline: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, float]:
    keys = sorted(set(baseline) | set(candidate))
    return {key: candidate.get(key, 0.0) - baseline.get(key, 0.0) for key in keys}


def _best_by_metric(runs: list[ExperimentRetrieverRun]) -> dict[str, str]:
    if not runs:
        return {}
    return {
        "hit_rate": max(
            runs,
            key=lambda item: (
                item.report.hit_rate,
                item.report.mrr,
                -item.report.fpr,
                -len(item.report.failure_cases),
            ),
        ).name,
        "mrr": max(
            runs,
            key=lambda item: (
                item.report.mrr,
                item.report.hit_rate,
                -item.report.fpr,
                -len(item.report.failure_cases),
            ),
        ).name,
        "fpr": min(
            runs,
            key=lambda item: (
                item.report.fpr,
                -item.report.hit_rate,
                -item.report.mrr,
                len(item.report.failure_cases),
            ),
        ).name,
    }


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "retriever"
