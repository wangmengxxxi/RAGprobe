"""LLM judge audit for existing testsets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragprobe.core.generator import DocumentChunk
from ragprobe.core.llm_generation import LLMGenerationConfig, LLMJudgeClient, LLMJudgeDecision
from ragprobe.core.models import TestCase, TestSet
from ragprobe.core.schema import SCHEMA_AUDIT_REPORT, schema_metadata
from ragprobe.io.jsonl import load_testset, save_json

AUDIT_PROMPT_VERSION = "ragprobe-v0.9-testset-audit-v1"


@dataclass
class AuditChunkFinding:
    chunk_id: str
    role: str
    answerable: bool
    confidence: float | None = None
    risk: str = "low"
    reason: str = ""


@dataclass
class AuditCaseFinding:
    case_id: str
    query: str
    status: str
    expected_findings: list[AuditChunkFinding] = field(default_factory=list)
    hard_negative_findings: list[AuditChunkFinding] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    recommended_action: str = "keep"


@dataclass
class AuditReport:
    testset_name: str
    total_cases: int
    audited_cases: int
    findings: list[AuditCaseFinding] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def audit_testset(
    testset: TestSet | str | Path,
    *,
    judge_client: LLMJudgeClient,
    config: LLMGenerationConfig,
    sample_size: int | None = None,
    case_ids: list[str] | None = None,
    cache_dir: str | Path | None = ".ragprobe_cache",
    use_cache: bool = True,
) -> AuditReport:
    loaded_testset = load_testset(testset) if isinstance(testset, (str, Path)) else testset
    selected_cases = _select_cases(loaded_testset.cases, sample_size=sample_size, case_ids=case_ids)
    chunks_by_id = _chunks_from_testset(loaded_testset)
    cache_path = Path(cache_dir) if cache_dir and use_cache else None

    findings = [
        _audit_case(
            case,
            chunks_by_id=chunks_by_id,
            judge_client=judge_client,
            config=config,
            cache_dir=cache_path,
        )
        for case in selected_cases
    ]
    return AuditReport(
        testset_name=loaded_testset.name,
        total_cases=len(loaded_testset.cases),
        audited_cases=len(findings),
        findings=findings,
        summary=_summarize_findings(findings),
        metadata={
            **schema_metadata(SCHEMA_AUDIT_REPORT),
            "source": "ragprobe-v0.9-testset-audit",
            "llm_provider": config.provider,
            "llm_model": config.model,
            "llm_base_url": config.base_url.split("?")[0],
            "llm_api_key_env": config.api_key_env,
            "audit_prompt_version": AUDIT_PROMPT_VERSION,
            "sample_size": sample_size,
            "case_ids": case_ids or [],
        },
    )


def save_audit_report(report: AuditReport, path: str | Path) -> None:
    save_json(report, path)


def _audit_case(
    case: TestCase,
    *,
    chunks_by_id: dict[str, DocumentChunk],
    judge_client: LLMJudgeClient,
    config: LLMGenerationConfig,
    cache_dir: Path | None,
) -> AuditCaseFinding:
    warnings: list[str] = []
    expected_findings = []
    hard_negative_findings = []

    for chunk_id in case.expected_chunks:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            warnings.append("missing_expected_chunk_content")
            expected_findings.append(
                AuditChunkFinding(
                    chunk_id=chunk_id,
                    role="expected_chunk",
                    answerable=False,
                    risk="high",
                    reason="Expected chunk content was not available in testset metadata.",
                )
            )
            continue
        decision = _judge_with_cache(
            judge_client,
            query=case.query,
            chunk=chunk,
            role="expected_chunk",
            config=config,
            cache_dir=cache_dir,
        )
        if not decision.answerable:
            warnings.append("expected_chunk_not_answerable")
        expected_findings.append(
            AuditChunkFinding(
                chunk_id=chunk_id,
                role="expected_chunk",
                answerable=decision.answerable,
                confidence=decision.confidence,
                risk="low" if decision.answerable else "high",
                reason=decision.reason,
            )
        )

    for hard_negative in case.hard_negatives:
        chunk = chunks_by_id.get(hard_negative.chunk_id)
        if chunk is None:
            warnings.append("missing_hard_negative_chunk_content")
            hard_negative_findings.append(
                AuditChunkFinding(
                    chunk_id=hard_negative.chunk_id,
                    role="hard_negative",
                    answerable=False,
                    risk="medium",
                    reason="Hard negative chunk content was not available in testset metadata.",
                )
            )
            continue
        decision = _judge_with_cache(
            judge_client,
            query=case.query,
            chunk=chunk,
            role="hard_negative",
            config=config,
            cache_dir=cache_dir,
        )
        if decision.answerable:
            warnings.append("hard_negative_answerable")
        hard_negative_findings.append(
            AuditChunkFinding(
                chunk_id=hard_negative.chunk_id,
                role="hard_negative",
                answerable=decision.answerable,
                confidence=decision.confidence,
                risk="high" if decision.answerable else "low",
                reason=decision.reason,
            )
        )

    warnings = sorted(set(warnings))
    status = _case_status(warnings)
    return AuditCaseFinding(
        case_id=case.id,
        query=case.query,
        status=status,
        expected_findings=expected_findings,
        hard_negative_findings=hard_negative_findings,
        warnings=warnings,
        recommended_action=_recommended_action(warnings),
    )


def _judge_with_cache(
    judge_client: LLMJudgeClient,
    *,
    query: str,
    chunk: DocumentChunk,
    role: str,
    config: LLMGenerationConfig,
    cache_dir: Path | None,
) -> LLMJudgeDecision:
    if cache_dir is not None:
        cached = _read_cached_decision(cache_dir, query, chunk, role, config)
        if cached is not None:
            return cached
    decision = judge_client.judge_answerability(
        query=query,
        chunk=chunk,
        role=role,
        config=config,
    )
    if cache_dir is not None:
        _write_cached_decision(cache_dir, query, chunk, role, config, decision)
    return decision


def _select_cases(
    cases: list[TestCase],
    *,
    sample_size: int | None,
    case_ids: list[str] | None,
) -> list[TestCase]:
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in cases if case.id in wanted]
        missing = sorted(wanted - {case.id for case in selected})
        if missing:
            raise ValueError(f"case ids not found: {', '.join(missing)}")
    else:
        selected = list(cases)
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be greater than 0")
        selected = selected[:sample_size]
    return selected


def _chunks_from_testset(testset: TestSet) -> dict[str, DocumentChunk]:
    raw_chunks = testset.metadata.get("chunks", {})
    if not isinstance(raw_chunks, dict):
        return {}
    chunks = {}
    for chunk_id, value in raw_chunks.items():
        if isinstance(value, dict):
            content = value.get("content", value.get("text", ""))
            metadata = dict(value.get("metadata", {}))
            source_document = str(value.get("source_document", value.get("source", "")))
        else:
            content = str(value)
            metadata = {}
            source_document = ""
        chunks[str(chunk_id)] = DocumentChunk(
            chunk_id=str(chunk_id),
            content=str(content),
            metadata=metadata,
            source_document=source_document,
        )
    return chunks


def _case_status(warnings: list[str]) -> str:
    if "expected_chunk_not_answerable" in warnings or "missing_expected_chunk_content" in warnings:
        return "failed"
    if any(
        warning in warnings
        for warning in [
            "hard_negative_answerable",
            "missing_hard_negative_chunk_content",
        ]
    ):
        return "suspicious"
    return "passed"


def _recommended_action(warnings: list[str]) -> str:
    if "expected_chunk_not_answerable" in warnings or "missing_expected_chunk_content" in warnings:
        return "reject_case"
    if "hard_negative_answerable" in warnings:
        return "remove_hard_negative_or_review_query"
    if "missing_hard_negative_chunk_content" in warnings:
        return "review_metadata_chunks"
    return "keep"


def _summarize_findings(findings: list[AuditCaseFinding]) -> dict[str, Any]:
    statuses = _count_items([finding.status for finding in findings])
    warnings = _count_items(
        [warning for finding in findings for warning in finding.warnings]
    )
    return {
        "status_counts": statuses,
        "warning_counts": warnings,
        "passed": statuses.get("passed", 0),
        "suspicious": statuses.get("suspicious", 0),
        "failed": statuses.get("failed", 0),
        "requires_review": statuses.get("suspicious", 0) + statuses.get("failed", 0),
    }


def _count_items(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items()))


def _read_cached_decision(
    cache_dir: Path,
    query: str,
    chunk: DocumentChunk,
    role: str,
    config: LLMGenerationConfig,
) -> LLMJudgeDecision | None:
    path = _cache_file(cache_dir, query, chunk, role, config)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return LLMJudgeDecision(
        answerable=bool(payload.get("answerable", False)),
        confidence=_optional_float(payload.get("confidence")),
        reason=str(payload.get("reason", "")),
    )


def _write_cached_decision(
    cache_dir: Path,
    query: str,
    chunk: DocumentChunk,
    role: str,
    config: LLMGenerationConfig,
    decision: LLMJudgeDecision,
) -> None:
    path = _cache_file(cache_dir, query, chunk, role, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "answerable": decision.answerable,
                "confidence": decision.confidence,
                "reason": decision.reason,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _cache_file(
    cache_dir: Path,
    query: str,
    chunk: DocumentChunk,
    role: str,
    config: LLMGenerationConfig,
) -> Path:
    payload = {
        "prompt_version": AUDIT_PROMPT_VERSION,
        "provider": config.provider,
        "model": config.model,
        "query": query,
        "chunk_id": chunk.chunk_id,
        "chunk_content": chunk.content,
        "role": role,
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return cache_dir / "audit" / f"{digest}.json"


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
