"""Testset generation and maintenance helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragprobe.core.models import HardNegative, TestCase, TestSet

MIN_QUERY_LENGTH = 4
WEAK_HARD_NEGATIVE_THRESHOLD = 0.05
STRONG_HARD_NEGATIVE_THRESHOLD = 0.2


@dataclass
class DocumentChunk:
    chunk_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    source_document: str = ""


def load_chunks(path: str | Path) -> list[DocumentChunk]:
    """Load chunks from JSON, JSONL, or a mapping of chunk_id to content."""

    source = Path(path)
    if source.suffix.lower() == ".jsonl":
        rows = _load_jsonl(source)
    else:
        payload = _load_json(source)
        rows = _normalise_chunk_payload(payload)

    chunks = [_chunk_from_row(row, index) for index, row in enumerate(rows, start=1)]
    _validate_chunks(chunks)
    return chunks


def generate_testset_from_chunks(
    chunks: list[DocumentChunk],
    num_cases: int | None = None,
    hard_negative_top_k: int = 1,
    name: str = "generated-testset",
    mode: str = "standard",
    hn_strategy: str = "hybrid",
) -> TestSet:
    if mode != "standard":
        raise ValueError("v0.5 currently supports only mode='standard'")
    if hn_strategy not in {"lexical", "hybrid"}:
        raise ValueError("hn_strategy must be 'lexical' or 'hybrid'")
    if not chunks:
        raise ValueError("chunks must contain at least one item")
    if num_cases is not None and num_cases <= 0:
        raise ValueError("num_cases must be greater than 0")

    selected = chunks[:num_cases] if num_cases is not None else chunks
    cases = []
    for index, chunk in enumerate(selected, start=1):
        query = generate_query(chunk)
        candidates = mine_hard_negatives(
            chunk,
            chunks,
            top_k=hard_negative_top_k,
            strategy=hn_strategy,
        )
        hard_negatives = [
            HardNegative(
                chunk_id=candidate.chunk.chunk_id,
                confusion_type=candidate.confusion_type,
                similarity_to_correct=candidate.similarity,
                reason=candidate.reason,
            )
            for candidate in candidates
        ]
        quality = assess_case_quality(
            query=query,
            expected_chunk=chunk,
            hard_negatives=hard_negatives,
        )
        cases.append(
            TestCase(
                id=f"generated_case_{index:03d}",
                query=query,
                expected_chunks=[chunk.chunk_id],
                hard_negatives=hard_negatives,
                difficulty=label_difficulty(hard_negatives),
                source_document=chunk.source_document,
                metadata={
                    "created_from": "chunks",
                    "source_chunk_id": chunk.chunk_id,
                    "generator_mode": mode,
                    "hard_negative_strategy": hn_strategy,
                    "tags": list(chunk.metadata.get("tags", [])),
                    "quality": quality,
                },
            )
        )

    quality_summary = summarize_testset_quality(cases)
    return TestSet(
        cases=cases,
        name=name,
        metadata={
            "source": "ragprobe-v0.5-quality-generator",
            "created_from": "chunks",
            "hard_negative_strategy": hn_strategy,
            "chunks": {chunk.chunk_id: chunk.content for chunk in chunks},
            "quality_summary": quality_summary,
        },
    )


@dataclass
class HardNegativeCandidate:
    chunk: DocumentChunk
    similarity: float
    confusion_type: str
    reason: str
    signals: list[str] = field(default_factory=list)


def mine_hard_negatives(
    target: DocumentChunk,
    chunks: list[DocumentChunk],
    top_k: int = 1,
    strategy: str = "hybrid",
) -> list[HardNegativeCandidate]:
    if top_k <= 0:
        return []
    if strategy not in {"lexical", "hybrid"}:
        raise ValueError("strategy must be 'lexical' or 'hybrid'")

    candidates = []
    for index, chunk in enumerate(chunks):
        if chunk.chunk_id == target.chunk_id:
            continue
        target_index = _find_chunk_index(chunks, target.chunk_id)
        score, signals = _candidate_score(target, chunk, target_index, index, strategy)
        if score <= 0:
            continue
        confusion_type = infer_confusion_type(target, chunk)
        candidates.append(
            HardNegativeCandidate(
                chunk=chunk,
                similarity=round(score, 4),
                confusion_type=confusion_type,
                reason=(
                    f"Candidate selected by {strategy} signals: {', '.join(signals)}; "
                    f"different {confusion_type.replace('_', ' ')}."
                ),
                signals=signals,
            )
        )

    candidates.sort(key=lambda item: (-item.similarity, item.chunk.chunk_id))
    return candidates[:top_k]


def generate_query(chunk: DocumentChunk) -> str:
    topic = _metadata_topic(chunk)
    if topic:
        if _contains_cjk(topic):
            return f"{topic}的相关规定是什么？"
        return f"What does the document say about {topic}?"

    snippet = _first_sentence(chunk.content)
    if _contains_cjk(snippet):
        return f"根据资料，{snippet}相关内容是什么？"
    return f"What does the document say about {snippet}?"


def add_case(
    testset: TestSet,
    query: str,
    expected_chunk: str,
    hard_negative_ids: list[str] | None = None,
    confusion_type: str = "manual",
    difficulty: str = "medium",
    source_document: str = "",
    tags: list[str] | None = None,
    case_id: str | None = None,
) -> TestSet:
    if not query:
        raise ValueError("query is required")
    if not expected_chunk:
        raise ValueError("expected_chunk is required")

    existing_ids = {case.id for case in testset.cases}
    new_id = case_id or _next_manual_case_id(existing_ids)
    if new_id in existing_ids:
        raise ValueError(f"case id already exists: {new_id}")

    hard_negatives = [
        HardNegative(
            chunk_id=chunk_id,
            confusion_type=confusion_type,
            reason="Added manually from a known bad case.",
        )
        for chunk_id in hard_negative_ids or []
    ]
    metadata = {"created_from": "manual_bad_case", "tags": tags or []}
    testset.cases.append(
        TestCase(
            id=new_id,
            query=query,
            expected_chunks=[expected_chunk],
            hard_negatives=hard_negatives,
            difficulty=difficulty,  # type: ignore[arg-type]
            source_document=source_document,
            metadata=metadata,
        )
    )
    chunks = testset.metadata.setdefault("chunks", {})
    if isinstance(chunks, dict):
        chunks.setdefault(expected_chunk, "")
        for chunk_id in hard_negative_ids or []:
            chunks.setdefault(chunk_id, "")
    return testset


def sample_testset(testset: TestSet, limit: int = 10) -> list[dict[str, Any]]:
    rows = []
    for case in testset.cases[: max(limit, 0)]:
        rows.append(
            {
                "id": case.id,
                "query": case.query,
                "expected_chunks": case.expected_chunks,
                "hard_negatives": [
                    {
                        "chunk_id": item.chunk_id,
                        "confusion_type": item.confusion_type,
                        "similarity_to_correct": item.similarity_to_correct,
                    }
                    for item in case.hard_negatives
                ],
                "difficulty": case.difficulty,
                "source_document": case.source_document,
                "review": {"accepted": None, "notes": ""},
            }
        )
    return rows


def label_difficulty(hard_negatives: list[HardNegative]) -> str:
    if not hard_negatives:
        return "easy"
    max_similarity = max(item.similarity_to_correct or 0.0 for item in hard_negatives)
    if max_similarity >= STRONG_HARD_NEGATIVE_THRESHOLD:
        return "hard"
    if max_similarity >= WEAK_HARD_NEGATIVE_THRESHOLD:
        return "medium"
    return "easy"


def assess_case_quality(
    query: str,
    expected_chunk: DocumentChunk,
    hard_negatives: list[HardNegative],
) -> dict[str, Any]:
    warnings: list[str] = []
    score = 1.0

    if len(query.strip()) < MIN_QUERY_LENGTH:
        warnings.append("query_too_short")
        score -= 0.25
    if not expected_chunk.content.strip():
        warnings.append("empty_expected_chunk")
        score -= 0.4
    if not hard_negatives:
        warnings.append("missing_hard_negative")
        score -= 0.3
    else:
        max_similarity = max(item.similarity_to_correct or 0.0 for item in hard_negatives)
        if max_similarity < WEAK_HARD_NEGATIVE_THRESHOLD:
            warnings.append("weak_hard_negative")
            score -= 0.2
        if len({item.chunk_id for item in hard_negatives}) != len(hard_negatives):
            warnings.append("duplicate_hard_negative")
            score -= 0.2

    score = max(0.0, min(1.0, round(score, 3)))
    return {
        "score": score,
        "warnings": warnings,
        "filter_passed": score >= 0.6 and not {"empty_expected_chunk"} & set(warnings),
    }


def summarize_testset_quality(cases: list[TestCase]) -> dict[str, Any]:
    if not cases:
        return {
            "total_cases": 0,
            "hard_negative_coverage": 0.0,
            "average_hard_negatives_per_case": 0.0,
            "difficulty_distribution": {},
            "confusion_distribution": {},
            "warning_counts": {},
        }

    cases_with_hn = sum(1 for case in cases if case.hard_negatives)
    total_hn = sum(len(case.hard_negatives) for case in cases)
    difficulty_distribution = _ratio_counts([case.difficulty for case in cases])
    confusion_distribution = _ratio_counts(
        [hn.confusion_type for case in cases for hn in case.hard_negatives]
    )
    warnings = [
        warning
        for case in cases
        for warning in case.metadata.get("quality", {}).get("warnings", [])
    ]

    return {
        "total_cases": len(cases),
        "hard_negative_coverage": round(cases_with_hn / len(cases), 4),
        "average_hard_negatives_per_case": round(total_hn / len(cases), 4),
        "difficulty_distribution": difficulty_distribution,
        "confusion_distribution": confusion_distribution,
        "warning_counts": dict(sorted(_count_items(warnings).items())),
        "average_quality_score": round(
            sum(case.metadata.get("quality", {}).get("score", 0.0) for case in cases)
            / len(cases),
            4,
        ),
    }


def render_quality_report(testset: TestSet) -> str:
    summary = testset.metadata.get("quality_summary") or summarize_testset_quality(testset.cases)
    lines = [
        "# RAGProbe Testset Quality Report",
        "",
        "## Summary",
        "",
        f"- Testset: {testset.name or 'unnamed'}",
        f"- Total cases: {summary.get('total_cases', 0)}",
        f"- Hard negative coverage: {summary.get('hard_negative_coverage', 0.0):.1%}",
        "- Average hard negatives per case: "
        f"{summary.get('average_hard_negatives_per_case', 0.0):.2f}",
        f"- Average quality score: {summary.get('average_quality_score', 0.0):.3f}",
        "",
        "## Difficulty Distribution",
        "",
    ]
    _append_distribution(lines, summary.get("difficulty_distribution", {}))
    lines.extend(["", "## Confusion Distribution", ""])
    _append_distribution(lines, summary.get("confusion_distribution", {}))
    lines.extend(["", "## Warnings", ""])
    warning_counts = summary.get("warning_counts", {})
    if warning_counts:
        for warning, count in warning_counts.items():
            lines.append(f"- {warning}: {count}")
    else:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def content_similarity(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    return overlap / union if union else 0.0


def metadata_similarity(left: DocumentChunk, right: DocumentChunk) -> float:
    keys = set(left.metadata) | set(right.metadata)
    useful_keys = {
        key
        for key in keys
        if key
        in {
            "subject",
            "party",
            "actor",
            "condition",
            "event",
            "trigger",
            "scope",
            "section",
            "topic",
            "title",
            "document_type",
        }
    }
    if not useful_keys:
        return 0.0
    matches = sum(
        1
        for key in useful_keys
        if left.metadata.get(key) is not None
        and left.metadata.get(key) == right.metadata.get(key)
    )
    return matches / len(useful_keys)


def infer_confusion_type(target: DocumentChunk, candidate: DocumentChunk) -> str:
    target_meta = target.metadata
    candidate_meta = candidate.metadata

    for key in ("subject", "party", "actor"):
        if _differs(target_meta.get(key), candidate_meta.get(key)):
            return "subject_confusion"
    for key in ("condition", "event", "trigger"):
        if _differs(target_meta.get(key), candidate_meta.get(key)):
            return "event_confusion" if key == "event" else "condition_confusion"
    for key in ("date", "year", "version"):
        if _differs(target_meta.get(key), candidate_meta.get(key)):
            return "temporal_confusion"
    for key in ("scope", "section", "document_type"):
        if _differs(target_meta.get(key), candidate_meta.get(key)):
            return "scope_confusion"

    if _numbers(target.content) != _numbers(candidate.content):
        return "condition_confusion"
    if (
        target.source_document
        and candidate.source_document
        and target.source_document != candidate.source_document
    ):
        return "scope_confusion"
    return "semantic_only"


def _chunk_from_row(row: dict[str, Any], index: int) -> DocumentChunk:
    chunk_id = row.get("chunk_id") or row.get("id") or f"chunk_{index:03d}"
    content = row.get("content", row.get("text", ""))
    metadata = dict(row.get("metadata", {}))
    metadata_keys = (
        "subject",
        "party",
        "actor",
        "condition",
        "event",
        "date",
        "year",
        "version",
        "scope",
        "section",
        "topic",
        "title",
    )
    for key in metadata_keys:
        if key in row and key not in metadata:
            metadata[key] = row[key]
    source_document = (
        row.get("source_document")
        or row.get("source")
        or metadata.get("source_document", "")
    )
    return DocumentChunk(
        chunk_id=str(chunk_id),
        content=str(content),
        metadata=metadata,
        source_document=str(source_document),
    )


def _normalise_chunk_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("chunks"), list):
        return payload["chunks"]
    if isinstance(payload, dict) and all(isinstance(value, str) for value in payload.values()):
        return [{"chunk_id": key, "content": value} for key, value in payload.items()]
    raise ValueError("chunks JSON must be a list, {'chunks': [...]}, or {chunk_id: content}")


def _validate_chunks(chunks: list[DocumentChunk]) -> None:
    seen_ids: set[str] = set()
    for index, chunk in enumerate(chunks):
        if not chunk.chunk_id:
            raise ValueError(f"chunk[{index}] is missing chunk_id")
        if chunk.chunk_id in seen_ids:
            raise ValueError(f"duplicate chunk_id: {chunk.chunk_id}")
        if not chunk.content.strip():
            raise ValueError(f"chunk[{index}] is missing content")
        seen_ids.add(chunk.chunk_id)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def _next_manual_case_id(existing_ids: set[str]) -> str:
    index = 1
    while True:
        case_id = f"manual_case_{index:03d}"
        if case_id not in existing_ids:
            return case_id
        index += 1


def _tokens(text: str) -> set[str]:
    lowered = text.lower()
    words = set(re.findall(r"[a-z0-9]+", lowered))
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = {"".join(cjk_chars[index : index + 2]) for index in range(len(cjk_chars) - 1)}
    return words | cjk_bigrams


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\d+(?:\.\d+)?", text))


def _differs(left: Any, right: Any) -> bool:
    return left is not None and right is not None and str(left) != str(right)


def _contains_cjk(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _first_sentence(content: str) -> str:
    compact = " ".join(content.split())
    parts = re.split(r"[。！？.!?]", compact, maxsplit=1)
    snippet = parts[0].strip() if parts else compact
    return snippet[:60] or "this chunk"


def _metadata_topic(chunk: DocumentChunk) -> str:
    for key in ("topic", "title", "section", "subject", "event"):
        value = chunk.metadata.get(key)
        if value:
            return str(value)
    return ""


def _candidate_score(
    target: DocumentChunk,
    candidate: DocumentChunk,
    target_index: int,
    candidate_index: int,
    strategy: str,
) -> tuple[float, list[str]]:
    lexical = content_similarity(target.content, candidate.content)
    score = lexical
    signals = []
    if lexical > 0:
        signals.append("lexical")

    if strategy == "hybrid":
        meta = metadata_similarity(target, candidate)
        if meta > 0:
            score += meta * 0.2
            signals.append("metadata")
        if _same_section(target, candidate):
            score += 0.15
            signals.append("same_section")
        if target_index >= 0 and abs(target_index - candidate_index) == 1:
            score += 0.1
            signals.append("adjacent")
        if infer_confusion_type(target, candidate) != "semantic_only":
            score += 0.05
            signals.append("confusion_metadata")

    return min(score, 1.0), signals


def _same_section(left: DocumentChunk, right: DocumentChunk) -> bool:
    return bool(
        left.metadata.get("section")
        and right.metadata.get("section")
        and left.metadata.get("section") == right.metadata.get("section")
    )


def _find_chunk_index(chunks: list[DocumentChunk], chunk_id: str) -> int:
    for index, chunk in enumerate(chunks):
        if chunk.chunk_id == chunk_id:
            return index
    return -1


def _count_items(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return counts


def _ratio_counts(items: list[str]) -> dict[str, float]:
    counts = _count_items(items)
    total = sum(counts.values())
    if not total:
        return {}
    return {key: round(value / total, 4) for key, value in sorted(counts.items())}


def _append_distribution(lines: list[str], distribution: dict[str, float]) -> None:
    if not distribution:
        lines.append("- none")
        return
    for key, value in distribution.items():
        lines.append(f"- {key}: {value:.1%}")
