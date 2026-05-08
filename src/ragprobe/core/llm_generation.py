"""Optional LLM-assisted testset generation."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ragprobe.core.generator import (
    DocumentChunk,
    assess_case_quality,
    content_similarity,
    label_difficulty,
    mine_hard_negatives,
    summarize_testset_quality,
)
from ragprobe.core.models import HardNegative, TestCase, TestSet
from ragprobe.core.schema import SCHEMA_TESTSET, schema_metadata

DEFAULT_QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
DEFAULT_QWEN_MODEL = "qwen-plus"
PROMPT_VERSION = "ragprobe-v0.7-openai-compatible-generation-v1"
EXTRACTIVE_QUERY_THRESHOLD = 0.8
GENERIC_QUERY_TOKENS = {
    "相关规定",
    "相关内容",
    "如何处理",
    "怎么办",
    "what",
    "about",
    "document",
}


class LLMGenerationError(RuntimeError):
    """Raised when optional LLM generation cannot complete."""


@dataclass
class LLMGenerationConfig:
    provider: str = "openai-compatible"
    model: str = DEFAULT_QWEN_MODEL
    base_url: str = DEFAULT_QWEN_BASE_URL
    api_key_env: str = "AI_API_KEY"
    hard_negative_top_k: int = 1
    hn_strategy: str = "hybrid"
    prompt_version: str = PROMPT_VERSION
    temperature: float = 0.2
    domain_hint: str | None = None


@dataclass
class LLMHardNegativeDecision:
    chunk_id: str
    accepted: bool = True
    confusion_type: str = "semantic_only"
    confidence: float | None = None
    reason: str = ""


@dataclass
class LLMGeneratedCase:
    query: str
    hard_negatives: list[LLMHardNegativeDecision] = field(default_factory=list)


@dataclass
class LLMJudgeDecision:
    answerable: bool
    confidence: float | None = None
    reason: str = ""


@dataclass
class LLMCaseValidation:
    status: str = "passed"
    warnings: list[str] = field(default_factory=list)
    rejected: bool = False
    expected_answerable: bool | None = None
    expected_confidence: float | None = None
    expected_reason: str = ""
    removed_hard_negatives: list[str] = field(default_factory=list)


class LLMClient(Protocol):
    def generate_case(
        self,
        target: DocumentChunk,
        candidates: list[DocumentChunk],
        config: LLMGenerationConfig,
    ) -> LLMGeneratedCase:
        """Generate one retrieval test case for a target chunk."""


class LLMJudgeClient(Protocol):
    def judge_answerability(
        self,
        query: str,
        chunk: DocumentChunk,
        role: str,
        config: LLMGenerationConfig,
    ) -> LLMJudgeDecision:
        """Judge whether a chunk can answer a query."""


class OpenAICompatibleClient:
    """Generic client for OpenAI-compatible chat completions endpoints."""

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_QWEN_MODEL,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        timeout: float = 60.0,
    ) -> None:
        if not api_key:
            raise LLMGenerationError("API key is required for LLM generation")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url
        self.timeout = timeout

    @classmethod
    def from_env(
        cls,
        env_var: str = "AI_API_KEY",
        model: str = DEFAULT_QWEN_MODEL,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        timeout: float = 60.0,
    ) -> "OpenAICompatibleClient":
        return cls(
            api_key=os.environ.get(env_var, ""),
            model=model,
            base_url=base_url,
            timeout=timeout,
        )

    def generate_case(
        self,
        target: DocumentChunk,
        candidates: list[DocumentChunk],
        config: LLMGenerationConfig,
    ) -> LLMGeneratedCase:
        payload = {
            "model": self.model,
            "temperature": config.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You create retrieval test cases for RAG diagnostics. "
                        "Return JSON only. Do not generate final answers. "
                        "The query must be answerable by the expected chunk. "
                        "Hard negatives must be similar but wrong for the query. "
                        "Infer the appropriate query language and style from the chunk content."
                    ),
                },
                {
                    "role": "user",
                    "content": build_generation_prompt(target, candidates, config.domain_hint),
                },
            ],
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMGenerationError(f"LLM API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMGenerationError(f"LLM API request failed: {exc.reason}") from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMGenerationError("LLM API response did not contain choices[0].message.content") from exc
        return parse_generated_case(content)

    def judge_answerability(
        self,
        query: str,
        chunk: DocumentChunk,
        role: str,
        config: LLMGenerationConfig,
    ) -> LLMJudgeDecision:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You audit retrieval test cases. Return JSON only. "
                        "Decide whether the provided chunk can answer the user query. "
                        "Do not answer the user query."
                    ),
                },
                {
                    "role": "user",
                    "content": build_judge_prompt(query=query, chunk=chunk, role=role),
                },
            ],
        }
        request = urllib.request.Request(
            self.base_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise LLMGenerationError(f"LLM judge API HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise LLMGenerationError(f"LLM judge API request failed: {exc.reason}") from exc

        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMGenerationError("LLM judge response did not contain choices[0].message.content") from exc
        return parse_judge_decision(content)


class QwenClient(OpenAICompatibleClient):
    """Alibaba Qwen preset using DashScope compatible mode."""

    @classmethod
    def from_env(
        cls,
        env_var: str = "AI_API_KEY",
        model: str = DEFAULT_QWEN_MODEL,
        base_url: str = DEFAULT_QWEN_BASE_URL,
        timeout: float = 60.0,
    ) -> "QwenClient":
        return cls(
            api_key=os.environ.get(env_var, ""),
            model=model,
            base_url=base_url,
            timeout=timeout,
        )


def generate_testset_from_chunks_llm(
    chunks: list[DocumentChunk],
    client: LLMClient,
    num_cases: int | None = None,
    hard_negative_top_k: int = 1,
    name: str = "generated-testset",
    hn_strategy: str = "hybrid",
    cache_dir: str | Path | None = ".ragprobe_cache",
    use_cache: bool = True,
    config: LLMGenerationConfig | None = None,
    validate_rules: bool = True,
    judge_client: LLMJudgeClient | None = None,
    keep_rejected: bool = False,
) -> TestSet:
    if hn_strategy not in {"lexical", "hybrid"}:
        raise ValueError("hn_strategy must be 'lexical' or 'hybrid'")
    if not chunks:
        raise ValueError("chunks must contain at least one item")
    if num_cases is not None and num_cases <= 0:
        raise ValueError("num_cases must be greater than 0")
    if hard_negative_top_k < 0:
        raise ValueError("hard_negative_top_k must be greater than or equal to 0")

    config = config or LLMGenerationConfig(hard_negative_top_k=hard_negative_top_k, hn_strategy=hn_strategy)
    selected = chunks[:num_cases] if num_cases is not None else chunks
    cases: list[TestCase] = []
    cache_path = Path(cache_dir) if cache_dir and use_cache else None

    for index, chunk in enumerate(selected, start=1):
        candidate_count = max(hard_negative_top_k * 3, hard_negative_top_k, 1)
        mined = mine_hard_negatives(chunk, chunks, top_k=candidate_count, strategy=hn_strategy)
        candidate_chunks = [candidate.chunk for candidate in mined]
        cached = False
        generated = None

        if cache_path:
            generated = _read_cached_case(cache_path, chunk, candidate_chunks, config)
            cached = generated is not None
        if generated is None:
            generated = client.generate_case(chunk, candidate_chunks, config)
            if cache_path:
                _write_cached_case(cache_path, chunk, candidate_chunks, config, generated)

        hard_negatives = _llm_decisions_to_hard_negatives(
            generated,
            mined,
            hard_negative_top_k,
        )
        validation = validate_llm_generated_case(
            generated=generated,
            expected_chunk=chunk,
            hard_negatives=hard_negatives,
            chunks_by_id={item.chunk_id: item for item in chunks},
            validate_rules=validate_rules,
            judge_client=judge_client,
            config=config,
        )
        if validation.rejected and not keep_rejected:
            continue
        quality = assess_case_quality(
            query=generated.query,
            expected_chunk=chunk,
            hard_negatives=hard_negatives,
        )
        quality = _merge_validation_into_quality(quality, validation)
        cases.append(
            TestCase(
                id=f"generated_case_{index:03d}",
                query=generated.query,
                expected_chunks=[chunk.chunk_id],
                hard_negatives=hard_negatives,
                difficulty=label_difficulty(hard_negatives),
                source_document=chunk.source_document,
                metadata={
                    "created_from": "chunks",
                    "source_chunk_id": chunk.chunk_id,
                    "generator_mode": "llm",
                    "llm_provider": config.provider,
                    "llm_model": config.model,
                    "llm_prompt_version": config.prompt_version,
                    "llm_cache_hit": cached,
                    "hard_negative_strategy": hn_strategy,
                    "tags": list(chunk.metadata.get("tags", [])),
                    "quality": quality,
                    "validation": _validation_to_dict(validation),
                },
            )
        )

    return TestSet(
        cases=cases,
        name=name,
        metadata={
            **schema_metadata(SCHEMA_TESTSET),
            "source": "ragprobe-v0.7-llm-generator",
            "created_from": "chunks",
            "generator_mode": "llm",
            "llm_provider": config.provider,
            "llm_model": config.model,
            "llm_base_url": _redact_url(config.base_url),
            "llm_api_key_env": config.api_key_env,
            "llm_prompt_version": config.prompt_version,
            "hard_negative_strategy": hn_strategy,
            "chunks": {chunk.chunk_id: chunk.content for chunk in chunks},
            "quality_summary": summarize_testset_quality(cases),
            "validation_summary": summarize_llm_validation(cases),
        },
    )


def estimate_llm_generation(chunks: list[DocumentChunk], num_cases: int | None = None) -> dict[str, Any]:
    selected = chunks[:num_cases] if num_cases is not None else chunks
    input_chars = sum(len(chunk.content) + len(json.dumps(chunk.metadata, ensure_ascii=False)) for chunk in selected)
    estimated_input_tokens = max(1, round(input_chars / 2.5))
    estimated_output_tokens = max(1, len(selected) * 220)
    return {
        "calls": len(selected),
        "estimated_input_tokens": estimated_input_tokens,
        "estimated_output_tokens": estimated_output_tokens,
        "note": "Rough estimate only; provider billing depends on the selected model.",
    }


def build_generation_prompt(target: DocumentChunk, candidates: list[DocumentChunk], domain_hint: str | None = None) -> str:
    payload = {
        "task": (
            "Generate one retrieval test case. Create a natural user query for the expected chunk. "
            "Judge which candidate chunks are hard negatives: similar but wrong for this query."
        ),
        "output_schema": {
            "query": "string",
            "hard_negatives": [
                {
                    "chunk_id": "candidate chunk id",
                    "accepted": True,
                    "confusion_type": "string - the dimension of confusion between this chunk and the expected chunk, e.g. subject_confusion, brand_confusion, category_confusion, indication_confusion, temporal_confusion, or any label describing WHY they are confusable",
                    "confidence": 0.0,
                    "reason": "short explanation",
                }
            ],
        },
        "expected_chunk": _chunk_prompt_payload(target),
        "candidate_chunks": [_chunk_prompt_payload(candidate) for candidate in candidates],
    }
    if domain_hint:
        payload["domain_context"] = domain_hint
    return json.dumps(payload, ensure_ascii=False, indent=2)


def build_judge_prompt(query: str, chunk: DocumentChunk, role: str) -> str:
    payload = {
        "task": "Judge whether the chunk can answer the query.",
        "role": role,
        "query": query,
        "chunk": _chunk_prompt_payload(chunk),
        "output_schema": {
            "answerable": True,
            "confidence": 0.0,
            "reason": "short explanation",
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def parse_generated_case(content: str) -> LLMGeneratedCase:
    payload = _extract_json_object(content)
    query = str(payload.get("query", "")).strip()
    if not query:
        raise LLMGenerationError("LLM response is missing query")
    hard_negatives = []
    for item in payload.get("hard_negatives", []):
        if not isinstance(item, dict) or not item.get("chunk_id"):
            continue
        hard_negatives.append(
            LLMHardNegativeDecision(
                chunk_id=str(item["chunk_id"]),
                accepted=bool(item.get("accepted", True)),
                confusion_type=str(item.get("confusion_type", "semantic_only")),
                confidence=_optional_float(item.get("confidence")),
                reason=str(item.get("reason", "")),
            )
        )
    return LLMGeneratedCase(query=query, hard_negatives=hard_negatives)


def parse_judge_decision(content: str) -> LLMJudgeDecision:
    payload = _extract_json_object(content)
    return LLMJudgeDecision(
        answerable=bool(payload.get("answerable", False)),
        confidence=_optional_float(payload.get("confidence")),
        reason=str(payload.get("reason", "")),
    )


def validate_llm_generated_case(
    generated: LLMGeneratedCase,
    expected_chunk: DocumentChunk,
    hard_negatives: list[HardNegative],
    chunks_by_id: dict[str, DocumentChunk],
    validate_rules: bool = True,
    judge_client: LLMJudgeClient | None = None,
    config: LLMGenerationConfig | None = None,
) -> LLMCaseValidation:
    validation = LLMCaseValidation()
    query = generated.query.strip()

    if validate_rules:
        _apply_rule_validation(validation, query, expected_chunk, hard_negatives, chunks_by_id)

    if judge_client is not None:
        if config is None:
            config = LLMGenerationConfig()
        expected_decision = judge_client.judge_answerability(
            query=query,
            chunk=expected_chunk,
            role="expected_chunk",
            config=config,
        )
        validation.expected_answerable = expected_decision.answerable
        validation.expected_confidence = expected_decision.confidence
        validation.expected_reason = expected_decision.reason
        if not expected_decision.answerable:
            validation.warnings.append("expected_chunk_not_answerable")
            validation.rejected = True

        kept_hard_negatives = []
        for hard_negative in hard_negatives:
            candidate = chunks_by_id.get(hard_negative.chunk_id)
            if candidate is None:
                kept_hard_negatives.append(hard_negative)
                continue
            decision = judge_client.judge_answerability(
                query=query,
                chunk=candidate,
                role="hard_negative",
                config=config,
            )
            if decision.answerable:
                validation.warnings.append("hard_negative_answerable")
                validation.removed_hard_negatives.append(hard_negative.chunk_id)
            else:
                kept_hard_negatives.append(hard_negative)
        hard_negatives[:] = kept_hard_negatives

    if validation.rejected:
        validation.status = "rejected"
    elif validation.warnings:
        validation.status = "warning"
    else:
        validation.status = "passed"
    return validation


def summarize_llm_validation(cases: list[TestCase]) -> dict[str, Any]:
    statuses = [
        str(case.metadata.get("validation", {}).get("status", "unknown"))
        for case in cases
    ]
    warnings = [
        warning
        for case in cases
        for warning in case.metadata.get("validation", {}).get("warnings", [])
    ]
    removed = sum(
        len(case.metadata.get("validation", {}).get("removed_hard_negatives", []))
        for case in cases
    )
    return {
        "status_distribution": _ratio_counts(statuses),
        "warning_counts": _count_items(warnings),
        "removed_hard_negatives": removed,
    }


def _apply_rule_validation(
    validation: LLMCaseValidation,
    query: str,
    expected_chunk: DocumentChunk,
    hard_negatives: list[HardNegative],
    chunks_by_id: dict[str, DocumentChunk],
) -> None:
    if not query.strip():
        validation.warnings.append("missing_query")
        validation.rejected = True
        return

    if len(query.strip()) < 4:
        validation.warnings.append("query_too_short")
        validation.rejected = True

    extractive_score = lcs_ratio(query, expected_chunk.content)
    if extractive_score > EXTRACTIVE_QUERY_THRESHOLD:
        validation.warnings.append("query_too_extractive")

    if _is_generic_query(query):
        validation.warnings.append("query_too_generic")

    expected_overlap = content_similarity(query, expected_chunk.content)
    for hard_negative in hard_negatives:
        candidate = chunks_by_id.get(hard_negative.chunk_id)
        if candidate is None:
            continue
        candidate_overlap = content_similarity(query, candidate.content)
        if candidate_overlap > 0 and candidate_overlap >= expected_overlap:
            validation.warnings.append("ambiguous_query")
            break


def _merge_validation_into_quality(
    quality: dict[str, Any],
    validation: LLMCaseValidation,
) -> dict[str, Any]:
    merged = dict(quality)
    warnings = list(merged.get("warnings", []))
    for warning in validation.warnings:
        if warning not in warnings:
            warnings.append(warning)
    score = float(merged.get("score", 1.0))
    if validation.rejected:
        score -= 0.5
    elif validation.status == "warning":
        score -= min(0.3, 0.1 * len(validation.warnings))
    score = max(0.0, min(1.0, round(score, 3)))
    merged["score"] = score
    merged["warnings"] = warnings
    merged["filter_passed"] = bool(merged.get("filter_passed", True)) and not validation.rejected
    return merged


def _validation_to_dict(validation: LLMCaseValidation) -> dict[str, Any]:
    return {
        "status": validation.status,
        "warnings": validation.warnings,
        "rejected": validation.rejected,
        "expected_answerable": validation.expected_answerable,
        "expected_confidence": validation.expected_confidence,
        "expected_reason": validation.expected_reason,
        "removed_hard_negatives": validation.removed_hard_negatives,
    }


def lcs_ratio(left: str, right: str) -> float:
    left = "".join(left.split())
    right = "".join(right.split())
    if not left or not right:
        return 0.0
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(current[-1], previous[index]))
        previous = current
    return previous[-1] / max(1, len(left))


def _is_generic_query(query: str) -> bool:
    compact = "".join(query.lower().split())
    if len(compact) <= 8:
        return True
    return any(token in compact for token in GENERIC_QUERY_TOKENS) and len(compact) <= 18


def _count_items(items: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items()))


def _ratio_counts(items: list[str]) -> dict[str, float]:
    counts = _count_items(items)
    total = sum(counts.values())
    if not total:
        return {}
    return {key: round(value / total, 4) for key, value in counts.items()}


def _llm_decisions_to_hard_negatives(
    generated: LLMGeneratedCase,
    mined: list,
    top_k: int,
) -> list[HardNegative]:
    mined_by_id = {candidate.chunk.chunk_id: candidate for candidate in mined}
    selected: list[HardNegative] = []
    for decision in generated.hard_negatives:
        if not decision.accepted or decision.chunk_id not in mined_by_id:
            continue
        candidate = mined_by_id[decision.chunk_id]
        reason = decision.reason or candidate.reason
        if decision.confidence is not None:
            reason = f"{reason} LLM confidence: {decision.confidence:.2f}."
        selected.append(
            HardNegative(
                chunk_id=decision.chunk_id,
                confusion_type=decision.confusion_type or candidate.confusion_type,
                similarity_to_correct=candidate.similarity,
                reason=reason,
            )
        )
        if len(selected) >= top_k:
            return selected

    for candidate in mined:
        if any(item.chunk_id == candidate.chunk.chunk_id for item in selected):
            continue
        selected.append(
            HardNegative(
                chunk_id=candidate.chunk.chunk_id,
                confusion_type=candidate.confusion_type,
                similarity_to_correct=candidate.similarity,
                reason=f"Fallback deterministic candidate. {candidate.reason}",
            )
        )
        if len(selected) >= top_k:
            break
    return selected


def _read_cached_case(
    cache_dir: Path,
    target: DocumentChunk,
    candidates: list[DocumentChunk],
    config: LLMGenerationConfig,
) -> LLMGeneratedCase | None:
    path = _cache_file(cache_dir, target, candidates, config)
    if not path.exists():
        return None
    return parse_generated_case(path.read_text(encoding="utf-8"))


def _write_cached_case(
    cache_dir: Path,
    target: DocumentChunk,
    candidates: list[DocumentChunk],
    config: LLMGenerationConfig,
    generated: LLMGeneratedCase,
) -> None:
    path = _cache_file(cache_dir, target, candidates, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "query": generated.query,
        "hard_negatives": [
            {
                "chunk_id": item.chunk_id,
                "accepted": item.accepted,
                "confusion_type": item.confusion_type,
                "confidence": item.confidence,
                "reason": item.reason,
            }
            for item in generated.hard_negatives
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _cache_file(
    cache_dir: Path,
    target: DocumentChunk,
    candidates: list[DocumentChunk],
    config: LLMGenerationConfig,
) -> Path:
    key_payload = {
        "prompt_version": config.prompt_version,
        "provider": config.provider,
        "model": config.model,
        "target": _chunk_prompt_payload(target),
        "candidates": [_chunk_prompt_payload(candidate) for candidate in candidates],
        "hard_negative_top_k": config.hard_negative_top_k,
        "hn_strategy": config.hn_strategy,
    }
    digest = hashlib.sha256(json.dumps(key_payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return cache_dir / "llm_generation" / f"{digest}.json"


def _chunk_prompt_payload(chunk: DocumentChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "content": chunk.content,
        "metadata": chunk.metadata,
        "source_document": chunk.source_document,
    }


def _redact_url(url: str) -> str:
    return url.split("?")[0]


def _extract_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1)
    else:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise LLMGenerationError(f"LLM response is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMGenerationError("LLM response JSON must be an object")
    return payload


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
