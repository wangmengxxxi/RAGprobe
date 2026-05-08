"""Built-in local baseline retrievers."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

from ragprobe.core.matching import apply_content_fallback
from ragprobe.core.models import RetrievalResult, RetrievedChunk, TestSet
from ragprobe.core.validation import validate_results

BaselineName = str

SUPPORTED_BASELINES = {"lexical", "embedding"}
DEFAULT_EMBEDDING_DIMENSIONS = 256


def run_baseline_retriever(
    testset: TestSet,
    baseline: BaselineName = "embedding",
    *,
    top_k: int = 10,
    dimensions: int = DEFAULT_EMBEDDING_DIMENSIONS,
    content_fallback_threshold: float = 0.9,
) -> list[RetrievalResult]:
    """Run a deterministic local retriever over testset.metadata.chunks."""

    if baseline not in SUPPORTED_BASELINES:
        raise ValueError(
            f"unsupported baseline retriever: {baseline}; "
            f"expected one of {', '.join(sorted(SUPPORTED_BASELINES))}"
        )
    if top_k <= 0:
        raise ValueError("top_k must be greater than 0")
    if dimensions <= 0:
        raise ValueError("dimensions must be greater than 0")

    chunks = _load_chunk_corpus(testset)
    index = _build_index(chunks, baseline=baseline, dimensions=dimensions)
    results = [
        RetrievalResult(
            test_case_id=case.id,
            query=case.query,
            retrieved=_retrieve(
                case.query,
                index,
                baseline=baseline,
                top_k=top_k,
                dimensions=dimensions,
            ),
        )
        for case in testset.cases
    ]
    matched = apply_content_fallback(testset, results, threshold=content_fallback_threshold)
    validate_results(testset, matched)
    return matched


def _load_chunk_corpus(testset: TestSet) -> dict[str, str]:
    chunks = testset.metadata.get("chunks")
    if not isinstance(chunks, dict) or not chunks:
        raise ValueError("testset.metadata.chunks is required for built-in baseline retrieval")
    corpus = {str(chunk_id): str(content) for chunk_id, content in chunks.items()}
    if not any(content.strip() for content in corpus.values()):
        raise ValueError("testset.metadata.chunks must contain non-empty chunk content")
    return corpus


def _build_index(
    chunks: dict[str, str],
    *,
    baseline: BaselineName,
    dimensions: int,
) -> list[dict]:
    return [
        {
            "chunk_id": chunk_id,
            "content": content,
            "tokens": _tokens(content),
            "vector": _hashed_vector(content, dimensions) if baseline == "embedding" else None,
        }
        for chunk_id, content in chunks.items()
    ]


def _retrieve(
    query: str,
    index: list[dict],
    *,
    baseline: BaselineName,
    top_k: int,
    dimensions: int,
) -> list[RetrievedChunk]:
    query_tokens = _tokens(query)
    query_vector = _hashed_vector(query, dimensions) if baseline == "embedding" else None
    ranked = []
    for row in index:
        if baseline == "embedding":
            score = _cosine(query_vector or {}, row["vector"] or {})
        else:
            score = _lexical_score(query_tokens, row["tokens"])
        ranked.append((score, row["chunk_id"], row["content"]))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            content=content,
            score=round(score, 6),
            metadata={"baseline_retriever": baseline},
        )
        for score, chunk_id, content in ranked[:top_k]
    ]


def _tokens(text: str) -> list[str]:
    lowered = text.lower()
    words = re.findall(r"[a-z0-9]+", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", text)
    cjk_bigrams = [
        "".join(cjk_chars[index : index + 2])
        for index in range(len(cjk_chars) - 1)
    ]
    return words + cjk_bigrams


def _lexical_score(query_tokens: list[str], chunk_tokens: list[str]) -> float:
    query_set = set(query_tokens)
    chunk_set = set(chunk_tokens)
    if not query_set or not chunk_set:
        return 0.0
    overlap = len(query_set & chunk_set)
    union = len(query_set | chunk_set)
    return overlap / union if union else 0.0


def _hashed_vector(text: str, dimensions: int) -> dict[int, float]:
    counts = Counter(_tokens(text))
    vector: dict[int, float] = {}
    for token, count in counts.items():
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] = vector.get(bucket, 0.0) + sign * (1.0 + math.log(count))
    return vector


def _cosine(left: dict[int, float], right: dict[int, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(index, 0.0) for index, value in left.items())
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if not left_norm or not right_norm:
        return 0.0
    return max(0.0, dot / (left_norm * right_norm))
