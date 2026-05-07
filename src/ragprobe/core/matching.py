"""Chunk matching for chunk_id first and content fallback diagnostics."""

from __future__ import annotations

from difflib import SequenceMatcher

from ragprobe.core.models import RetrievalResult, RetrievedChunk, TestSet


DEFAULT_CONTENT_MATCH_THRESHOLD = 0.9


def apply_content_fallback(
    testset: TestSet,
    results: list[RetrievalResult],
    threshold: float = DEFAULT_CONTENT_MATCH_THRESHOLD,
) -> list[RetrievalResult]:
    """Fill missing chunk IDs by matching retrieved content against testset chunk text.

    v0.2 still prefers exact chunk_id. Fallback matches are marked in chunk metadata so
    reports can warn users about inferred IDs.
    """
    chunk_catalog = _chunk_catalog(testset)
    if not chunk_catalog:
        return results

    matched_results: list[RetrievalResult] = []
    for result in results:
        matched_chunks = []
        for chunk in result.retrieved:
            if chunk.chunk_id:
                metadata = {
                    **chunk.metadata,
                    "ragprobe_match_method": chunk.metadata.get("ragprobe_match_method", "chunk_id"),
                    "ragprobe_match_confidence": chunk.metadata.get("ragprobe_match_confidence", 1.0),
                }
                matched_chunks.append(
                    RetrievedChunk(
                        content=chunk.content,
                        score=chunk.score,
                        metadata=metadata,
                        chunk_id=chunk.chunk_id,
                    )
                )
                continue

            matched_id, confidence = _best_content_match(chunk.content, chunk_catalog)
            metadata = dict(chunk.metadata)
            if matched_id is not None and confidence >= threshold:
                metadata["ragprobe_match_method"] = "content_fallback"
                metadata["ragprobe_match_confidence"] = confidence
                matched_chunks.append(
                    RetrievedChunk(
                        content=chunk.content,
                        score=chunk.score,
                        metadata=metadata,
                        chunk_id=matched_id,
                    )
                )
            else:
                metadata["ragprobe_match_method"] = "unmatched"
                metadata["ragprobe_match_confidence"] = confidence
                matched_chunks.append(
                    RetrievedChunk(
                        content=chunk.content,
                        score=chunk.score,
                        metadata=metadata,
                        chunk_id=None,
                    )
                )

        matched_results.append(
            RetrievalResult(
                test_case_id=result.test_case_id,
                query=result.query,
                retrieved=matched_chunks,
                hit=result.hit,
                correct_rank=result.correct_rank,
                false_positives=list(result.false_positives),
            )
        )

    return matched_results


def collect_match_stats(results: list[RetrievalResult]) -> dict[str, int]:
    stats = {"chunk_id": 0, "content_fallback": 0, "unmatched": 0}
    for result in results:
        for chunk in result.retrieved:
            method = chunk.metadata.get("ragprobe_match_method")
            if method in stats:
                stats[method] += 1
    return stats


def _chunk_catalog(testset: TestSet) -> dict[str, str]:
    chunks = testset.metadata.get("chunks", {})
    if isinstance(chunks, dict):
        return {str(chunk_id): str(content) for chunk_id, content in chunks.items()}
    return {}


def _best_content_match(content: str, catalog: dict[str, str]) -> tuple[str | None, float]:
    best_id: str | None = None
    best_score = 0.0
    for chunk_id, expected_content in catalog.items():
        score = SequenceMatcher(None, _normalize(content), _normalize(expected_content)).ratio()
        if score > best_score:
            best_id = chunk_id
            best_score = score
    return best_id, best_score


def _normalize(text: str) -> str:
    return "".join(str(text).split()).lower()
