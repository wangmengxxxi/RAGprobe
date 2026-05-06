from __future__ import annotations

import re
from collections import Counter

from qualira.core.schema import EvidenceUnit


def naive_topk(query: str, units: list[EvidenceUnit], k: int = 5) -> list[tuple[EvidenceUnit, float]]:
    scored = [(unit, score_text_similarity(query, unit.content)) for unit in units]
    scored.sort(key=lambda item: (-item[1], item[0].id))
    return [(unit, score) for unit, score in scored[:k] if score > 0]


def score_text_similarity(query: str, text: str) -> float:
    query_terms = _features(query)
    text_terms = _features(text)
    if not query_terms or not text_terms:
        return 0.0
    intersection = query_terms & text_terms
    numerator = sum(intersection.values())
    denominator = (sum(value * value for value in query_terms.values()) ** 0.5) * (
        sum(value * value for value in text_terms.values()) ** 0.5
    )
    return round(numerator / denominator, 6) if denominator else 0.0


def _features(text: str) -> Counter[str]:
    normalized = text.lower()
    tokens = re.findall(r"[a-z0-9_]+", normalized)
    cjk = re.findall(r"[\u4e00-\u9fff]", normalized)
    bigrams = ["".join(cjk[index : index + 2]) for index in range(max(len(cjk) - 1, 0))]
    trigrams = ["".join(cjk[index : index + 3]) for index in range(max(len(cjk) - 2, 0))]
    return Counter(tokens + bigrams + trigrams)
