"""Domain-neutral confusion type helpers."""

from __future__ import annotations

import re
from typing import Any

NON_SEMANTIC_METADATA_KEYS = {
    "chunk_id",
    "content",
    "created_at",
    "doc_id",
    "file",
    "filename",
    "hash",
    "id",
    "index",
    "line",
    "line_number",
    "offset",
    "page",
    "page_number",
    "path",
    "position",
    "source",
    "source_document",
    "timestamp",
    "updated_at",
    "url",
}

GENERIC_CONFUSION_TYPES = {
    "semantic_only",
    "numeric_confusion",
    "source_confusion",
    "manual",
}

SEMANTIC_METADATA_KEY_HINTS = (
    "actor",
    "brand",
    "category",
    "condition",
    "department",
    "difficulty",
    "dosage",
    "drug",
    "event",
    "grade",
    "indication",
    "intent",
    "issue",
    "party",
    "price",
    "product",
    "scope",
    "section",
    "seller",
    "subject",
    "tag",
    "topic",
    "trigger",
    "type",
    "version",
    "year",
)


def is_semantic_metadata_key(key: str) -> bool:
    normalized = _normalize_key(key)
    if not normalized or normalized in NON_SEMANTIC_METADATA_KEYS:
        return False
    if normalized.endswith("_id") or normalized.endswith("_ids"):
        return False
    return any(hint in normalized for hint in SEMANTIC_METADATA_KEY_HINTS)


def semantic_metadata_keys(*metadata_items: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    for metadata in metadata_items:
        keys.update(str(key) for key in metadata)
    return {key for key in keys if is_semantic_metadata_key(key)}


def is_metadata_confusion_type(label: str) -> bool:
    if not label or label in GENERIC_CONFUSION_TYPES:
        return False
    if not label.endswith("_confusion"):
        return False
    key = label.removesuffix("_confusion")
    return is_semantic_metadata_key(key)


def metadata_confusion_type(key: str) -> str:
    return f"{_normalize_key(key)}_confusion"


def _normalize_key(key: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", str(key).strip().lower()).strip("_")
