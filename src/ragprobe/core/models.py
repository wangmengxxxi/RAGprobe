"""Phase 0 data model skeletons for the diagnostic core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Difficulty = Literal["easy", "medium", "hard"]
FailureType = Literal["miss", "false_positive", "both"]
Severity = Literal["high", "medium", "low"]


@dataclass
class HardNegative:
    chunk_id: str
    confusion_type: str
    similarity_to_correct: float | None = None
    reason: str = ""


@dataclass
class TestCase:
    id: str
    query: str
    expected_chunks: list[str]
    hard_negatives: list[HardNegative] = field(default_factory=list)
    difficulty: Difficulty = "medium"
    source_document: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TestSet:
    cases: list[TestCase]
    name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    content: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_id: str | None = None


@dataclass
class RetrievalResult:
    test_case_id: str
    query: str
    retrieved: list[RetrievedChunk]
    hit: bool | None = None
    correct_rank: int | None = None
    false_positives: list[str] = field(default_factory=list)


@dataclass
class FailureCase:
    test_case_id: str
    failure_type: FailureType
    confusion_type: str | None = None
    correct_chunk_similarity: float | None = None
    false_positive_similarity: float | None = None


@dataclass
class SystemIssue:
    issue_type: str
    severity: Severity
    evidence: str
    affected_percentage: float


@dataclass
class Recommendation:
    priority: int
    action: str
    expected_impact: str = ""
    effort: str = ""


@dataclass
class DiagnosticReport:
    hit_rate: float = 0.0
    mrr: float = 0.0
    precision_at_k: dict[int, float] = field(default_factory=dict)
    fpr: float = 0.0
    failure_cases: list[FailureCase] = field(default_factory=list)
    confusion_distribution: dict[str, float] = field(default_factory=dict)
    system_issues: list[SystemIssue] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
