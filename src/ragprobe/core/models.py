"""Data models for RAGProbe diagnostic artifacts."""

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
    score: float | None = None
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
    query: str = ""
    confusion_type: str | None = None
    correct_chunk_similarity: float | None = None
    false_positive_similarity: float | None = None
    correct_rank: int | None = None
    false_positives: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    difficulty: Difficulty | None = None


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
class MetricSignal:
    name: str
    severity: Severity
    summary: str
    evidence: str


@dataclass
class DiagnosticReport:
    hit_rate: float = 0.0
    mrr: float = 0.0
    precision_at_k: dict[int, float] = field(default_factory=dict)
    fpr: float = 0.0
    failure_cases: list[FailureCase] = field(default_factory=list)
    confusion_distribution: dict[str, float] = field(default_factory=dict)
    system_issues: list[SystemIssue] = field(default_factory=list)
    metric_signals: list[MetricSignal] = field(default_factory=list)
    recommendations: list[Recommendation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricDelta:
    metric: str
    before: float
    after: float
    delta: float


@dataclass
class ComparisonReport:
    before: DiagnosticReport
    after: DiagnosticReport
    deltas: list[MetricDelta] = field(default_factory=list)
    improved_cases: list[str] = field(default_factory=list)
    regressed_cases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
