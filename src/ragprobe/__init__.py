"""RAGProbe public package surface."""

from ragprobe.api import RAGProbe
from ragprobe.core.audit import AuditReport
from ragprobe.core.experiment import ExperimentReport
from ragprobe.core.models import (
    ComparisonReport,
    DiagnosticReport,
    FailureCase,
    FailurePattern,
    HardNegative,
    MetricDelta,
    MetricSignal,
    Recommendation,
    RetrievalResult,
    RetrievedChunk,
    SystemIssue,
    TestCase,
    TestSet,
)

__version__ = "0.9.0"

__all__ = [
    "__version__",
    "RAGProbe",
    "AuditReport",
    "ExperimentReport",
    "DiagnosticReport",
    "FailureCase",
    "FailurePattern",
    "HardNegative",
    "ComparisonReport",
    "MetricSignal",
    "MetricDelta",
    "Recommendation",
    "RetrievalResult",
    "RetrievedChunk",
    "SystemIssue",
    "TestCase",
    "TestSet",
]
