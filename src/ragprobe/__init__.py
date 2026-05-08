"""RAGProbe public package surface."""

from ragprobe.api import RAGProbe
from ragprobe.core.models import (
    DiagnosticReport,
    FailureCase,
    FailurePattern,
    HardNegative,
    ComparisonReport,
    MetricSignal,
    MetricDelta,
    Recommendation,
    RetrievalResult,
    RetrievedChunk,
    SystemIssue,
    TestCase,
    TestSet,
)

__version__ = "0.7.0"

__all__ = [
    "__version__",
    "RAGProbe",
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
