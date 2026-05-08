"""RAGProbe public package surface."""

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

__version__ = "0.6.0"

__all__ = [
    "__version__",
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
