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
from ragprobe.core.repair import RepairApplyResult, RepairPlan

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "RAGProbe",
    "AuditReport",
    "ExperimentReport",
    "RepairApplyResult",
    "RepairPlan",
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
