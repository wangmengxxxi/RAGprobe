"""RAGProbe public package surface."""

from ragprobe.api import RAGProbe
from ragprobe.core.audit import AuditReport
from ragprobe.core.baseline import run_baseline_retriever
from ragprobe.core.experiment import ExperimentReport
from ragprobe.core.models import (
    ComparisonReport,
    DiagnosticReport,
    FailureCase,
    FailurePattern,
    HardNegative,
    MetricDelta,
    MetricSignal,
    PipelineResult,
    Recommendation,
    RetrievalResult,
    RetrievedChunk,
    SystemIssue,
    TestCase,
    TestSet,
)
from ragprobe.core.repair import RepairApplyResult, RepairPlan

__version__ = "1.5.0"

__all__ = [
    "__version__",
    "RAGProbe",
    "run_baseline_retriever",
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
    "PipelineResult",
    "Recommendation",
    "RetrievalResult",
    "RetrievedChunk",
    "SystemIssue",
    "TestCase",
    "TestSet",
]
