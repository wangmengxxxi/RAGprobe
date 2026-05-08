from __future__ import annotations

from ragprobe.core.analyzer import DiagnosticAnalyzer
from ragprobe.core.generator import DocumentChunk, infer_confusion_type, metadata_similarity
from ragprobe.core.issues import detect_system_issues
from ragprobe.core.models import HardNegative, RetrievalResult, RetrievedChunk, TestCase, TestSet


def test_infer_confusion_type_uses_domain_metadata_keys() -> None:
    target = DocumentChunk(
        chunk_id="p1",
        content="华为 Mate 手机支持 66W 快充。",
        metadata={"brand": "华为", "category": "phone", "url": "https://example.com/a"},
    )
    candidate = DocumentChunk(
        chunk_id="p2",
        content="小米 Mate 手机支持 66W 快充。",
        metadata={"brand": "小米", "category": "phone", "url": "https://example.com/b"},
    )

    assert infer_confusion_type(target, candidate) == "brand_confusion"


def test_infer_confusion_type_ignores_non_semantic_metadata_noise() -> None:
    target = DocumentChunk(
        chunk_id="a",
        content="同一条内容 100 元。",
        metadata={"url": "https://example.com/a", "page": 1, "updated_at": "2026-01-01"},
    )
    candidate = DocumentChunk(
        chunk_id="b",
        content="同一条内容 200 元。",
        metadata={"url": "https://example.com/b", "page": 2, "updated_at": "2026-02-01"},
    )

    assert infer_confusion_type(target, candidate) == "numeric_confusion"


def test_metadata_similarity_scores_non_legal_domain_keys() -> None:
    left = DocumentChunk(
        chunk_id="left",
        content="退货流程",
        metadata={"product": "laptop", "department": "support", "url": "a"},
    )
    right = DocumentChunk(
        chunk_id="right",
        content="换货流程",
        metadata={"product": "laptop", "department": "sales", "url": "b"},
    )

    assert metadata_similarity(left, right) == 0.5


def test_domain_metadata_confusion_triggers_filter_issue() -> None:
    issues = detect_system_issues(
        hit_rate=0.9,
        mrr=0.9,
        fpr=0.4,
        failure_patterns=[],
        confusion_distribution={"brand_confusion": 0.75, "semantic_only": 0.25},
        match_stats={"chunk_id": 4},
        testset=TestSet(
            cases=[
                TestCase(
                    id="case_1",
                    query="华为手机支持什么快充？",
                    expected_chunks=["p1"],
                    hard_negatives=[
                        HardNegative(chunk_id="p2", confusion_type="brand_confusion")
                    ],
                )
            ],
            metadata={"chunks": {"p1": "华为手机支持 66W 快充。", "p2": "小米手机支持 66W 快充。"}},
        ),
    )

    assert any(issue.issue_type == "metadata_filter_needed" for issue in issues)
    assert any("brand_confusion" in issue.evidence for issue in issues)


def test_analyzer_reports_domain_specific_confusion_distribution() -> None:
    testset = TestSet(
        cases=[
            TestCase(
                id="case_1",
                query="华为手机支持什么快充？",
                expected_chunks=["p1"],
                hard_negatives=[HardNegative(chunk_id="p2", confusion_type="brand_confusion")],
            )
        ],
        metadata={"chunks": {"p1": "华为手机支持 66W 快充。", "p2": "小米手机支持 66W 快充。"}},
    )
    results = [
        RetrievalResult(
            test_case_id="case_1",
            query="华为手机支持什么快充？",
            retrieved=[
                RetrievedChunk(chunk_id="p2", content="小米手机支持 66W 快充。", score=0.9),
                RetrievedChunk(chunk_id="p1", content="华为手机支持 66W 快充。", score=0.8),
            ],
        )
    ]

    report = DiagnosticAnalyzer().analyze(testset, results)

    assert report.confusion_distribution == {"brand_confusion": 1.0}
    assert any(issue.issue_type == "metadata_filter_needed" for issue in report.system_issues)
