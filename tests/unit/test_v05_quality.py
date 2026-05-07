from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ragprobe.cli.main import main
from ragprobe.core.generator import (
    DocumentChunk,
    generate_testset_from_chunks,
    load_chunks,
    mine_hard_negatives,
    render_quality_report,
)
from ragprobe.core.validation import validate_testset
from ragprobe.io.jsonl import load_testset

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp_outputs"


def test_hybrid_hard_negative_mining_uses_metadata_and_adjacency() -> None:
    chunks = [
        DocumentChunk(
            chunk_id="a",
            content="Buyer payment notice must be confirmed within 10 days.",
            metadata={"section": "payment", "subject": "buyer"},
        ),
        DocumentChunk(
            chunk_id="b",
            content="Seller delivery notice must be confirmed within 10 days.",
            metadata={"section": "payment", "subject": "seller"},
        ),
        DocumentChunk(
            chunk_id="c",
            content="Public disclosure rules are listed separately.",
            metadata={"section": "disclosure", "subject": "public"},
        ),
    ]

    candidates = mine_hard_negatives(chunks[0], chunks, top_k=1, strategy="hybrid")

    assert candidates[0].chunk.chunk_id == "b"
    assert "metadata" in candidates[0].signals
    assert "same_section" in candidates[0].signals
    assert candidates[0].similarity > 0.2


def test_generated_testset_contains_quality_metadata_and_summary() -> None:
    chunks = load_chunks(FIXTURES / "chunks.jsonl")
    testset = generate_testset_from_chunks(
        chunks,
        num_cases=3,
        hard_negative_top_k=2,
        hn_strategy="hybrid",
    )

    assert testset.metadata["hard_negative_strategy"] == "hybrid"
    assert testset.metadata["quality_summary"]["total_cases"] == 3
    assert testset.metadata["quality_summary"]["hard_negative_coverage"] > 0
    assert "average_quality_score" in testset.metadata["quality_summary"]
    assert testset.cases[0].metadata["quality"]["filter_passed"] is True
    assert testset.cases[0].metadata["hard_negative_strategy"] == "hybrid"


def test_quality_report_renders_summary_sections() -> None:
    chunks = load_chunks(FIXTURES / "chunks.jsonl")
    testset = generate_testset_from_chunks(chunks, num_cases=2)

    report = render_quality_report(testset)

    assert "RAGProbe Testset Quality Report" in report
    assert "Hard negative coverage" in report
    assert "Difficulty Distribution" in report
    assert "Warnings" in report


def test_generate_cli_writes_quality_report() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    testset_path = OUTPUT_DIR / f"v05-generated-{uuid4().hex}.json"
    report_path = OUTPUT_DIR / f"v05-quality-{uuid4().hex}.md"

    code = main(
        [
            "generate",
            "--chunks",
            str(FIXTURES / "chunks.jsonl"),
            "--output",
            str(testset_path),
            "--quality-report",
            str(report_path),
            "--hard-negative-top-k",
            "2",
        ]
    )

    assert code == 0
    assert load_testset(testset_path).metadata["quality_summary"]["total_cases"] == 3
    assert "RAGProbe Testset Quality Report" in report_path.read_text(encoding="utf-8")


def test_validate_reports_generated_quality_warnings() -> None:
    chunks = [
        DocumentChunk(chunk_id="only", content="Only one chunk.", metadata={"topic": "single"})
    ]
    testset = generate_testset_from_chunks(chunks)

    report = validate_testset(testset)

    assert report.valid
    assert any("hard negative coverage" in warning for warning in report.warnings)
    assert any("missing_hard_negative" in warning for warning in report.warnings)
