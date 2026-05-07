from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from ragprobe.cli.main import main
from ragprobe.core.generator import (
    add_case,
    generate_testset_from_chunks,
    load_chunks,
    sample_testset,
)
from ragprobe.core.validation import validate_testset
from ragprobe.io.jsonl import load_testset

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp_outputs"


def test_load_chunks_from_jsonl_and_generate_testset() -> None:
    chunks = load_chunks(FIXTURES / "chunks.jsonl")
    testset = generate_testset_from_chunks(chunks, num_cases=2, name="contract-generated")

    assert testset.name == "contract-generated"
    assert len(testset.cases) == 2
    assert testset.metadata["created_from"] == "chunks"
    assert testset.metadata["chunks"]["buyer_payment_30"].startswith("买方逾期付款")
    assert testset.cases[0].expected_chunks == ["buyer_payment_30"]
    assert testset.cases[0].hard_negatives
    assert testset.cases[0].hard_negatives[0].chunk_id in {
        "buyer_payment_notice",
        "seller_delivery_15",
    }


def test_add_case_appends_manual_regression_case() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")

    updated = add_case(
        testset,
        query="主合同解除通知需要提前多少天发出？",
        expected_chunk="main_notice_30",
        hard_negative_ids=["appendix_notice_7"],
        confusion_type="scope_confusion",
        difficulty="hard",
        tags=["production"],
    )

    case = updated.cases[-1]
    assert case.id == "manual_case_001"
    assert case.expected_chunks == ["main_notice_30"]
    assert case.hard_negatives[0].confusion_type == "scope_confusion"
    assert case.metadata["tags"] == ["production"]
    assert "main_notice_30" in updated.metadata["chunks"]


def test_sample_testset_exports_review_rows() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")
    rows = sample_testset(testset, limit=1)

    assert rows == [
        {
            "id": "case_1",
            "query": testset.cases[0].query,
            "expected_chunks": ["buyer_payment_30"],
            "hard_negatives": [
                {
                    "chunk_id": "seller_delivery_15",
                    "confusion_type": "subject_confusion",
                    "similarity_to_correct": 0.94,
                }
            ],
            "difficulty": "hard",
            "source_document": "",
            "review": {"accepted": None, "notes": ""},
        }
    ]


def test_v03_cli_generate_add_case_and_sample() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    generated = OUTPUT_DIR / f"generated-{uuid4().hex}.json"
    added = OUTPUT_DIR / f"added-{uuid4().hex}.json"
    review = OUTPUT_DIR / f"review-{uuid4().hex}.jsonl"

    generate_code = main(
        [
            "generate",
            "--chunks",
            str(FIXTURES / "chunks.jsonl"),
            "--output",
            str(generated),
            "--num-cases",
            "2",
            "--name",
            "cli-generated",
        ]
    )
    add_code = main(
        [
            "add-case",
            "--testset",
            str(generated),
            "--output",
            str(added),
            "--query",
            "买方付款通知需要多久确认？",
            "--expected-chunk",
            "buyer_payment_notice",
            "--hard-negative",
            "buyer_payment_30",
            "--confusion-type",
            "event_confusion",
            "--tag",
            "manual",
        ]
    )
    sample_code = main(
        [
            "sample",
            "--testset",
            str(added),
            "--output",
            str(review),
            "--limit",
            "2",
        ]
    )

    assert generate_code == 0
    assert add_code == 0
    assert sample_code == 0
    assert load_testset(generated).name == "cli-generated"
    assert load_testset(added).cases[-1].metadata["created_from"] == "manual_bad_case"
    lines = [json.loads(line) for line in review.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2
    assert "review" in lines[0]


def test_validate_warns_when_cases_reference_unknown_chunks() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")
    add_case(
        testset,
        query="没有进入 chunks 的条款是什么？",
        expected_chunk="missing_chunk",
        hard_negative_ids=["missing_hard_negative"],
    )
    testset.metadata["chunks"].pop("missing_chunk")
    testset.metadata["chunks"].pop("missing_hard_negative")

    report = validate_testset(testset)

    assert report.valid
    assert any("expected_chunks are not present" in warning for warning in report.warnings)
    assert any("hard_negatives are not present" in warning for warning in report.warnings)
