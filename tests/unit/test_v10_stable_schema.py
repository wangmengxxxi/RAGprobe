from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from ragprobe.cli.main import main
from ragprobe.core.audit import audit_testset
from ragprobe.core.llm_generation import LLMGenerationConfig, LLMJudgeDecision
from ragprobe.core.repair import build_repair_plan
from ragprobe.io.jsonl import load_json, load_results, load_testset

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "contract"
OUTPUT_DIR = ROOT / "tests" / "tmp_outputs"


class PassingJudge:
    def judge_answerability(self, query, chunk, role, config):
        return LLMJudgeDecision(
            answerable=role == "expected_chunk",
            confidence=0.9,
            reason="fixture decision",
        )


def test_version_is_1_0_0(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--version"])

    captured = capsys.readouterr()
    assert exc.value.code == 0
    assert "ragprobe 1.0.0" in captured.out


def test_generated_testset_and_results_include_schema_metadata() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    testset_path = OUTPUT_DIR / f"schema-testset-{uuid4().hex}.json"
    results_path = OUTPUT_DIR / f"schema-results-{uuid4().hex}.json"

    generate_code = main(
        [
            "generate",
            "--chunks",
            str(EXAMPLES / "chunks.jsonl"),
            "--output",
            str(testset_path),
            "--num-cases",
            "1",
        ]
    )
    run_code = main(
        [
            "run",
            "--testset",
            str(EXAMPLES / "testset.json"),
            "--retriever",
            str(EXAMPLES / "python_retriever.py"),
            "--output",
            str(results_path),
        ]
    )

    assert generate_code == 0
    assert run_code == 0
    testset = load_testset(testset_path)
    results_payload = load_json(results_path)
    assert testset.metadata["schema_version"] == "ragprobe.testset.v1"
    assert testset.metadata["ragprobe_version"] == "1.0.0"
    assert results_payload["metadata"]["schema_version"] == "ragprobe.results.v1"
    assert len(load_results(results_path)) == len(load_testset(EXAMPLES / "testset.json").cases)


def test_reports_and_repair_plans_include_schema_metadata() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    results_path = OUTPUT_DIR / f"schema-report-results-{uuid4().hex}.json"
    report_path = OUTPUT_DIR / f"schema-diagnostic-{uuid4().hex}.json"
    compare_path = OUTPUT_DIR / f"schema-compare-{uuid4().hex}.json"

    assert (
        main(
            [
                "run",
                "--testset",
                str(EXAMPLES / "testset.json"),
                "--retriever",
                str(EXAMPLES / "python_retriever.py"),
                "--output",
                str(results_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "diagnose",
                "--testset",
                str(EXAMPLES / "testset.json"),
                "--results",
                str(results_path),
                "--format",
                "json",
                "--output",
                str(report_path),
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "compare",
                "--testset",
                str(EXAMPLES / "testset.json"),
                "--before",
                str(results_path),
                "--after",
                str(results_path),
                "--format",
                "json",
                "--output",
                str(compare_path),
            ]
        )
        == 0
    )

    diagnostic = load_json(report_path)
    comparison = load_json(compare_path)
    audit = audit_testset(
        EXAMPLES / "testset.json",
        judge_client=PassingJudge(),
        config=LLMGenerationConfig(provider="fake", model="fake"),
        sample_size=1,
        cache_dir=None,
    )
    repair_plan = build_repair_plan(
        {
            "findings": [
                {
                    "case_id": "contract_case_1",
                    "warnings": ["hard_negative_answerable"],
                    "hard_negative_findings": [
                        {"chunk_id": "seller_delivery_15", "answerable": True}
                    ],
                }
            ],
            "summary": {},
            "metadata": {},
        }
    )

    assert diagnostic["metadata"]["schema_version"] == "ragprobe.diagnostic_report.v1"
    assert comparison["metadata"]["schema_version"] == "ragprobe.comparison_report.v1"
    assert audit.metadata["schema_version"] == "ragprobe.audit_report.v1"
    assert repair_plan.metadata["schema_version"] == "ragprobe.repair_plan.v1"
