from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ragprobe import RAGProbe
from ragprobe.cli.main import main
from ragprobe.core.repair import apply_repair_plan, build_repair_plan
from ragprobe.io.jsonl import load_json, load_testset

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "contract"
OUTPUT_DIR = ROOT / "tests" / "tmp_outputs"


def _audit_payload():
    return {
        "testset_name": "contract-demo",
        "total_cases": 3,
        "audited_cases": 2,
        "summary": {"suspicious": 1, "failed": 1},
        "findings": [
            {
                "case_id": "contract_case_1",
                "query": "q1",
                "status": "suspicious",
                "warnings": ["hard_negative_answerable"],
                "hard_negative_findings": [
                    {
                        "chunk_id": "seller_delivery_15",
                        "answerable": True,
                        "risk": "high",
                    }
                ],
            },
            {
                "case_id": "contract_case_2",
                "query": "q2",
                "status": "failed",
                "warnings": ["expected_chunk_not_answerable"],
                "hard_negative_findings": [],
            },
        ],
        "metadata": {"source": "test-audit"},
    }


def test_build_repair_plan_from_audit_report() -> None:
    plan = build_repair_plan(_audit_payload())

    actions = {(item.action, item.case_id, item.chunk_id) for item in plan.actions}
    assert ("remove_hard_negative", "contract_case_1", "seller_delivery_15") in actions
    assert ("reject_case", "contract_case_2", "") in actions
    assert plan.summary["total_actions"] == 2
    assert plan.summary["applicable_actions"] == 2


def test_apply_repair_plan_removes_hard_negative_but_keeps_rejected_case_by_default() -> None:
    testset = load_testset(EXAMPLES / "testset.json")
    plan = build_repair_plan(_audit_payload())

    result = apply_repair_plan(testset, plan)
    case_1 = next(case for case in result.testset.cases if case.id == "contract_case_1")

    assert "seller_delivery_15" not in [item.chunk_id for item in case_1.hard_negatives]
    assert any(case.id == "contract_case_2" for case in result.testset.cases)
    assert result.summary["applied_actions"] == 1
    assert result.summary["skipped_actions"] == 1


def test_apply_repair_plan_can_reject_cases_when_explicitly_allowed() -> None:
    plan = build_repair_plan(_audit_payload())

    result = apply_repair_plan(
        EXAMPLES / "testset.json",
        plan,
        allow_reject_cases=True,
    )

    assert not any(case.id == "contract_case_2" for case in result.testset.cases)
    assert result.summary["rejected_cases"] == 1
    assert result.summary["applied_actions"] == 2


def test_repair_plan_and_apply_cli(capsys) -> None:
    run_id = uuid4().hex
    audit_path = OUTPUT_DIR / f"audit-{run_id}.json"
    plan_path = OUTPUT_DIR / f"repair-plan-{run_id}.json"
    plan_md = OUTPUT_DIR / f"repair-plan-{run_id}.md"
    fixed_path = OUTPUT_DIR / f"fixed-testset-{run_id}.json"
    apply_md = OUTPUT_DIR / f"apply-{run_id}.md"
    OUTPUT_DIR.mkdir(exist_ok=True)
    audit_path.write_text(
        __import__("json").dumps(_audit_payload(), ensure_ascii=False),
        encoding="utf-8",
    )

    plan_code = main(
        [
            "repair-plan",
            "--audit-report",
            str(audit_path),
            "--output",
            str(plan_path),
            "--markdown",
            str(plan_md),
        ]
    )
    apply_code = main(
        [
            "apply-audit-fixes",
            "--testset",
            str(EXAMPLES / "testset.json"),
            "--repair-plan",
            str(plan_path),
            "--output",
            str(fixed_path),
            "--report",
            str(apply_md),
        ]
    )

    captured = capsys.readouterr()
    assert plan_code == 0
    assert apply_code == 0
    assert "repair plan:" in captured.out
    assert "applied audit fixes:" in captured.out
    assert load_json(plan_path)["summary"]["total_actions"] == 2
    assert len(load_testset(fixed_path).cases) == 3
    assert "RAGProbe Audit Repair Plan" in plan_md.read_text(encoding="utf-8")
    assert "RAGProbe Audit Repair Apply Report" in apply_md.read_text(encoding="utf-8")


def test_python_api_repair_plan_and_apply() -> None:
    output = OUTPUT_DIR / f"repair-plan-{uuid4().hex}.json"
    markdown = OUTPUT_DIR / f"repair-plan-{uuid4().hex}.md"
    fixed = OUTPUT_DIR / f"fixed-testset-{uuid4().hex}.json"
    probe = RAGProbe()

    plan = probe.repair_plan(
        audit_report=_audit_payload(),
        output=output,
        markdown=markdown,
    )
    result = probe.apply_audit_fixes(
        testset=EXAMPLES / "testset.json",
        repair_plan=plan,
        output=fixed,
    )

    assert output.exists()
    assert fixed.exists()
    assert plan.summary["total_actions"] == 2
    assert result.summary["applied_actions"] == 1
