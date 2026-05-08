from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ragprobe import RAGProbe
from ragprobe.cli.main import main
from ragprobe.core.audit import audit_testset
from ragprobe.core.llm_generation import LLMGenerationConfig, LLMJudgeDecision
from ragprobe.io.jsonl import load_json
from ragprobe.reports.markdown import render_audit_markdown

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "contract"
OUTPUT_DIR = ROOT / "tests" / "tmp_outputs"


class FakeAuditJudge:
    def __init__(self, *, answerable_hard_negative: bool = False) -> None:
        self.answerable_hard_negative = answerable_hard_negative
        self.calls: list[tuple[str, str]] = []

    def judge_answerability(self, query, chunk, role, config):
        self.calls.append((role, chunk.chunk_id))
        if role == "expected_chunk":
            return LLMJudgeDecision(
                answerable=True,
                confidence=0.95,
                reason="Expected chunk contains the answer.",
            )
        return LLMJudgeDecision(
            answerable=self.answerable_hard_negative,
            confidence=0.9,
            reason=(
                "Hard negative can answer the query."
                if self.answerable_hard_negative
                else "Hard negative is related but not sufficient."
            ),
        )


def test_audit_testset_passes_clean_cases() -> None:
    judge = FakeAuditJudge()

    report = audit_testset(
        EXAMPLES / "testset.json",
        judge_client=judge,
        config=LLMGenerationConfig(provider="fake", model="fake"),
        sample_size=1,
        cache_dir=None,
    )

    assert report.audited_cases == 1
    assert report.summary["passed"] == 1
    assert report.findings[0].recommended_action == "keep"
    assert judge.calls


def test_audit_marks_answerable_hard_negative_suspicious() -> None:
    judge = FakeAuditJudge(answerable_hard_negative=True)

    report = audit_testset(
        EXAMPLES / "testset.json",
        judge_client=judge,
        config=LLMGenerationConfig(provider="fake", model="fake"),
        sample_size=1,
        cache_dir=None,
    )

    finding = report.findings[0]
    assert finding.status == "suspicious"
    assert "hard_negative_answerable" in finding.warnings
    assert finding.recommended_action == "remove_hard_negative_or_review_query"


def test_python_api_audit_writes_outputs() -> None:
    output = OUTPUT_DIR / f"audit-{uuid4().hex}.json"
    markdown = OUTPUT_DIR / f"audit-{uuid4().hex}.md"
    probe = RAGProbe()

    report = probe.audit(
        testset=EXAMPLES / "testset.json",
        output=output,
        markdown=markdown,
        judge_client=FakeAuditJudge(),
        sample_size=1,
        use_cache=False,
    )

    assert report.summary["passed"] == 1
    assert load_json(output)["summary"]["passed"] == 1
    assert "RAGProbe Testset Audit Report" in markdown.read_text(encoding="utf-8")


def test_audit_cli_writes_json_and_markdown(monkeypatch, capsys) -> None:
    output = OUTPUT_DIR / f"audit-cli-{uuid4().hex}.json"
    markdown = OUTPUT_DIR / f"audit-cli-{uuid4().hex}.md"

    class FakeClient(FakeAuditJudge):
        @classmethod
        def from_env(cls, **kwargs):
            return cls()

    monkeypatch.setattr("ragprobe.cli.main.QwenClient", FakeClient)
    code = main(
        [
            "audit",
            "--testset",
            str(EXAMPLES / "testset.json"),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
            "--llm",
            "qwen",
            "--sample-size",
            "1",
            "--no-cache",
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "audit summary:" in captured.out
    assert load_json(output)["audited_cases"] == 1
    assert "# RAGProbe Testset Audit Report" in markdown.read_text(encoding="utf-8")


def test_render_audit_markdown_includes_recommended_action() -> None:
    report = audit_testset(
        EXAMPLES / "testset.json",
        judge_client=FakeAuditJudge(answerable_hard_negative=True),
        config=LLMGenerationConfig(provider="fake", model="fake"),
        sample_size=1,
        cache_dir=None,
    )

    text = render_audit_markdown(report)
    assert "remove_hard_negative_or_review_query" in text
