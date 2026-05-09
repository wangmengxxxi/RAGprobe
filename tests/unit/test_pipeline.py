from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ragprobe import RAGProbe, PipelineResult


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "contract"
OUTPUT_DIR = ROOT / "tests" / "tmp_outputs"


def test_pipeline_with_testset_and_baseline() -> None:
    probe = RAGProbe()
    result = probe.pipeline(
        testset=EXAMPLES / "testset.json",
        baseline="lexical",
    )

    assert isinstance(result, PipelineResult)
    assert result.report.hit_rate >= 0.0
    assert result.check is None
    assert len(result.results) == len(result.testset.cases)


def test_pipeline_with_chunks_and_retriever() -> None:
    probe = RAGProbe()
    result = probe.pipeline(
        chunks=EXAMPLES / "chunks.jsonl",
        retriever=EXAMPLES / "python_retriever.py",
        num_cases=2,
        top_k=5,
    )

    assert isinstance(result, PipelineResult)
    assert len(result.testset.cases) == 2
    assert len(result.results) == 2


def test_pipeline_with_ci_check() -> None:
    probe = RAGProbe()
    result = probe.pipeline(
        testset=EXAMPLES / "testset.json",
        retriever=EXAMPLES / "python_retriever.py",
        min_hit_rate=0.5,
        max_fpr=1.0,
    )

    assert result.check is not None
    assert result.check.passed


def test_pipeline_with_output_dir() -> None:
    probe = RAGProbe()
    OUTPUT_DIR.mkdir(exist_ok=True)
    out = OUTPUT_DIR / f"pipeline-out-{uuid4().hex}"
    result = probe.pipeline(
        testset=EXAMPLES / "testset.json",
        retriever=EXAMPLES / "python_retriever.py",
        output_dir=out,
    )

    assert (out / "testset.json").exists()
    assert (out / "results.json").exists()
    assert (out / "report.json").exists()
    assert (out / "report.md").exists()


def test_pipeline_with_source_object() -> None:
    from ragprobe.core.generator import load_chunks

    chunks = load_chunks(EXAMPLES / "chunks.jsonl")

    class FakeSource:
        def export(self):
            return chunks

    probe = RAGProbe()
    result = probe.pipeline(
        source=FakeSource(),
        baseline="lexical",
        num_cases=2,
    )

    assert isinstance(result, PipelineResult)
    assert len(result.testset.cases) == 2


def test_pipeline_requires_data_source() -> None:
    probe = RAGProbe()
    try:
        probe.pipeline(baseline="lexical")
        assert False, "should have raised"
    except ValueError as e:
        assert "testset, chunks, or source" in str(e)


def test_run_with_headers_no_config_file(monkeypatch) -> None:
    """run() accepts headers directly without endpoint_config file."""
    from ragprobe.io.jsonl import load_testset

    captured = {}

    def fake_run_endpoint(*args, **kwargs):
        captured["headers"] = kwargs["headers"]
        captured["timeout"] = kwargs["timeout"]
        captured["batch_size"] = kwargs["batch_size"]
        return []

    monkeypatch.setattr("ragprobe.api.run_endpoint", fake_run_endpoint)

    probe = RAGProbe()
    testset = load_testset(EXAMPLES / "testset.json")

    probe.run(
        testset=testset,
        endpoint="http://localhost:99999/nonexistent",
        headers={"Authorization": "Bearer test"},
        timeout=0.1,
        batch_size=2,
    )

    assert captured == {
        "headers": {"Authorization": "Bearer test"},
        "timeout": 0.1,
        "batch_size": 2,
    }
