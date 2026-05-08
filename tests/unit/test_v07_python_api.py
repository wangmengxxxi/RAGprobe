from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ragprobe import RAGProbe
from ragprobe.io.jsonl import load_results, load_testset


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "contract"
OUTPUT_DIR = ROOT / "tests" / "tmp_outputs"


def test_python_api_generate_run_diagnose_check() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    testset_path = OUTPUT_DIR / f"api-generated-{uuid4().hex}.json"
    results_path = OUTPUT_DIR / f"api-results-{uuid4().hex}.json"
    quality_path = OUTPUT_DIR / f"api-quality-{uuid4().hex}.md"

    probe = RAGProbe()
    testset = probe.generate(
        chunks=EXAMPLES / "chunks.jsonl",
        output=testset_path,
        num_cases=2,
        quality_report=quality_path,
    )
    results = probe.run(
        testset=testset,
        retriever=EXAMPLES / "python_retriever.py",
        output=results_path,
    )
    report = probe.diagnose(testset=testset, results=results)
    check = probe.check(report, min_hit_rate=0.5, max_fpr=1.0)

    assert testset.metadata["source"] == "ragprobe-v0.5-quality-generator"
    assert len(load_testset(testset_path).cases) == 2
    assert len(load_results(results_path)) == 2
    assert "RAGProbe Testset Quality Report" in quality_path.read_text(encoding="utf-8")
    assert report.hit_rate >= 0.5
    assert check.passed


def test_python_api_evaluate_with_existing_testset() -> None:
    probe = RAGProbe()

    report = probe.evaluate(
        testset=EXAMPLES / "testset.json",
        retriever=EXAMPLES / "python_retriever.py",
    )

    assert report.hit_rate == 1.0
    assert report.fpr == 0.0


def test_python_api_run_accepts_callable_retriever() -> None:
    testset = load_testset(EXAMPLES / "testset.json")
    probe = RAGProbe()

    def retrieve(query: str, top_k: int = 10) -> list[dict]:
        if "延期交货" in query:
            chunk_id = "seller_delivery_15"
        elif "通知" in query:
            chunk_id = "buyer_payment_notice"
        else:
            chunk_id = "buyer_payment_30"
        return [{"chunk_id": chunk_id, "content": testset.metadata["chunks"][chunk_id]}][:top_k]

    results = probe.run(testset=testset, retriever_fn=retrieve)
    report = probe.diagnose(testset=testset, results=results)

    assert len(results) == len(testset.cases)
    assert report.hit_rate == 1.0


def test_python_api_requires_base_url_for_generic_provider() -> None:
    probe = RAGProbe(llm="openai-compatible")

    try:
        probe.generate(chunks=EXAMPLES / "chunks.jsonl", num_cases=1)
    except ValueError as exc:
        assert "base_url is required" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_python_api_accepts_llm_validation_flag_for_default_generation() -> None:
    probe = RAGProbe()
    testset = probe.generate(
        chunks=EXAMPLES / "chunks.jsonl",
        num_cases=1,
        llm_validate=True,
    )

    assert testset.cases[0].metadata["generator_mode"] == "standard"
