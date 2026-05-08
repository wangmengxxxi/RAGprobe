from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from ragprobe import RAGProbe
from ragprobe.cli.main import main
from ragprobe.core.baseline import run_baseline_retriever
from ragprobe.core.experiment import run_experiment
from ragprobe.core.models import TestSet
from ragprobe.io.jsonl import load_json, load_results, load_testset

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "contract"
OUTPUT_DIR = ROOT / "tests" / "tmp_outputs"


def test_embedding_baseline_retrieves_from_testset_chunks() -> None:
    testset = load_testset(EXAMPLES / "testset.json")

    results = run_baseline_retriever(testset, "embedding", top_k=2, dimensions=64)

    assert len(results) == len(testset.cases)
    assert results[0].retrieved[0].chunk_id == "buyer_payment_30"
    assert results[0].retrieved[0].metadata["baseline_retriever"] == "embedding"
    assert results[0].retrieved[0].score is not None


def test_baseline_requires_chunk_corpus() -> None:
    with pytest.raises(ValueError, match="metadata.chunks"):
        run_baseline_retriever(TestSet(cases=[]), "embedding")


def test_cli_run_supports_embedding_baseline() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = OUTPUT_DIR / f"baseline-results-{uuid4().hex}.json"

    code = main(
        [
            "run",
            "--testset",
            str(EXAMPLES / "testset.json"),
            "--baseline",
            "embedding",
            "--output",
            str(output),
            "--top-k",
            "2",
        ]
    )

    payload = load_json(output)
    results = load_results(output)
    assert code == 0
    assert payload["metadata"]["schema_version"] == "ragprobe.results.v1"
    assert results[0].retrieved[0].metadata["baseline_retriever"] == "embedding"


def test_python_api_run_supports_lexical_baseline() -> None:
    probe = RAGProbe()

    results = probe.run(
        testset=EXAMPLES / "testset.json",
        baseline="lexical",
        top_k=2,
    )

    assert results[0].retrieved[0].metadata["baseline_retriever"] == "lexical"
    assert results[0].retrieved[0].chunk_id == "buyer_payment_30"


def test_experiment_accepts_builtin_baseline_retriever() -> None:
    output_dir = OUTPUT_DIR / f"baseline-experiment-{uuid4().hex}"

    report = run_experiment(
        {
            "name": "builtin-baseline-experiment",
            "testset": str(EXAMPLES / "testset.json"),
            "baseline": "builtin",
            "retrievers": [
                {
                    "name": "builtin",
                    "baseline": "embedding",
                    "embedding_dimensions": 64,
                },
                {
                    "name": "script",
                    "retriever": str(EXAMPLES / "python_retriever.py"),
                },
            ],
        },
        output_dir=output_dir,
    )

    assert report.baseline == "builtin"
    assert [run.name for run in report.runs] == ["builtin", "script"]
    assert (output_dir / "builtin.results.json").exists()
    assert report.runs[0].results[0].retrieved[0].metadata["baseline_retriever"] == "embedding"
