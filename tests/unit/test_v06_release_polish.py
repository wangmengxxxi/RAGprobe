from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ragprobe.cli.main import main
from ragprobe.io.jsonl import load_results, load_testset


ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "contract"
OUTPUT_DIR = ROOT / "tests" / "tmp_outputs"


def test_contract_python_example_runs_and_passes_check() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    results_path = OUTPUT_DIR / f"v06-contract-results-{uuid4().hex}.json"

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
    check_code = main(
        [
            "check",
            "--testset",
            str(EXAMPLES / "testset.json"),
            "--results",
            str(results_path),
            "--min-hit-rate",
            "0.7",
            "--min-mrr",
            "0.5",
            "--max-fpr",
            "0.3",
        ]
    )

    assert run_code == 0
    assert check_code == 0
    assert len(load_results(results_path)) == len(load_testset(EXAMPLES / "testset.json").cases)


def test_v06_docs_exist() -> None:
    assert (ROOT / "README.md").read_text(encoding="utf-8")
    assert (ROOT / "docs" / "schemas.md").read_text(encoding="utf-8")
    assert (ROOT / ".github" / "workflows" / "ragprobe.yml").read_text(encoding="utf-8")
