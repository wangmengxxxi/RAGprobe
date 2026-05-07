from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ragprobe.cli.main import main
from ragprobe.io.jsonl import load_results


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp_outputs"


def test_run_cli_with_retriever_script() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = OUTPUT_DIR / f"run-{uuid4().hex}.json"

    code = main(
        [
            "run",
            "--testset",
            str(FIXTURES / "contract_testset.json"),
            "--retriever",
            str(FIXTURES / "retriever_with_ids.py"),
            "--output",
            str(output),
        ]
    )

    assert code == 0
    results = load_results(output)
    assert len(results) == 2
    assert results[0].retrieved[0].chunk_id == "buyer_payment_30"


def test_run_cli_with_retriever_command() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = OUTPUT_DIR / f"run-cmd-{uuid4().hex}.json"

    code = main(
        [
            "run",
            "--testset",
            str(FIXTURES / "contract_testset.json"),
            "--retriever-cmd",
            f"{__import__('sys').executable} {FIXTURES / 'jsonl_retriever.py'}",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    results = load_results(output)
    assert len(results) == 2


def test_export_queries_and_import_results() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    queries = OUTPUT_DIR / f"queries-{uuid4().hex}.jsonl"
    output = OUTPUT_DIR / f"imported-{uuid4().hex}.json"

    export_code = main(
        [
            "export-queries",
            "--testset",
            str(FIXTURES / "contract_testset.json"),
            "--output",
            str(queries),
        ]
    )
    import_code = main(
        [
            "import-results",
            "--queries",
            str(queries),
            "--results",
            str(FIXTURES / "import_results.jsonl"),
            "--output",
            str(output),
        ]
    )

    assert export_code == 0
    assert import_code == 0
    assert len(load_results(output)) == 2


def test_diagnose_cli_uses_content_fallback(capsys) -> None:
    code = main(
        [
            "diagnose",
            "--testset",
            str(FIXTURES / "contract_testset.json"),
            "--results",
            str(FIXTURES / "results_without_ids.json"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "Hit Rate: 1.000" in captured.out
    assert "matched by content fallback" in captured.out


def test_validate_cli_reports_valid_artifacts(capsys) -> None:
    code = main(
        [
            "validate",
            "--testset",
            str(FIXTURES / "contract_testset.json"),
            "--results",
            str(FIXTURES / "results_without_ids.json"),
        ]
    )

    captured = capsys.readouterr()
    assert code == 0
    assert "testset: valid" in captured.out
    assert "results: valid" in captured.out
