import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from ragprobe.cli.main import main


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp_outputs"


def test_diagnose_cli_writes_markdown() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = OUTPUT_DIR / f"report-{uuid4().hex}.md"

    code = main(
        [
            "diagnose",
            "--testset",
            str(FIXTURES / "contract_testset.json"),
            "--results",
            str(FIXTURES / "results_v1.json"),
            "--format",
            "markdown",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert "RAGProbe Diagnostic Report" in output.read_text(encoding="utf-8")


def test_compare_cli_runs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    output = OUTPUT_DIR / f"compare-{uuid4().hex}.md"

    code = main(
        [
            "compare",
            "--testset",
            str(FIXTURES / "contract_testset.json"),
            "--before",
            str(FIXTURES / "results_v1.json"),
            "--after",
            str(FIXTURES / "results_v2.json"),
            "--format",
            "markdown",
            "--output",
            str(output),
        ]
    )

    assert code == 0
    assert "hit_rate: 0.500 -> 1.000" in output.read_text(encoding="utf-8")


def test_check_cli_returns_nonzero_on_threshold_failure() -> None:
    code = main(
        [
            "check",
            "--testset",
            str(FIXTURES / "contract_testset.json"),
            "--results",
            str(FIXTURES / "results_v1.json"),
            "--min-hit-rate",
            "0.7",
            "--max-fpr",
            "0.3",
        ]
    )

    assert code == 1


def test_demo_cli_runs() -> None:
    assert main(["demo", "--format", "json"]) == 0


def test_python_module_entrypoint_runs() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ragprobe",
            "compare",
            "--testset",
            str(FIXTURES / "contract_testset.json"),
            "--before",
            str(FIXTURES / "results_v1.json"),
            "--after",
            str(FIXTURES / "results_v2.json"),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "RAGProbe Compare Report" in completed.stdout
