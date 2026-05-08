from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ragprobe import RAGProbe
from ragprobe.cli.main import main
from ragprobe.core.experiment import run_experiment
from ragprobe.io.jsonl import load_json, load_results
from ragprobe.reports.markdown import render_experiment_markdown

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples" / "contract"
OUTPUT_DIR = ROOT / "tests" / "tmp_outputs"


def test_run_experiment_compares_multiple_retrievers() -> None:
    output_dir = OUTPUT_DIR / f"experiment-{uuid4().hex}"

    report = run_experiment(EXAMPLES / "experiment.json", output_dir=output_dir)

    assert report.baseline == "weak"
    assert [run.name for run in report.runs] == ["weak", "tuned"]
    assert report.best_by_metric["hit_rate"] == "tuned"
    deltas = {item.metric: item.delta for item in report.comparisons[0].comparison.deltas}
    assert deltas["mrr"] > 0
    assert deltas["fpr"] < 0
    assert (output_dir / "weak.results.json").exists()
    assert (output_dir / "tuned.report.json").exists()
    assert len(load_results(output_dir / "tuned.results.json")) == 3
    assert load_json(output_dir / "experiment_report.json")["baseline"] == "weak"


def test_experiment_cli_writes_markdown_report(capsys) -> None:
    output_dir = OUTPUT_DIR / f"experiment-cli-{uuid4().hex}"

    code = main(
        [
            "experiment",
            "--config",
            str(EXAMPLES / "experiment.json"),
            "--output-dir",
            str(output_dir),
        ]
    )

    captured = capsys.readouterr()
    markdown = (output_dir / "experiment_report.md").read_text(encoding="utf-8")
    assert code == 0
    assert "experiment report:" in captured.out
    assert "# RAGProbe Experiment Report" in markdown
    assert "| tuned |" in markdown


def test_python_api_experiment_accepts_config_dict() -> None:
    output_dir = OUTPUT_DIR / f"experiment-api-{uuid4().hex}"
    probe = RAGProbe()

    report = probe.experiment(
        config={
            "name": "api-experiment",
            "testset": str(EXAMPLES / "testset.json"),
            "baseline": "weak",
            "retrievers": [
                {"name": "weak", "retriever": str(EXAMPLES / "weak_python_retriever.py")},
                {"name": "tuned", "retriever": str(EXAMPLES / "python_retriever.py")},
            ],
        },
        output_dir=output_dir,
    )

    text = render_experiment_markdown(report)
    assert report.name == "api-experiment"
    assert report.comparisons[0].comparison.improved_cases
    assert "Best By Metric" in text
