from ragprobe import DiagnosticReport, HardNegative
from ragprobe import TestCase as RAGTestCase
from ragprobe import TestSet as RAGTestSet
from ragprobe.cli.main import build_parser


def test_public_models_are_importable() -> None:
    case = RAGTestCase(
        id="case-1",
        query="买方逾期付款超过30天的违约金是多少？",
        expected_chunks=["chunk-buyer-payment-30"],
        hard_negatives=[
            HardNegative(
                chunk_id="chunk-seller-delivery-15",
                confusion_type="subject_confusion",
            )
        ],
    )
    testset = RAGTestSet(cases=[case], name="contract-demo")
    report = DiagnosticReport(hit_rate=0.0)

    assert testset.cases[0].id == "case-1"
    assert report.hit_rate == 0.0


def test_phase0_cli_surface_exists() -> None:
    parser = build_parser()
    subcommands = parser._subparsers._group_actions[0].choices

    assert {
        "demo",
        "generate",
        "add-case",
        "sample",
        "run",
        "validate",
        "export-queries",
        "import-results",
        "diagnose",
        "compare",
        "check",
    }.issubset(subcommands)
