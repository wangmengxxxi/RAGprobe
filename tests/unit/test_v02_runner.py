from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ragprobe.core.analyzer import DiagnosticAnalyzer
from ragprobe.core.matching import apply_content_fallback
import sys

from ragprobe.core.runner import run_endpoint, run_retriever_command, run_retriever_script
from ragprobe.io.jsonl import load_results, load_testset


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_run_retriever_script_with_chunk_ids() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")

    results = run_retriever_script(testset, FIXTURES / "retriever_with_ids.py")
    report = DiagnosticAnalyzer().analyze(testset, results)

    assert report.hit_rate == 1.0
    assert report.fpr == 0.0
    assert report.metadata["match_stats"]["chunk_id"] == 2


def test_run_retriever_command_jsonl_protocol() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")

    results = run_retriever_command(
        testset,
        f"{sys.executable} {FIXTURES / 'jsonl_retriever.py'}",
    )
    report = DiagnosticAnalyzer().analyze(testset, results)

    assert report.hit_rate == 1.0
    assert report.fpr == 0.0


def test_content_fallback_matches_missing_chunk_ids() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")
    results = load_results(FIXTURES / "results_without_ids.json")

    matched = apply_content_fallback(testset, results)
    report = DiagnosticAnalyzer().analyze(testset, matched)

    assert report.hit_rate == 1.0
    assert report.metadata["match_stats"]["content_fallback"] == 2


def test_run_retriever_script_without_ids_uses_content_fallback() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")

    results = run_retriever_script(testset, FIXTURES / "retriever_without_ids.py")

    assert [result.retrieved[0].chunk_id for result in results] == [
        "buyer_payment_30",
        "seller_delivery_15",
    ]


def test_score_is_optional_for_retriever_results() -> None:
    testset = load_testset(FIXTURES / "contract_testset.json")

    results = run_retriever_script(testset, FIXTURES / "retriever_without_scores.py")
    report = DiagnosticAnalyzer().analyze(testset, results)

    assert results[0].retrieved[0].score is None
    assert report.hit_rate == 1.0


def test_run_endpoint() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if "买方" in payload["query"]:
                body = [{"chunk_id": "buyer_payment_30", "content": "买方逾期付款条款", "score": 0.95}]
            else:
                body = [{"chunk_id": "seller_delivery_15", "content": "卖方延期交货条款", "score": 0.96}]
            encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        testset = load_testset(FIXTURES / "contract_testset.json")
        results = run_endpoint(testset, f"http://127.0.0.1:{server.server_port}/search")
        report = DiagnosticAnalyzer().analyze(testset, results)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert report.hit_rate == 1.0


def test_run_endpoint_batch_with_headers() -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            assert self.headers["Authorization"] == "Bearer test-token"
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            body = []
            for item in payload["queries"]:
                if "买方" in item["query"]:
                    retrieved = [
                        {"chunk_id": "buyer_payment_30", "content": "买方逾期付款条款", "score": 0.95}
                    ]
                else:
                    retrieved = [
                        {"chunk_id": "seller_delivery_15", "content": "卖方延期交货条款", "score": 0.96}
                    ]
                body.append({"test_case_id": item["id"], "query": item["query"], "retrieved": retrieved})
            encoded = json.dumps({"results": body}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        testset = load_testset(FIXTURES / "contract_testset.json")
        results = run_endpoint(
            testset,
            f"http://127.0.0.1:{server.server_port}/search",
            headers={"Authorization": "Bearer test-token"},
            batch_size=2,
        )
        report = DiagnosticAnalyzer().analyze(testset, results)
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert report.hit_rate == 1.0
