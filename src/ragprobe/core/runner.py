"""Run user retrievers against a testset."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragprobe.core.matching import apply_content_fallback
from ragprobe.core.models import RetrievalResult, RetrievedChunk, TestSet
from ragprobe.core.validation import validate_results


RetrieverFn = Callable[[str, int], list[dict[str, Any]]]


class RetrieverLoadError(ValueError):
    """Raised when a retriever script cannot be loaded."""


@dataclass
class EndpointConfig:
    headers: dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    batch_size: int = 1


def run_retriever_script(
    testset: TestSet,
    retriever_path: str | Path,
    top_k: int = 10,
    content_fallback_threshold: float = 0.9,
) -> list[RetrievalResult]:
    retrieve = load_retriever_function(retriever_path)
    return run_retriever(
        testset,
        retrieve,
        top_k=top_k,
        content_fallback_threshold=content_fallback_threshold,
    )


def run_retriever(
    testset: TestSet,
    retrieve: RetrieverFn,
    top_k: int = 10,
    content_fallback_threshold: float = 0.9,
) -> list[RetrievalResult]:
    results = []
    for case in testset.cases:
        raw_chunks = retrieve(case.query, top_k)
        results.append(
            RetrievalResult(
                test_case_id=case.id,
                query=case.query,
                retrieved=[_coerce_chunk(item) for item in raw_chunks[:top_k]],
            )
        )
    matched = apply_content_fallback(testset, results, threshold=content_fallback_threshold)
    validate_results(testset, matched)
    return matched


def run_retriever_command(
    testset: TestSet,
    command: str,
    top_k: int = 10,
    timeout: float = 30.0,
    content_fallback_threshold: float = 0.9,
) -> list[RetrievalResult]:
    """Run a language-agnostic JSONL subprocess retriever."""
    process = subprocess.Popen(
        shlex.split(command, posix=(os.name != "nt")),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    try:
        results = []
        for case in testset.cases:
            assert process.stdin is not None
            assert process.stdout is not None
            request = {"query": case.query, "top_k": top_k}
            process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
            process.stdin.flush()
            line = _readline_with_timeout(process, timeout)
            raw_chunks = json.loads(line)
            results.append(
                RetrievalResult(
                    test_case_id=case.id,
                    query=case.query,
                    retrieved=[_coerce_chunk(item) for item in raw_chunks[:top_k]],
                )
            )
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait(timeout=timeout)
        if return_code != 0:
            stderr = process.stderr.read() if process.stderr else ""
            raise RuntimeError(f"retriever command exited with {return_code}: {stderr.strip()}")
    except Exception:
        process.kill()
        raise

    matched = apply_content_fallback(testset, results, threshold=content_fallback_threshold)
    validate_results(testset, matched)
    return matched


def run_endpoint(
    testset: TestSet,
    endpoint: str,
    top_k: int = 10,
    timeout: float = 30.0,
    headers: dict[str, str] | None = None,
    batch_size: int = 1,
    content_fallback_threshold: float = 0.9,
) -> list[RetrievalResult]:
    headers = headers or {}
    results = []
    if batch_size > 1:
        results = _run_endpoint_batch(testset, endpoint, top_k, timeout, headers, batch_size)
        matched = apply_content_fallback(testset, results, threshold=content_fallback_threshold)
        validate_results(testset, matched)
        return matched

    for case in testset.cases:
        payload = json.dumps({"query": case.query, "top_k": top_k}).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_chunks = json.loads(response.read().decode("utf-8"))
        results.append(
            RetrievalResult(
                test_case_id=case.id,
                query=case.query,
                retrieved=[_coerce_chunk(item) for item in raw_chunks[:top_k]],
            )
        )
    matched = apply_content_fallback(testset, results, threshold=content_fallback_threshold)
    validate_results(testset, matched)
    return matched


def load_retriever_function(path: str | Path) -> RetrieverFn:
    retriever_path = Path(path)
    spec = importlib.util.spec_from_file_location("ragprobe_user_retriever", retriever_path)
    if spec is None or spec.loader is None:
        raise RetrieverLoadError(f"cannot load retriever script: {retriever_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    retrieve = getattr(module, "retrieve", None)
    if retrieve is None or not callable(retrieve):
        raise RetrieverLoadError(f"{retriever_path} must expose a callable retrieve(query, top_k)")
    return retrieve


def _coerce_chunk(item: Any) -> RetrievedChunk:
    if not isinstance(item, dict):
        raise ValueError("retriever results must be dictionaries")
    return RetrievedChunk(
        content=str(item.get("content", "")),
        score=float(item["score"]) if "score" in item and item["score"] is not None else None,
        metadata=dict(item.get("metadata", {})),
        chunk_id=item.get("chunk_id"),
    )


def load_endpoint_config(path: str | Path | None) -> EndpointConfig:
    if path is None:
        return EndpointConfig()
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return EndpointConfig(
        headers={str(key): str(value) for key, value in data.get("headers", {}).items()},
        timeout=float(data.get("timeout", 30.0)),
        batch_size=int(data.get("batch_size", 1)),
    )


def _run_endpoint_batch(
    testset: TestSet,
    endpoint: str,
    top_k: int,
    timeout: float,
    headers: dict[str, str],
    batch_size: int,
) -> list[RetrievalResult]:
    results = []
    cases = testset.cases
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        payload = json.dumps(
            {"queries": [{"id": case.id, "query": case.query} for case in batch], "top_k": top_k}
        ).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json", **headers},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw_payload = json.loads(response.read().decode("utf-8"))
        raw_results = raw_payload.get("results", raw_payload)
        for item in raw_results:
            results.append(
                RetrievalResult(
                    test_case_id=item["test_case_id"],
                    query=item.get("query", ""),
                    retrieved=[_coerce_chunk(chunk) for chunk in item.get("retrieved", [])[:top_k]],
                )
            )
    return results


def _readline_with_timeout(process: subprocess.Popen[str], timeout: float) -> str:
    # Windows does not support select() on pipes; communicate with a reader thread.
    import queue
    import threading

    assert process.stdout is not None
    output: queue.Queue[str] = queue.Queue(maxsize=1)

    def read() -> None:
        output.put(process.stdout.readline())

    thread = threading.Thread(target=read, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise TimeoutError(f"retriever command did not respond within {timeout} seconds")
    line = output.get()
    if not line:
        stderr = process.stderr.read() if process.stderr else ""
        raise RuntimeError(f"retriever command closed stdout unexpectedly: {stderr.strip()}")
    return line
