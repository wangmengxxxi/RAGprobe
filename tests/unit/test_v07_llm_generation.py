from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ragprobe.cli.main import main
from ragprobe.core.generator import load_chunks
from ragprobe.core.llm_generation import (
    LLMGenerationConfig,
    LLMJudgeDecision,
    LLMGeneratedCase,
    LLMHardNegativeDecision,
    generate_testset_from_chunks_llm,
    validate_llm_generated_case,
    parse_generated_case,
)
from ragprobe.io.jsonl import load_testset


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tmp_outputs"


class FakeLLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_case(self, target, candidates, config):
        self.calls += 1
        return LLMGeneratedCase(
            query=f"请问{target.metadata.get('topic', target.chunk_id)}应该如何处理？",
            hard_negatives=[
                LLMHardNegativeDecision(
                    chunk_id=candidates[0].chunk_id,
                    accepted=True,
                    confusion_type="subject_confusion",
                    confidence=0.91,
                    reason="主体不同但语义接近。",
                )
            ]
            if candidates
            else [],
        )


class FakeJudgeClient:
    def __init__(self, expected_answerable: bool = True, hard_negative_answerable: bool = False) -> None:
        self.expected_answerable = expected_answerable
        self.hard_negative_answerable = hard_negative_answerable
        self.calls = []

    def judge_answerability(self, query, chunk, role, config):
        self.calls.append((query, chunk.chunk_id, role))
        if role == "expected_chunk":
            return LLMJudgeDecision(
                answerable=self.expected_answerable,
                confidence=0.9,
                reason="expected judgment",
            )
        return LLMJudgeDecision(
            answerable=self.hard_negative_answerable,
            confidence=0.8,
            reason="negative judgment",
        )


def test_parse_generated_case_accepts_json_fence() -> None:
    case = parse_generated_case(
        """
        ```json
        {
          "query": "买方逾期付款超过30天怎么办？",
          "hard_negatives": [
            {
              "chunk_id": "seller_delivery_15",
              "accepted": true,
              "confusion_type": "subject_confusion",
              "confidence": 0.87,
              "reason": "主体不同。"
            }
          ]
        }
        ```
        """
    )

    assert case.query == "买方逾期付款超过30天怎么办？"
    assert case.hard_negatives[0].chunk_id == "seller_delivery_15"
    assert case.hard_negatives[0].confidence == 0.87


def test_llm_generation_uses_client_and_writes_metadata() -> None:
    chunks = load_chunks(FIXTURES / "chunks.jsonl")
    client = FakeLLMClient()
    cache_dir = OUTPUT_DIR / f"v07-cache-{uuid4().hex}"
    OUTPUT_DIR.mkdir(exist_ok=True)

    testset = generate_testset_from_chunks_llm(
        chunks,
        client=client,
        num_cases=2,
        hard_negative_top_k=1,
        cache_dir=cache_dir,
    )

    assert client.calls == 2
    assert testset.metadata["generator_mode"] == "llm"
    assert testset.metadata["llm_provider"] == "openai-compatible"
    assert testset.cases[0].metadata["generator_mode"] == "llm"
    assert testset.cases[0].hard_negatives[0].reason.startswith("主体不同但语义接近。")
    assert "LLM confidence: 0.91" in testset.cases[0].hard_negatives[0].reason


def test_llm_generation_cache_reuses_previous_response() -> None:
    chunks = load_chunks(FIXTURES / "chunks.jsonl")
    first_client = FakeLLMClient()
    second_client = FakeLLMClient()
    cache_dir = OUTPUT_DIR / f"v07-cache-{uuid4().hex}"
    OUTPUT_DIR.mkdir(exist_ok=True)

    first = generate_testset_from_chunks_llm(
        chunks,
        client=first_client,
        num_cases=1,
        cache_dir=cache_dir,
    )
    second = generate_testset_from_chunks_llm(
        chunks,
        client=second_client,
        num_cases=1,
        cache_dir=cache_dir,
    )

    assert first_client.calls == 1
    assert second_client.calls == 0
    assert first.cases[0].query == second.cases[0].query
    assert second.cases[0].metadata["llm_cache_hit"] is True


def test_default_generate_cli_remains_deterministic() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    testset_path = OUTPUT_DIR / f"v07-default-generated-{uuid4().hex}.json"

    code = main(
        [
            "generate",
            "--chunks",
            str(FIXTURES / "chunks.jsonl"),
            "--output",
            str(testset_path),
            "--num-cases",
            "1",
        ]
    )

    testset = load_testset(testset_path)

    assert code == 0
    assert testset.metadata["source"] == "ragprobe-v0.5-quality-generator"
    assert testset.cases[0].metadata["generator_mode"] == "standard"


def test_llm_generation_preserves_provider_config_metadata() -> None:
    chunks = load_chunks(FIXTURES / "chunks.jsonl")
    client = FakeLLMClient()
    cache_dir = OUTPUT_DIR / f"v07-config-cache-{uuid4().hex}"
    OUTPUT_DIR.mkdir(exist_ok=True)
    config = LLMGenerationConfig(
        provider="openai-compatible",
        model="custom-model",
        base_url="http://localhost:8000/v1/chat/completions?token=redacted",
        api_key_env="CUSTOM_API_KEY",
    )

    testset = generate_testset_from_chunks_llm(
        chunks,
        client=client,
        num_cases=1,
        cache_dir=cache_dir,
        config=config,
    )

    assert testset.metadata["llm_provider"] == "openai-compatible"
    assert testset.metadata["llm_model"] == "custom-model"
    assert testset.metadata["llm_base_url"] == "http://localhost:8000/v1/chat/completions"
    assert testset.metadata["llm_api_key_env"] == "CUSTOM_API_KEY"


def test_rule_validation_flags_extractive_and_generic_query() -> None:
    chunks = load_chunks(FIXTURES / "chunks.jsonl")
    generated = LLMGeneratedCase(
        query="买方逾期付款超过30天，应按未付款金额支付违约金。",
        hard_negatives=[],
    )
    validation = validate_llm_generated_case(
        generated=generated,
        expected_chunk=chunks[0],
        hard_negatives=[],
        chunks_by_id={chunk.chunk_id: chunk for chunk in chunks},
    )

    assert "query_too_extractive" in validation.warnings
    assert validation.status == "warning"


def test_llm_judge_removes_answerable_hard_negative() -> None:
    chunks = load_chunks(FIXTURES / "chunks.jsonl")
    client = FakeLLMClient()
    judge = FakeJudgeClient(expected_answerable=True, hard_negative_answerable=True)
    cache_dir = OUTPUT_DIR / f"v07-judge-cache-{uuid4().hex}"
    OUTPUT_DIR.mkdir(exist_ok=True)

    testset = generate_testset_from_chunks_llm(
        chunks,
        client=client,
        judge_client=judge,
        num_cases=1,
        hard_negative_top_k=1,
        cache_dir=cache_dir,
    )

    case = testset.cases[0]
    assert case.hard_negatives == []
    assert case.metadata["validation"]["removed_hard_negatives"]
    assert "hard_negative_answerable" in case.metadata["validation"]["warnings"]
    assert testset.metadata["validation_summary"]["removed_hard_negatives"] == 1


def test_llm_judge_rejected_case_is_dropped_by_default() -> None:
    chunks = load_chunks(FIXTURES / "chunks.jsonl")
    client = FakeLLMClient()
    judge = FakeJudgeClient(expected_answerable=False)
    cache_dir = OUTPUT_DIR / f"v07-reject-cache-{uuid4().hex}"
    OUTPUT_DIR.mkdir(exist_ok=True)

    testset = generate_testset_from_chunks_llm(
        chunks,
        client=client,
        judge_client=judge,
        num_cases=1,
        cache_dir=cache_dir,
    )

    assert testset.cases == []
    assert testset.metadata["quality_summary"]["total_cases"] == 0
