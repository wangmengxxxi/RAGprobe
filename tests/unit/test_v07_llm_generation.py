from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from ragprobe.cli.main import main
from ragprobe.core.generator import load_chunks
from ragprobe.core.llm_generation import (
    LLMGenerationConfig,
    LLMGeneratedCase,
    LLMHardNegativeDecision,
    generate_testset_from_chunks_llm,
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
