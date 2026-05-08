"""Python API facade for RAGProbe workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ragprobe.core.analyzer import DiagnosticAnalyzer
from ragprobe.core.audit import AuditReport, audit_testset, save_audit_report
from ragprobe.core.checks import CheckResult, check_thresholds
from ragprobe.core.compare import compare_reports
from ragprobe.core.experiment import ExperimentReport, run_experiment
from ragprobe.core.generator import (
    generate_testset_from_chunks,
    load_chunks,
    render_quality_report,
)
from ragprobe.core.llm_generation import (
    DEFAULT_QWEN_BASE_URL,
    DEFAULT_QWEN_MODEL,
    LLMGenerationConfig,
    LLMGenerationError,
    OpenAICompatibleClient,
    QwenClient,
    generate_testset_from_chunks_llm,
)
from ragprobe.core.matching import apply_content_fallback
from ragprobe.core.models import ComparisonReport, DiagnosticReport, RetrievalResult, TestSet
from ragprobe.core.repair import (
    RepairApplyResult,
    RepairPlan,
    apply_repair_plan,
    build_repair_plan,
    save_repair_plan,
)
from ragprobe.core.runner import (
    load_endpoint_config,
    run_endpoint,
    run_retriever,
    run_retriever_command,
    run_retriever_script,
)
from ragprobe.io.jsonl import load_results, load_testset, save_json


class RAGProbe:
    """Convenience wrapper around the core RAGProbe workflow.

    The CLI remains the best interface for shell scripts and CI. This facade is
    intended for Python applications and notebooks that want to call RAGProbe
    directly without assembling command-line arguments.
    """

    def __init__(
        self,
        *,
        llm: str | None = None,
        model: str = DEFAULT_QWEN_MODEL,
        base_url: str | None = None,
        api_key_env: str = "AI_API_KEY",
        cache_dir: str | Path = ".ragprobe_cache",
        use_cache: bool = True,
    ) -> None:
        self.llm = llm
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.cache_dir = cache_dir
        self.use_cache = use_cache

    def load_testset(self, path: str | Path) -> TestSet:
        return load_testset(path)

    def load_results(self, path: str | Path) -> list[RetrievalResult]:
        return load_results(path)

    def generate(
        self,
        *,
        chunks: str | Path | list,
        output: str | Path | None = None,
        num_cases: int | None = None,
        hard_negative_top_k: int = 1,
        name: str = "generated-testset",
        hn_strategy: str = "hybrid",
        llm: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool | None = None,
        quality_report: str | Path | None = None,
        llm_validate: bool = False,
        judge_llm: str | None = None,
        judge_model: str | None = None,
        judge_base_url: str | None = None,
        judge_api_key_env: str | None = None,
        keep_rejected: bool = False,
    ) -> TestSet:
        chunk_items = load_chunks(chunks) if isinstance(chunks, (str, Path)) else chunks
        provider = llm if llm is not None else self.llm
        selected_model = model or self.model
        selected_api_key_env = api_key_env or self.api_key_env
        selected_cache_dir = self.cache_dir if cache_dir is None else cache_dir
        selected_use_cache = self.use_cache if use_cache is None else use_cache

        if provider:
            if provider == "qwen":
                selected_base_url = base_url or self.base_url or DEFAULT_QWEN_BASE_URL
                client = QwenClient.from_env(
                    env_var=selected_api_key_env,
                    model=selected_model,
                    base_url=selected_base_url,
                )
            elif provider == "openai-compatible":
                selected_base_url = base_url or self.base_url
                if not selected_base_url:
                    raise ValueError("base_url is required for llm='openai-compatible'")
                client = OpenAICompatibleClient.from_env(
                    env_var=selected_api_key_env,
                    model=selected_model,
                    base_url=selected_base_url,
                )
            else:
                raise ValueError(f"unsupported LLM provider: {provider}")
            config = LLMGenerationConfig(
                provider=provider,
                model=selected_model,
                base_url=selected_base_url,
                api_key_env=selected_api_key_env,
                hard_negative_top_k=hard_negative_top_k,
                hn_strategy=hn_strategy,
            )
            judge_client = None
            if llm_validate:
                judge_provider = judge_llm or provider
                judge_selected_model = judge_model or selected_model
                judge_selected_env = judge_api_key_env or selected_api_key_env
                if judge_provider == "qwen":
                    judge_client = QwenClient.from_env(
                        env_var=judge_selected_env,
                        model=judge_selected_model,
                        base_url=judge_base_url or DEFAULT_QWEN_BASE_URL,
                    )
                elif judge_provider == "openai-compatible":
                    judge_selected_base_url = judge_base_url or selected_base_url
                    if not judge_selected_base_url:
                        raise ValueError(
                            "judge_base_url is required for judge_llm='openai-compatible'"
                        )
                    judge_client = OpenAICompatibleClient.from_env(
                        env_var=judge_selected_env,
                        model=judge_selected_model,
                        base_url=judge_selected_base_url,
                    )
                else:
                    raise ValueError(f"unsupported judge LLM provider: {judge_provider}")
            testset = generate_testset_from_chunks_llm(
                chunk_items,
                client=client,
                num_cases=num_cases,
                hard_negative_top_k=hard_negative_top_k,
                name=name,
                hn_strategy=hn_strategy,
                cache_dir=selected_cache_dir,
                use_cache=selected_use_cache,
                config=config,
                judge_client=judge_client,
                keep_rejected=keep_rejected,
            )
        else:
            testset = generate_testset_from_chunks(
                chunk_items,
                num_cases=num_cases,
                hard_negative_top_k=hard_negative_top_k,
                name=name,
                hn_strategy=hn_strategy,
            )

        if output:
            save_json(testset, output)
        if quality_report:
            _write_text(quality_report, render_quality_report(testset))
        return testset

    def run(
        self,
        *,
        testset: TestSet | str | Path,
        retriever: str | Path | None = None,
        retriever_fn=None,
        retriever_cmd: str | None = None,
        endpoint: str | None = None,
        endpoint_config: str | Path | None = None,
        output: str | Path | None = None,
        top_k: int = 10,
        timeout: float = 30.0,
        batch_size: int = 1,
        content_match_threshold: float = 0.9,
        ) -> list[RetrievalResult]:
        loaded_testset = _coerce_testset(testset)
        sources = [
            retriever is not None,
            retriever_fn is not None,
            retriever_cmd is not None,
            endpoint is not None,
        ]
        if sum(sources) != 1:
            raise ValueError("run requires exactly one retriever source")

        if retriever is not None:
            results = run_retriever_script(
                loaded_testset,
                retriever,
                top_k=top_k,
                content_fallback_threshold=content_match_threshold,
            )
        elif retriever_fn is not None:
            results = run_retriever(
                loaded_testset,
                retriever_fn,
                top_k=top_k,
                content_fallback_threshold=content_match_threshold,
            )
        elif retriever_cmd is not None:
            results = run_retriever_command(
                loaded_testset,
                retriever_cmd,
                top_k=top_k,
                timeout=timeout,
                content_fallback_threshold=content_match_threshold,
            )
        else:
            config = load_endpoint_config(endpoint_config)
            results = run_endpoint(
                loaded_testset,
                endpoint or "",
                top_k=top_k,
                timeout=config.timeout if endpoint_config else timeout,
                headers=config.headers,
                batch_size=config.batch_size if endpoint_config else batch_size,
                content_fallback_threshold=content_match_threshold,
            )

        if output:
            save_json({"results": results}, output)
        return results

    def diagnose(
        self,
        *,
        testset: TestSet | str | Path,
        results: list[RetrievalResult] | str | Path,
        content_match_threshold: float = 0.9,
    ) -> DiagnosticReport:
        loaded_testset = _coerce_testset(testset)
        loaded_results = _coerce_results(results)
        matched = apply_content_fallback(
            loaded_testset,
            loaded_results,
            threshold=content_match_threshold,
        )
        return DiagnosticAnalyzer().analyze(loaded_testset, matched)

    def compare(
        self,
        *,
        testset: TestSet | str | Path,
        before: list[RetrievalResult] | str | Path,
        after: list[RetrievalResult] | str | Path,
        content_match_threshold: float = 0.9,
    ) -> ComparisonReport:
        before_report = self.diagnose(
            testset=testset,
            results=before,
            content_match_threshold=content_match_threshold,
        )
        after_report = self.diagnose(
            testset=testset,
            results=after,
            content_match_threshold=content_match_threshold,
        )
        return compare_reports(before_report, after_report)

    def experiment(
        self,
        *,
        config: dict[str, Any] | str | Path,
        output_dir: str | Path | None = None,
    ) -> ExperimentReport:
        """Run a multi-retriever experiment from a JSON-like config."""
        return run_experiment(config, output_dir=output_dir)

    def audit(
        self,
        *,
        testset: TestSet | str | Path,
        output: str | Path | None = None,
        markdown: str | Path | None = None,
        llm: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        sample_size: int | None = None,
        case_ids: list[str] | None = None,
        cache_dir: str | Path | None = None,
        use_cache: bool | None = None,
        judge_client=None,
    ) -> AuditReport:
        """Audit an existing testset with an LLM judge."""
        provider = llm or self.llm
        if judge_client is None and not provider:
            raise ValueError("audit requires llm or judge_client")
        selected_model = model or self.model
        selected_api_key_env = api_key_env or self.api_key_env
        selected_cache_dir = self.cache_dir if cache_dir is None else cache_dir
        selected_use_cache = self.use_cache if use_cache is None else use_cache

        if judge_client is None:
            if provider == "qwen":
                selected_base_url = base_url or self.base_url or DEFAULT_QWEN_BASE_URL
                judge_client = QwenClient.from_env(
                    env_var=selected_api_key_env,
                    model=selected_model,
                    base_url=selected_base_url,
                )
            elif provider == "openai-compatible":
                selected_base_url = base_url or self.base_url
                if not selected_base_url:
                    raise ValueError("base_url is required for llm='openai-compatible'")
                judge_client = OpenAICompatibleClient.from_env(
                    env_var=selected_api_key_env,
                    model=selected_model,
                    base_url=selected_base_url,
                )
            else:
                raise ValueError(f"unsupported LLM provider: {provider}")
        else:
            selected_base_url = base_url or self.base_url or DEFAULT_QWEN_BASE_URL
            provider = provider or "custom"

        config = LLMGenerationConfig(
            provider=provider,
            model=selected_model,
            base_url=selected_base_url,
            api_key_env=selected_api_key_env,
            prompt_version="ragprobe-v0.9-testset-audit-v1",
        )
        report = audit_testset(
            testset,
            judge_client=judge_client,
            config=config,
            sample_size=sample_size,
            case_ids=case_ids,
            cache_dir=selected_cache_dir,
            use_cache=selected_use_cache,
        )
        if output:
            save_audit_report(report, output)
        if markdown:
            from ragprobe.reports.markdown import render_audit_markdown

            _write_text(markdown, render_audit_markdown(report))
        return report

    def repair_plan(
        self,
        *,
        audit_report,
        output: str | Path | None = None,
        markdown: str | Path | None = None,
    ) -> RepairPlan:
        """Build a reviewable repair plan from an audit report."""
        plan = build_repair_plan(
            audit_report,
            source=str(audit_report) if isinstance(audit_report, (str, Path)) else "",
        )
        if output:
            save_repair_plan(plan, output)
        if markdown:
            from ragprobe.reports.markdown import render_repair_plan_markdown

            _write_text(markdown, render_repair_plan_markdown(plan))
        return plan

    def apply_audit_fixes(
        self,
        *,
        testset: TestSet | str | Path,
        repair_plan: RepairPlan | dict | str | Path,
        output: str | Path | None = None,
        report: str | Path | None = None,
        allow_reject_cases: bool = False,
    ) -> RepairApplyResult:
        """Apply an audit repair plan to a copy of a testset."""
        result = apply_repair_plan(
            testset,
            repair_plan,
            allow_reject_cases=allow_reject_cases,
        )
        if output:
            save_json(result.testset, output)
        if report:
            from ragprobe.reports.markdown import render_repair_apply_markdown

            _write_text(report, render_repair_apply_markdown(result))
        return result

    def check(
        self,
        report: DiagnosticReport,
        *,
        min_hit_rate: float | None = None,
        min_mrr: float | None = None,
        max_fpr: float | None = None,
        max_low_confidence_match_rate: float | None = None,
    ) -> CheckResult:
        return check_thresholds(
            report,
            min_hit_rate=min_hit_rate,
            min_mrr=min_mrr,
            max_fpr=max_fpr,
            max_low_confidence_match_rate=max_low_confidence_match_rate,
        )

    def evaluate(
        self,
        *,
        testset: TestSet | str | Path | None = None,
        chunks: str | Path | list | None = None,
        retriever: str | Path | None = None,
        retriever_fn=None,
        retriever_cmd: str | None = None,
        endpoint: str | None = None,
        endpoint_config: str | Path | None = None,
        llm: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key_env: str | None = None,
        num_cases: int | None = None,
        hard_negative_top_k: int = 1,
        top_k: int = 10,
        llm_validate: bool = False,
    ) -> DiagnosticReport:
        if testset is None:
            if chunks is None:
                raise ValueError("evaluate requires either testset or chunks")
            testset = self.generate(
                chunks=chunks,
                llm=llm,
                model=model,
                base_url=base_url,
                api_key_env=api_key_env,
                num_cases=num_cases,
                hard_negative_top_k=hard_negative_top_k,
                llm_validate=llm_validate,
            )
        results = self.run(
            testset=testset,
            retriever=retriever,
            retriever_fn=retriever_fn,
            retriever_cmd=retriever_cmd,
            endpoint=endpoint,
            endpoint_config=endpoint_config,
            top_k=top_k,
        )
        return self.diagnose(testset=testset, results=results)


def _coerce_testset(testset: TestSet | str | Path) -> TestSet:
    return load_testset(testset) if isinstance(testset, (str, Path)) else testset


def _coerce_results(results: list[RetrievalResult] | str | Path) -> list[RetrievalResult]:
    return load_results(results) if isinstance(results, (str, Path)) else results


def _write_text(path: str | Path, text: str) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")


__all__ = ["RAGProbe", "LLMGenerationError"]
