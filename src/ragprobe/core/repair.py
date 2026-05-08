"""Repair plans derived from testset audit reports."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ragprobe.core.models import TestSet
from ragprobe.core.schema import SCHEMA_REPAIR_PLAN, SCHEMA_TESTSET, schema_metadata
from ragprobe.io.jsonl import load_json, load_testset, save_json

REPAIR_PLAN_VERSION = "ragprobe-v0.10-audit-repair-plan-v1"


@dataclass
class RepairAction:
    action: str
    case_id: str
    chunk_id: str = ""
    reason: str = ""
    source_warning: str = ""
    applies: bool = True


@dataclass
class RepairPlan:
    actions: list[RepairAction] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RepairApplyResult:
    testset: TestSet
    applied_actions: list[RepairAction] = field(default_factory=list)
    skipped_actions: list[RepairAction] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def build_repair_plan(audit_report: Any, *, source: str = "") -> RepairPlan:
    payload = load_json(audit_report) if isinstance(audit_report, (str, Path)) else audit_report
    findings = _get(payload, "findings", [])
    actions: list[RepairAction] = []

    for finding in findings:
        case_id = str(_get(finding, "case_id", ""))
        warnings = set(_get(finding, "warnings", []))
        if "expected_chunk_not_answerable" in warnings:
            actions.append(
                RepairAction(
                    action="reject_case",
                    case_id=case_id,
                    reason="Expected chunk was judged unable to answer the query.",
                    source_warning="expected_chunk_not_answerable",
                )
            )
        if "missing_expected_chunk_content" in warnings:
            actions.append(
                RepairAction(
                    action="reject_case",
                    case_id=case_id,
                    reason="Expected chunk content was missing from testset metadata.",
                    source_warning="missing_expected_chunk_content",
                )
            )

        for hard_negative in _get(finding, "hard_negative_findings", []):
            chunk_id = str(_get(hard_negative, "chunk_id", ""))
            if bool(_get(hard_negative, "answerable", False)):
                actions.append(
                    RepairAction(
                        action="remove_hard_negative",
                        case_id=case_id,
                        chunk_id=chunk_id,
                        reason=(
                            "Hard negative was judged answerable and may be a mislabeled "
                            "negative."
                        ),
                        source_warning="hard_negative_answerable",
                    )
                )
            elif _get(hard_negative, "risk", "") == "medium" and (
                "missing_hard_negative_chunk_content" in warnings
            ):
                actions.append(
                    RepairAction(
                        action="review_metadata_chunks",
                        case_id=case_id,
                        chunk_id=chunk_id,
                        reason="Hard negative chunk content was missing from testset metadata.",
                        source_warning="missing_hard_negative_chunk_content",
                        applies=False,
                    )
                )

    deduped = _dedupe_actions(actions)
    return RepairPlan(
        actions=deduped,
        summary=_summarize_actions(deduped),
        metadata={
            **schema_metadata(SCHEMA_REPAIR_PLAN),
            "source": "ragprobe-v0.10-audit-repair-plan",
            "repair_plan_version": REPAIR_PLAN_VERSION,
            "source_audit": source or _get(payload, "metadata", {}).get("source", ""),
            "audit_summary": _get(payload, "summary", {}),
        },
    )


def apply_repair_plan(
    testset: TestSet | str | Path,
    repair_plan: RepairPlan | dict[str, Any] | str | Path,
    *,
    allow_reject_cases: bool = False,
) -> RepairApplyResult:
    loaded_testset = load_testset(testset) if isinstance(testset, (str, Path)) else testset
    plan = _coerce_plan(repair_plan)
    updated = copy.deepcopy(loaded_testset)
    applied: list[RepairAction] = []
    skipped: list[RepairAction] = []

    cases_by_id = {case.id: case for case in updated.cases}
    rejected_case_ids: set[str] = set()

    for action in plan.actions:
        case = cases_by_id.get(action.case_id)
        if case is None or not action.applies:
            skipped.append(action)
            continue

        if action.action == "remove_hard_negative":
            before_count = len(case.hard_negatives)
            case.hard_negatives = [
                item for item in case.hard_negatives if item.chunk_id != action.chunk_id
            ]
            if len(case.hard_negatives) < before_count:
                applied.append(action)
            else:
                skipped.append(action)
        elif action.action == "reject_case":
            if allow_reject_cases:
                rejected_case_ids.add(action.case_id)
                applied.append(action)
            else:
                skipped.append(action)
        else:
            skipped.append(action)

    if rejected_case_ids:
        updated.cases = [case for case in updated.cases if case.id not in rejected_case_ids]

    updated.metadata.setdefault("repair_history", [])
    updated.metadata.update(schema_metadata(SCHEMA_TESTSET))
    if isinstance(updated.metadata["repair_history"], list):
        updated.metadata["repair_history"].append(
            {
                "repair_plan_version": plan.metadata.get(
                    "repair_plan_version", REPAIR_PLAN_VERSION
                ),
                "applied_actions": len(applied),
                "skipped_actions": len(skipped),
                "allow_reject_cases": allow_reject_cases,
            }
        )

    return RepairApplyResult(
        testset=updated,
        applied_actions=applied,
        skipped_actions=skipped,
        summary={
            "applied_actions": len(applied),
            "skipped_actions": len(skipped),
            "remaining_cases": len(updated.cases),
            "rejected_cases": len(rejected_case_ids),
        },
    )


def save_repair_plan(plan: RepairPlan, path: str | Path) -> None:
    save_json(plan, path)


def _coerce_plan(plan: RepairPlan | dict[str, Any] | str | Path) -> RepairPlan:
    payload = load_json(plan) if isinstance(plan, (str, Path)) else plan
    if isinstance(payload, RepairPlan):
        return payload
    return RepairPlan(
        actions=[
            RepairAction(
                action=item["action"],
                case_id=item["case_id"],
                chunk_id=item.get("chunk_id", ""),
                reason=item.get("reason", ""),
                source_warning=item.get("source_warning", ""),
                applies=bool(item.get("applies", True)),
            )
            for item in payload.get("actions", [])
        ],
        summary=dict(payload.get("summary", {})),
        metadata=dict(payload.get("metadata", {})),
    )


def _dedupe_actions(actions: list[RepairAction]) -> list[RepairAction]:
    seen = set()
    deduped = []
    for action in actions:
        key = (action.action, action.case_id, action.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def _summarize_actions(actions: list[RepairAction]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    applicable = 0
    for action in actions:
        counts[action.action] = counts.get(action.action, 0) + 1
        if action.applies:
            applicable += 1
    return {
        "total_actions": len(actions),
        "applicable_actions": applicable,
        "review_only_actions": len(actions) - applicable,
        "action_counts": dict(sorted(counts.items())),
    }


def _get(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)
