from __future__ import annotations

import hashlib

from qualira.backends.base import EvidenceBackend
from qualira.core.schema import (
    EvidenceUnit,
    Exclusion,
    QueryTrace,
    RetrievalPlan,
    RetrievalResult,
    ScoredEvidence,
)


def parse_query_to_plan(query: str, domain: str = "contract") -> RetrievalPlan:
    normalized = query.lower()
    required: dict[str, str] = {}
    exclude: list[dict[str, str]] = []

    answer_type = _parse_answer_type(query)

    event_patterns = [
        ("late_payment", ["逾期付款", "迟延付款", "未按期付款", "付款逾期", "payment delay", "late payment"]),
        ("late_delivery", ["延期交货", "延期交付", "迟延交货", "逾期交货", "未按期交货", "不能按期交货", "delivery delay", "late delivery"]),
        ("quality_defect", ["质量不合格", "质量缺陷", "瑕疵", "defect"]),
        ("confidentiality_breach", ["泄密", "泄露", "保密", "confidential"]),
        ("invoice_delay", ["发票", "invoice"]),
        ("acceptance_delay", ["验收", "acceptance"]),
        ("force_majeure", ["不可抗力", "不能履约", "force majeure"]),
        ("warranty", ["质保", "保修", "warranty"]),
        ("termination", ["解除", "终止", "停止供货", "termination"]),
        ("notice", ["通知", "notice"]),
    ]
    for value, patterns in event_patterns:
        if any(pattern in query or pattern in normalized for pattern in patterns):
            required["event"] = value
            break

    condition_patterns = [
        ("over_15_days", ["超过15日但未超过30日", "超过15天但未超过30天", "超过15", "超15", "15日", "15天", "over 15"]),
        ("over_30_days", ["超过30", "超30", "30日", "30天", "over 30"]),
        ("over_10_days", ["超过10", "超10", "10日", "10天", "over 10"]),
        ("within_5_days", ["5日内", "5天内", "within 5"]),
        ("within_3_days", ["3日内", "3天内", "within 3"]),
        ("after_notice", ["及时通知", "已及时通知", "after notice"]),
        ("material_breach", ["根本违约", "重大违约", "material breach"]),
        ("non_conforming_goods", ["不合格货物", "non-conforming"]),
        ("unpaid_amount", ["未付款项", "欠款"]),
    ]
    for value, patterns in condition_patterns:
        if any(pattern in query or pattern in normalized for pattern in patterns):
            required["condition"] = value
            break

    scope_patterns = [
        ("main_contract", ["本合同", "主合同", "main contract"]),
        ("appendix", ["附件", "appendix"]),
        ("purchase_order", ["订单", "purchase order"]),
    ]
    for value, patterns in scope_patterns:
        if any(pattern in query or pattern in normalized for pattern in patterns):
            required["scope"] = value
            break

    subject = _parse_subject(query, answer_type)
    if subject:
        required["subject"] = subject
        if subject == "buyer":
            exclude.append({"subject": "seller"})
        elif subject == "seller":
            exclude.append({"subject": "buyer"})

    intent = {
        "liability": "ask_liability",
        "notice_obligation": "ask_notice",
        "termination_right": "ask_termination",
        "warranty_obligation": "ask_warranty",
        "invoice_obligation": "ask_invoice",
    }.get(answer_type, "ask")

    return RetrievalPlan(
        intent=intent,
        domain=domain,
        required_claims=required,
        target_answer_type=answer_type,
        exclude=exclude,
        recall_strategy=["symbolic", "fulltext"],
    )


def _parse_answer_type(query: str) -> str | None:
    if any(token in query for token in ["质保", "保修"]):
        return "warranty_obligation"
    if any(token in query for token in ["补开", "补开发票", "提供合格发票"]):
        return "invoice_obligation"
    if any(token in query for token in ["补开发票", "提供合格发票", "发票"]):
        if not any(token in query for token in ["未付款", "逾期付款", "付款", "责任"]):
            return "invoice_obligation"
    if any(token in query for token in ["是否可以解除", "有权解除", "可以解除"]):
        return "termination_right"
    if any(token in query for token in ["责任", "违约金", "赔偿", "承担", "后果"]):
        return "liability"
    if any(token in query for token in ["通知", "告知"]):
        return "notice_obligation"
    if any(token in query for token in ["解除权"]):
        return "termination_right"
    return None


def _parse_subject(query: str, answer_type: str | None) -> str | None:
    if any(pattern in query for pattern in ["卖方需要", "卖方应", "卖方是否", "卖方收到", "卖方没有", "卖方未", "卖方预计"]):
        return "seller"
    if any(pattern in query for pattern in ["买方需要", "买方应", "买方是否", "买方收到", "买方发现"]):
        return "buyer"
    buyer_patterns = [
        "买方逾期",
        "买方未",
        "买方无合同",
        "买方违反",
        "买方泄露",
        "买方发现",
        "买方收到",
        "买方应",
        "买方需要",
        "买方是否可以解除",
    ]
    seller_patterns = [
        "卖方延期",
        "卖方预计",
        "卖方交付",
        "卖方未",
        "卖方没有",
        "卖方无合同",
        "卖方违反",
        "卖方泄露",
        "卖方停止",
        "卖方收到",
        "卖方承担",
        "卖方应",
        "卖方需要",
    ]
    if any(pattern in query for pattern in buyer_patterns):
        return "buyer"
    if any(pattern in query for pattern in seller_patterns):
        return "seller"
    if answer_type == "termination_right" and "买方" in query:
        return "buyer"
    if "卖方" in query and any(token in query for token in ["责任", "违约金", "赔偿", "通知", "质保", "发票"]):
        return "seller"
    if "买方" in query and any(token in query for token in ["责任", "违约金", "赔偿", "通知"]):
        return "buyer"
    return None


class RetrievalExecutor:
    def __init__(self, backend: EvidenceBackend) -> None:
        self.backend = backend

    def execute(self, query: str, plan: RetrievalPlan | None = None, limit: int = 5) -> RetrievalResult:
        plan = plan or parse_query_to_plan(query)
        symbolic = self.backend.query_by_claims(
            plan.required_claims,
            answer_type=plan.target_answer_type,
            exclude=None,
        )
        fulltext = self.backend.fulltext_search(query, limit=20)
        candidates = _dedupe([*symbolic, *fulltext])

        selected: list[ScoredEvidence] = []
        excluded: list[Exclusion] = []
        for unit in candidates:
            exclusion = self._exclusion_for(unit, plan)
            if exclusion:
                excluded.append(exclusion)
                continue
            score, matched = score_eligibility(unit, plan)
            if score > 0:
                selected.append(ScoredEvidence(unit=unit, score=score, matched_claims=matched))

        selected.sort(key=lambda item: (-item.score, item.unit.id))
        selected = selected[:limit]
        trace = QueryTrace(
            query_id=_query_id(query, plan),
            query=query,
            plan=plan,
            candidates=[unit.id for unit in candidates],
            selected=[item.unit.id for item in selected],
            excluded=excluded,
        )
        self.backend.record_trace(trace)
        return RetrievalResult(
            plan=plan,
            selected=selected,
            excluded=excluded,
            candidates=[unit.id for unit in candidates],
            trace=trace,
        )

    def _exclusion_for(self, unit: EvidenceUnit, plan: RetrievalPlan) -> Exclusion | None:
        claim_map = unit.claim_map
        if plan.exclude:
            for item in plan.exclude:
                if _mapping_matches(claim_map, item):
                    return Exclusion(
                        unit_id=unit.id,
                        reason="plan exclusion",
                        difference_type="excluded_claims",
                        detail=f"matched exclude_when={item}",
                    )
        plan_values = dict(plan.required_claims)
        if plan.target_answer_type:
            plan_values["target_answer_type"] = plan.target_answer_type
        for boundary in unit.boundaries:
            if _mapping_matches(plan_values, boundary.exclude_when):
                return Exclusion(
                    unit_id=unit.id,
                    reason="boundary exclusion",
                    difference_type=boundary.difference_type,
                    detail=boundary.difference_detail,
                )
        return None


def score_eligibility(unit: EvidenceUnit, plan: RetrievalPlan) -> tuple[float, dict[str, str]]:
    claim_map = unit.claim_map
    if plan.target_answer_type is not None and unit.answer_type.value != plan.target_answer_type:
        return 0.0, {}
    for field, value in plan.required_claims.items():
        if field in claim_map and claim_map[field] != value:
            return 0.0, {}
    matched = {
        field: value
        for field, value in plan.required_claims.items()
        if claim_map.get(field) == value
    }
    required_count = max(len(plan.required_claims), 1)
    claim_score = len(matched) / required_count
    answer_score = 1.0 if plan.target_answer_type is None or unit.answer_type.value == plan.target_answer_type else 0.0
    trust_score = {
        "human_verified": 1.0,
        "auto_verified": 0.9,
        "draft": 0.5,
        "conflict": 0.2,
        "deprecated": 0.0,
    }.get(unit.status, 0.5)
    source_score = unit.claim_source_grounding_rate()
    score = (0.55 * claim_score) + (0.25 * answer_score) + (0.10 * trust_score) + (0.10 * source_score)
    if plan.required_claims and claim_score < 0.5:
        score = 0.0
    if plan.target_answer_type is not None and answer_score == 0.0:
        score = 0.0
    return round(score, 4), matched


def _dedupe(units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    seen: set[str] = set()
    results: list[EvidenceUnit] = []
    for unit in units:
        if unit.id not in seen:
            seen.add(unit.id)
            results.append(unit)
    return results


def _mapping_matches(values: dict[str, str], required: dict[str, str]) -> bool:
    return all(values.get(field) == value for field, value in required.items())


def _query_id(query: str, plan: RetrievalPlan) -> str:
    digest = hashlib.sha1(f"{query}|{plan.to_dict()}".encode("utf-8")).hexdigest()[:12]
    return f"q_{digest}"
