from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


InferenceType = Literal["explicit", "inherited", "inferred"]
EvidenceStatus = Literal["draft", "auto_verified", "human_verified", "conflict", "deprecated"]
BoundarySource = Literal["structural", "contrastive", "feedback"]
Feedback = Literal["correct", "wrong_evidence", "missing_evidence"]


@dataclass(slots=True)
class SourceSpan:
    text: str
    file: str
    page: int | None = None
    section_path: list[str] = field(default_factory=list)
    char_range: tuple[int, int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SourceSpan:
        char_range = data.get("char_range")
        return cls(
            text=str(data.get("text", "")),
            file=str(data.get("file", "")),
            page=data.get("page"),
            section_path=list(data.get("section_path", [])),
            char_range=tuple(char_range) if char_range is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "file": self.file,
            "page": self.page,
            "section_path": self.section_path,
            "char_range": list(self.char_range) if self.char_range is not None else None,
        }


@dataclass(slots=True)
class FieldClaim:
    field: str
    value: str
    inference_type: InferenceType = "explicit"
    source: SourceSpan | None = None
    confidence: float = 1.0
    verified_by: list[str] = field(default_factory=list)
    verified_at: datetime | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FieldClaim:
        verified_at = data.get("verified_at")
        return cls(
            field=str(data["field"]),
            value=str(data["value"]),
            inference_type=data.get("inference_type", "explicit"),
            source=SourceSpan.from_dict(data["source"]) if data.get("source") else None,
            confidence=float(data.get("confidence", 1.0)),
            verified_by=list(data.get("verified_by", [])),
            verified_at=datetime.fromisoformat(verified_at) if verified_at else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "inference_type": self.inference_type,
            "source": self.source.to_dict() if self.source else None,
            "confidence": self.confidence,
            "verified_by": self.verified_by,
            "verified_at": self.verified_at.isoformat() if self.verified_at else None,
        }


@dataclass(slots=True)
class Boundary:
    confusable_with: str
    difference_type: str
    difference_detail: str
    exclude_when: dict[str, str]
    source: BoundarySource = "structural"
    confidence: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Boundary:
        return cls(
            confusable_with=str(data["confusable_with"]),
            difference_type=str(data["difference_type"]),
            difference_detail=str(data.get("difference_detail", "")),
            exclude_when={str(k): str(v) for k, v in data.get("exclude_when", {}).items()},
            source=data.get("source", "structural"),
            confidence=float(data.get("confidence", 1.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "confusable_with": self.confusable_with,
            "difference_type": self.difference_type,
            "difference_detail": self.difference_detail,
            "exclude_when": self.exclude_when,
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass(slots=True)
class EvidenceUnit:
    id: str
    content: str
    source: SourceSpan
    domain: str
    claims: list[FieldClaim]
    answer_type: FieldClaim
    boundaries: list[Boundary] = field(default_factory=list)
    status: EvidenceStatus = "human_verified"
    version: str = "1.0"
    supersedes: str | None = None
    related: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceUnit:
        answer_type = data["answer_type"]
        if isinstance(answer_type, str):
            answer_claim = FieldClaim(field="answer_type", value=answer_type, verified_by=["manual"])
        else:
            answer_claim = FieldClaim.from_dict(answer_type)
        return cls(
            id=str(data["id"]),
            content=str(data["content"]),
            source=SourceSpan.from_dict(data["source"]),
            domain=str(data.get("domain", "contract")),
            claims=[FieldClaim.from_dict(item) for item in data.get("claims", [])],
            answer_type=answer_claim,
            boundaries=[Boundary.from_dict(item) for item in data.get("boundaries", [])],
            status=data.get("status", "human_verified"),
            version=str(data.get("version", "1.0")),
            supersedes=data.get("supersedes"),
            related=list(data.get("related", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source.to_dict(),
            "domain": self.domain,
            "claims": [claim.to_dict() for claim in self.claims],
            "answer_type": self.answer_type.to_dict(),
            "boundaries": [boundary.to_dict() for boundary in self.boundaries],
            "status": self.status,
            "version": self.version,
            "supersedes": self.supersedes,
            "related": self.related,
        }

    @property
    def claim_map(self) -> dict[str, str]:
        return {claim.field: claim.value for claim in self.claims}

    def claim_source_grounding_rate(self) -> float:
        claims = [*self.claims, self.answer_type]
        if not claims:
            return 0.0
        grounded = sum(1 for claim in claims if claim.source is not None or self.source.text)
        return grounded / len(claims)


@dataclass(slots=True)
class RetrievalPlan:
    intent: str
    domain: str
    required_claims: dict[str, str]
    target_answer_type: str | None = None
    exclude: list[dict[str, str]] = field(default_factory=list)
    recall_strategy: list[str] = field(default_factory=lambda: ["symbolic", "fulltext"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RetrievalPlan:
        return cls(
            intent=str(data.get("intent", "ask")),
            domain=str(data.get("domain", "contract")),
            required_claims={str(k): str(v) for k, v in data.get("required_claims", {}).items()},
            target_answer_type=data.get("target_answer_type"),
            exclude=[{str(k): str(v) for k, v in item.items()} for item in data.get("exclude", [])],
            recall_strategy=list(data.get("recall_strategy", ["symbolic", "fulltext"])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "domain": self.domain,
            "required_claims": self.required_claims,
            "target_answer_type": self.target_answer_type,
            "exclude": self.exclude,
            "recall_strategy": self.recall_strategy,
        }


@dataclass(slots=True)
class Exclusion:
    unit_id: str
    reason: str
    difference_type: str | None = None
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "reason": self.reason,
            "difference_type": self.difference_type,
            "detail": self.detail,
        }


@dataclass(slots=True)
class ScoredEvidence:
    unit: EvidenceUnit
    score: float
    matched_claims: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.unit.id,
            "score": self.score,
            "matched_claims": self.matched_claims,
            "content": self.unit.content,
            "source": self.unit.source.to_dict(),
        }


@dataclass(slots=True)
class QueryTrace:
    query_id: str
    query: str
    plan: RetrievalPlan
    candidates: list[str]
    selected: list[str]
    excluded: list[Exclusion]
    feedback: Feedback | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "plan": self.plan.to_dict(),
            "candidates": self.candidates,
            "selected": self.selected,
            "excluded": [item.to_dict() for item in self.excluded],
            "feedback": self.feedback,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass(slots=True)
class RetrievalResult:
    plan: RetrievalPlan
    selected: list[ScoredEvidence]
    excluded: list[Exclusion]
    candidates: list[str]
    trace: QueryTrace

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "selected": [item.to_dict() for item in self.selected],
            "excluded": [item.to_dict() for item in self.excluded],
            "candidates": self.candidates,
            "trace": self.trace.to_dict(),
        }
