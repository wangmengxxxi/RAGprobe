from __future__ import annotations

from abc import ABC, abstractmethod

from qualira.core.schema import Boundary, EvidenceUnit, QueryTrace


class EvidenceBackend(ABC):
    @abstractmethod
    def store(self, unit: EvidenceUnit) -> str:
        raise NotImplementedError

    @abstractmethod
    def store_many(self, units: list[EvidenceUnit]) -> int:
        raise NotImplementedError

    @abstractmethod
    def get(self, unit_id: str) -> EvidenceUnit | None:
        raise NotImplementedError

    @abstractmethod
    def all_units(self) -> list[EvidenceUnit]:
        raise NotImplementedError

    @abstractmethod
    def query_by_claims(
        self,
        claims: dict[str, str],
        answer_type: str | None = None,
        exclude: list[dict[str, str]] | None = None,
    ) -> list[EvidenceUnit]:
        raise NotImplementedError

    @abstractmethod
    def fulltext_search(self, text: str, limit: int = 20) -> list[EvidenceUnit]:
        raise NotImplementedError

    @abstractmethod
    def get_boundaries(self, unit_id: str) -> list[Boundary]:
        raise NotImplementedError

    @abstractmethod
    def record_trace(self, trace: QueryTrace) -> None:
        raise NotImplementedError
