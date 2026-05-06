from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from qualira.backends.base import EvidenceBackend
from qualira.core.schema import Boundary, EvidenceUnit, QueryTrace


class SQLiteEvidenceBackend(EvidenceBackend):
    def __init__(self, path: str | Path = "qualira.db") -> None:
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        if str(path) != ":memory:":
            self.conn.execute("pragma journal_mode=MEMORY")
            self.conn.execute("pragma synchronous=NORMAL")
        self._init_schema()

    def close(self) -> None:
        self.conn.close()

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            create table if not exists evidence_units (
                id text primary key,
                domain text not null,
                content text not null,
                answer_type text not null,
                source_file text not null,
                source_page integer,
                status text not null,
                payload text not null
            );

            create table if not exists claims (
                unit_id text not null,
                field text not null,
                value text not null,
                confidence real not null,
                primary key (unit_id, field, value),
                foreign key (unit_id) references evidence_units(id) on delete cascade
            );

            create index if not exists idx_claims_field_value on claims(field, value);
            create index if not exists idx_evidence_domain_answer on evidence_units(domain, answer_type);

            create table if not exists traces (
                query_id text primary key,
                payload text not null,
                created_at text not null
            );
            """
        )
        self.conn.commit()

    def reset(self) -> None:
        self.conn.executescript(
            """
            delete from traces;
            delete from claims;
            delete from evidence_units;
            """
        )
        self.conn.commit()

    def store(self, unit: EvidenceUnit) -> str:
        with self.conn:
            self.conn.execute(
                """
                insert or replace into evidence_units
                (id, domain, content, answer_type, source_file, source_page, status, payload)
                values (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    unit.id,
                    unit.domain,
                    unit.content,
                    unit.answer_type.value,
                    unit.source.file,
                    unit.source.page,
                    unit.status,
                    json.dumps(unit.to_dict(), ensure_ascii=False),
                ),
            )
            self.conn.execute("delete from claims where unit_id = ?", (unit.id,))
            self.conn.executemany(
                """
                insert into claims (unit_id, field, value, confidence)
                values (?, ?, ?, ?)
                """,
                [(unit.id, claim.field, claim.value, claim.confidence) for claim in unit.claims],
            )
        return unit.id

    def store_many(self, units: list[EvidenceUnit]) -> int:
        for unit in units:
            self.store(unit)
        return len(units)

    def get(self, unit_id: str) -> EvidenceUnit | None:
        row = self.conn.execute("select payload from evidence_units where id = ?", (unit_id,)).fetchone()
        if row is None:
            return None
        return EvidenceUnit.from_dict(json.loads(row["payload"]))

    def all_units(self) -> list[EvidenceUnit]:
        rows = self.conn.execute("select payload from evidence_units order by id").fetchall()
        return [EvidenceUnit.from_dict(json.loads(row["payload"])) for row in rows]

    def query_by_claims(
        self,
        claims: dict[str, str],
        answer_type: str | None = None,
        exclude: list[dict[str, str]] | None = None,
    ) -> list[EvidenceUnit]:
        units = self.all_units()
        results: list[EvidenceUnit] = []
        for unit in units:
            if answer_type is not None and unit.answer_type.value != answer_type:
                continue
            claim_map = unit.claim_map
            if all(claim_map.get(field) == value for field, value in claims.items()):
                if exclude and any(_mapping_matches(claim_map, item) for item in exclude):
                    continue
                results.append(unit)
        return results

    def fulltext_search(self, text: str, limit: int = 20) -> list[EvidenceUnit]:
        from qualira.eval.baselines.naive_chunk import score_text_similarity

        scored = [(score_text_similarity(text, unit.content), unit) for unit in self.all_units()]
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [unit for score, unit in scored[:limit] if score > 0]

    def get_boundaries(self, unit_id: str) -> list[Boundary]:
        unit = self.get(unit_id)
        return unit.boundaries if unit else []

    def record_trace(self, trace: QueryTrace) -> None:
        payload = trace.to_dict()
        with self.conn:
            self.conn.execute(
                """
                insert or replace into traces (query_id, payload, created_at)
                values (?, ?, ?)
                """,
                (trace.query_id, json.dumps(payload, ensure_ascii=False), payload["timestamp"]),
            )


def load_evidence_file(path: str | Path) -> list[EvidenceUnit]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_units = data["evidence_units"] if isinstance(data, dict) else data
    return [EvidenceUnit.from_dict(item) for item in raw_units]


def _mapping_matches(values: dict[str, str], required: dict[str, str]) -> bool:
    return all(values.get(field) == value for field, value in required.items())
