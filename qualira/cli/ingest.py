from __future__ import annotations

import argparse

from qualira.backends.sqlite import SQLiteEvidenceBackend, load_evidence_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import manual EvidenceUnit JSON into SQLite.")
    parser.add_argument("path", help="Path to contract_evidence.json")
    parser.add_argument("--db", default="qualira.db", help="SQLite database path")
    parser.add_argument("--domain", default="contract", help="Expected domain label")
    parser.add_argument("--mode", default="manual", choices=["manual"], help="Phase 0 supports manual only")
    parser.add_argument("--reset", action="store_true", help="Clear existing evidence before import")
    args = parser.parse_args(argv)

    units = load_evidence_file(args.path)
    for unit in units:
        if unit.domain != args.domain:
            unit.domain = args.domain

    backend = SQLiteEvidenceBackend(args.db)
    if args.reset:
        backend.reset()
    count = backend.store_many(units)
    print(f"Imported {count} EvidenceUnit records into {args.db}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
