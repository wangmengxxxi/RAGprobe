from __future__ import annotations

import argparse
import json

from qualira.backends.sqlite import SQLiteEvidenceBackend
from qualira.retrieval.executor import RetrievalExecutor


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run evidence-qualified retrieval.")
    parser.add_argument("query", help="Natural-language query")
    parser.add_argument("--db", default="qualira.db", help="SQLite database path")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    args = parser.parse_args(argv)

    backend = SQLiteEvidenceBackend(args.db)
    result = RetrievalExecutor(backend).execute(args.query, limit=args.limit)
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0

    print("RetrievalPlan:")
    print(f"  intent: {result.plan.intent}")
    print(f"  domain: {result.plan.domain}")
    print("  required_claims:")
    for field, value in result.plan.required_claims.items():
        print(f"    {field}: {value}")
    print(f"  target_answer_type: {result.plan.target_answer_type}")

    print("\nResults:")
    print("  selected:")
    if not result.selected:
        print("    (none)")
    for item in result.selected:
        print(f"    {item.unit.id} score={item.score:.4f}")
        print(f"      content: {item.unit.content}")
        print(f"      source: {item.unit.source.file} page={item.unit.source.page} section={' > '.join(item.unit.source.section_path)}")
        if args.explain:
            print("      matched:")
            for field, value in item.matched_claims.items():
                print(f"        {field}={value}")

    if args.explain:
        print("\n  excluded:")
        if not result.excluded:
            print("    (none)")
        for item in result.excluded:
            unit = backend.get(item.unit_id)
            print(f"    {item.unit_id}")
            if unit:
                print(f"      content: {unit.content}")
            print(f"      reason: {item.reason}, {item.difference_type}")
            if item.detail:
                print(f"      detail: {item.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
