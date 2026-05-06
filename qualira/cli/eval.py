from __future__ import annotations

import argparse
import json

from qualira.eval.benchmark import run_benchmark


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 0 hard-negative benchmark.")
    parser.add_argument("--evidence", default="tests/fixtures/contract_evidence.json")
    parser.add_argument("--queries", default="tests/fixtures/contract_queries.json")
    parser.add_argument("--db", default=":memory:")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--details", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    metrics, details, table = run_benchmark(args.evidence, args.queries, db_path=args.db, top_k=args.top_k)
    if args.json:
        print(
            json.dumps(
                {
                    "metrics": [metric.as_row() for metric in metrics],
                    "details": details if args.details else None,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    print(table)
    if args.details:
        print("\nDetails:")
        for item in details:
            print(f"- {item['query']}")
            print(f"  expected: {', '.join(item['expected_evidence'])}")
            print(f"  qualira selected: {', '.join(item['qualira_selected']) or '(none)'}")
            print(f"  qualira excluded: {', '.join(item['qualira_excluded']) or '(none)'}")
            print(f"  naive selected: {', '.join(item['naive_selected']) or '(none)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
