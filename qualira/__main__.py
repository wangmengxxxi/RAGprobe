from __future__ import annotations

import argparse

from qualira.cli import eval as eval_cli
from qualira.cli import ingest, query


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="qualira")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("ingest")
    subparsers.add_parser("query")
    subparsers.add_parser("eval")
    args, rest = parser.parse_known_args(argv)

    if args.command == "ingest":
        return ingest.main(rest)
    if args.command == "query":
        return query.main(rest)
    if args.command == "eval":
        return eval_cli.main(rest)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
