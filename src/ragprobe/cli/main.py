"""RAGProbe command line entry point.

Phase 0 exposes the command surface without implementing behavior.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragprobe")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("demo", help="Run the bundled offline demo.")

    diagnose = subparsers.add_parser("diagnose", help="Diagnose offline retrieval results.")
    diagnose.add_argument("--testset", required=False)
    diagnose.add_argument("--results", required=False)
    diagnose.add_argument("--output", required=False)
    diagnose.add_argument("--format", choices=["terminal", "markdown", "json"], default="terminal")

    compare = subparsers.add_parser("compare", help="Compare two retrieval result sets.")
    compare.add_argument("--testset", required=False)
    compare.add_argument("--before", required=False)
    compare.add_argument("--after", required=False)
    compare.add_argument("--output", required=False)

    check = subparsers.add_parser("check", help="Fail when diagnostic metrics cross thresholds.")
    check.add_argument("--report", required=False)
    check.add_argument("--results", required=False)
    check.add_argument("--testset", required=False)
    check.add_argument("--min-hit-rate", type=float, required=False)
    check.add_argument("--max-fpr", type=float, required=False)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0

    raise NotImplementedError(
        f"Command '{args.command}' is part of the v0.1 surface and is not implemented in Phase 0."
    )
