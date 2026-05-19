"""Command-line entry point."""

from __future__ import annotations

import argparse
from typing import List, Optional

from ad_phmm_align import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the AD-PHMM-align CLI parser."""

    parser = argparse.ArgumentParser(prog="ad-phmm-align")
    parser.add_argument("--version", action="version", version=__version__)
    subcommands = parser.add_subparsers(dest="command")

    subcommands.add_parser(
        "validate-artifacts",
        help="Validate graph and initialization artifacts.",
    )
    subcommands.add_parser("train", help="Train an AD-PHMM model.")
    subcommands.add_parser("decode", help="Decode alignments with a trained model.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Run the command-line interface."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    raise NotImplementedError(f"CLI command is not implemented yet: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
