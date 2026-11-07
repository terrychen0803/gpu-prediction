from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="O1 operator pilot")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    sub.add_parser("calibrate")
    worker = sub.add_parser("worker")
    worker.add_argument("--request", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("--root", required=True)
    summary.add_argument("--out", required=True)
    args, rest = parser.parse_known_args()
    if args.command == "run":
        from .runner import main as run_main
        return run_main(rest)
    if args.command == "calibrate":
        from .calibration import main as calibrate_main
        return calibrate_main(rest)
    if rest:
        parser.error(f"Unexpected arguments: {rest}")
    if args.command == "worker":
        from .worker import run_worker
        return run_worker(args.request)
    from .summarize import summarize
    summarize(args.root, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

