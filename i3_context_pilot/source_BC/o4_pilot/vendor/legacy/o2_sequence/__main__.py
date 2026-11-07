from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="O2 mixed sequence collection")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run")
    review = sub.add_parser("review")
    review.add_argument("--root", required=True)
    review.add_argument("--out", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--request", required=True)
    args, rest = parser.parse_known_args()
    if args.command == "run":
        from .runner import main as run_main
        return run_main(rest)
    if rest:
        parser.error(f"Unknown arguments: {rest}")
    if args.command == "review":
        from .alignment import review_root
        print(review_root(args.root, args.out))
        return 0
    from .worker import run
    return run(args.request)


if __name__ == "__main__":
    sys.exit(main())
