from __future__ import annotations

import argparse
import json
import sys

from .common import MODES, load_settings


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description="ResNet50 inference pilot: G/A/P0/P1, Nsight correlation, summaries")
    subs = parser.add_subparsers(dest="command", required=True)
    d = subs.add_parser("doctor", help="Inspect PyTorch/CUDA/Nsight readiness without running an experiment")
    d.add_argument("--out")
    p = subs.add_parser("prepare", help="Download fixed IMAGENET1K_V2 weights once; no timing")
    p.add_argument("--out", default="assets/resnet50_imagenet1k_v2.pth")
    p = subs.add_parser("prepare-input", help="Create a portable FP16 input file once; no GPU timing")
    p.add_argument("--out", required=True)
    p.add_argument("--max-batch", type=positive_int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p = subs.add_parser("prepare-grid-set", help="Generate/verify all E3 fixed inputs from the committed manifest")
    p.add_argument("--root", default="inputs/e3_grid_v1")
    p.add_argument("--manifest", default="inputs/e3_grid_manifest.json")
    p = subs.add_parser("prepare-grid-input", help="Create architecture-independent FP16 input for E3 resolution grid")
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=positive_int, required=True)
    p.add_argument("--max-batch", type=positive_int, default=32)
    p.add_argument("--seed", type=int, default=42)
    for command in ("run", "suite"):
        p = subs.add_parser(command)
        p.add_argument("--out", required=True)
        p.add_argument("--weights", default="assets/resnet50_imagenet1k_v2.pth")
        p.add_argument("--input-file", help="Checksum-verified FP16 binary shared with every device")
        p.add_argument("--settings", help="JSON overrides of preregistered defaults")
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--label", required=True, help="Platform identifier; stored alongside actual detected hardware")
        p.add_argument("--expect-gpu", help="Require this substring in the detected GPU name, e.g. 4090")
        p.add_argument("--nsys", default="nsys")
        p.add_argument("--nsys-cuda-trace", choices=("cuda", "cuda-sw"), default="cuda",
                       help="Nsight CUDA tracing backend; cuda-sw is a fallback for incomplete hardware traces")
        p.add_argument("--dry-run", action="store_true", help="Write commands/schedule only; no model execution or download")
        if command == "run":
            p.add_argument("--mode", choices=MODES, required=True)
            p.add_argument("--batch", type=positive_int, default=1)
            p.add_argument("--resolution", type=positive_int, default=224)
        else:
            p.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
            p.add_argument("--batches", type=positive_int, nargs="+", default=[1, 8, 32])
            p.add_argument("--resolutions", type=positive_int, nargs="+", default=[224])
            p.add_argument("--repeats", type=positive_int, default=3)
            p.add_argument("--order-seed", type=int, default=2026)
            p.add_argument("--keep-going", action="store_true", help="Continue other scheduled runs after a failed run")
    p = subs.add_parser("parse", help="Parse a completed P0/P1 SQLite export")
    p.add_argument("run_dir")
    p.add_argument("--sqlite")
    p = subs.add_parser("summarize", help="Aggregate successful runs; never use target test labels for predictions")
    p.add_argument("roots", nargs="+")
    p.add_argument("--out", required=True)
    p = subs.add_parser("plot", help="Generate diagnostic plots with matplotlib")
    p.add_argument("roots", nargs="+")
    p.add_argument("--out", required=True)
    p = subs.add_parser("_worker", help=argparse.SUPPRESS)
    p.add_argument("--request", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from .runner import doctor
            info = doctor(args.out)
            print(json.dumps(info, indent=2, ensure_ascii=False))
            return 0 if info["ready_for_P0_P1"] else 1
        if args.command == "prepare":
            from .benchmark import prepare_weights
            prepare_weights(args.out)
        elif args.command == "prepare-input":
            from .input_data import prepare_input
            info = prepare_input(args.out, args.max_batch, args.seed)
            print(json.dumps(info, ensure_ascii=False, indent=2))
        elif args.command == "prepare-grid-input":
            from .input_data import prepare_grid_input
            info = prepare_grid_input(args.out, args.resolution, args.max_batch, args.seed)
            print(json.dumps(info, ensure_ascii=False, indent=2))
        elif args.command == "prepare-grid-set":
            from .input_data import prepare_grid_set
            info = prepare_grid_set(args.root, args.manifest)
            print(json.dumps(info, ensure_ascii=False, indent=2))
        elif args.command == "_worker":
            from .benchmark import worker
            return worker(args.request)
        elif args.command in ("run", "suite"):
            from .runner import make_suite, run_one
            if args.command == "run":
                request = {"out": args.out, "weights": args.weights, "settings": load_settings(args.settings),
                           "device": args.device, "label": args.label, "expect_gpu": args.expect_gpu,
                           "mode": args.mode, "batch": args.batch, "repeat": 1,
                           "resolution": args.resolution,
                           "input_file": args.input_file,
                           "nsys_cuda_trace": args.nsys_cuda_trace}
                return 0 if run_one(request, args.dry_run, args.nsys) else 1
            requests = make_suite(args)
            failed = 0
            for i, request in enumerate(requests, 1):
                print(f"[{i}/{len(requests)}] {request['label']} R{request['resolution']} B{request['batch']} {request['mode']} run{request['repeat']}", flush=True)
                if not run_one(request, args.dry_run, args.nsys):
                    failed += 1
                    if not args.keep_going:
                        break
            print(f"{'Planned' if args.dry_run else 'Finished'}; failures={failed}; output={args.out}")
            return 1 if failed else 0
        elif args.command == "parse":
            from .trace import parse_trace
            result = parse_trace(args.run_dir, args.sqlite)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0 if result["status"] == "passed" else 1
        elif args.command == "summarize":
            from .report import summarize
            summarize(args.roots, args.out)
        elif args.command == "plot":
            from .plots import plot
            plot(args.roots, args.out)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
