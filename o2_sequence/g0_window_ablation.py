"""Independent O2 60-template G0 window-size comparison using the frozen pilot worker.

The worker measurement protocol remains o2_g0_window_pilot_v4_exploratory_20261101.
This separate design protocol specifies the full 60-template, randomized 8/256/512
comparison, 3 repeat processes and immutable run output. Neither is O2 V3 G0.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import subprocess
import sys
from pathlib import Path

from o1_operator.full_matrix import BY_ID as FORMAL
from o1_operator.spec import INPUT_FORMAT, write_inputs
from o2_sequence.catalog import FROZEN_PATH, PROTOCOL as V3_PROTOCOL, SPLIT_PATH, load
from o2_sequence import g0_window_pilot as pilot
from o2_sequence.runner import code_hash as v3_hash, hash_file

DESIGN_PROTOCOL = "o2_g0_window_60template_ablation_v1_20261101"
COUNTS = (8, 256, 512)
REPEATS = 3
GROUPS = 5
SHUFFLE_SEED = 20261101
ROOT = Path(__file__).resolve().parent.parent


def read(path):
    return json.loads(Path(path).read_text())


def put(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def execute(out, expect_gpu, dry_run=False):
    out = Path(out).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite session: {out}")
    if v3_hash() != pilot.REQUIRED_SOURCE_HASH:
        raise RuntimeError("O2 V3 code hash changed; refuse incompatible experiment")
    if GROUPS != pilot.GROUPS or not set(COUNTS).issubset(set(pilot.COUNTS)):
        raise RuntimeError("Experiment diverges from frozen pilot measurement worker")
    templates = load(True)
    if len(templates) != 60:
        raise RuntimeError("Expected exactly 60 formal templates")
    ids = sorted({step["config_id"] for t in templates for step in t["steps"]})
    configs = {config_id: FORMAL[config_id] for config_id in ids}
    out.mkdir(parents=True)
    inputs = write_inputs(out / "inputs", configs.values())
    put(out / "input_inventory.json", inputs)

    jobs = [
        {"template_id": t["id"], "sequence_count": count, "repeat": repeat}
        for t in templates for count in COUNTS for repeat in range(1, REPEATS + 1)
    ]
    random.Random(SHUFFLE_SEED).shuffle(jobs)
    plan = {
        "design_protocol": DESIGN_PROTOCOL,
        "measurement_protocol": pilot.PROTOCOL,
        "base_v3_protocol": V3_PROTOCOL,
        "base_v3_code_sha256": v3_hash(),
        "pilot_worker_sha256": pilot.file_sha(pilot.__file__),
        "catalog_sha256": hash_file(FROZEN_PATH),
        "o1_split_sha256": hash_file(SPLIT_PATH),
        "input_format": INPUT_FORMAT,
        "matrix": "formal", "mode": "G0",
        "groups": GROUPS, "sequence_counts": list(COUNTS),
        "repeats": REPEATS, "shuffle_seed": SHUFFLE_SEED,
        "expect_gpu": expect_gpu,
        "template_ids": [t["id"] for t in templates],
        "jobs": jobs, "failures": [],
        "status": "planned_only" if dry_run else "running",
        "measurement": "whole-sequence host batch time / sequence count",
        "group_window_target_ms": None,
        "preflight_gpu": pilot.short_device_snapshot(),
    }
    put(out / "plan.json", plan)
    by_id = {t["id"]: t for t in templates}

    for i, job in enumerate(jobs, 1):
        template_id = job["template_id"]
        n = job["sequence_count"]
        repeat = job["repeat"]
        folder = out / "runs" / template_id / f"n{n:04d}" / f"run{repeat:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        request = {
            "protocol": pilot.PROTOCOL,
            "source_protocol": V3_PROTOCOL,
            "source_code_sha256": v3_hash(),
            "matrix": "formal", "mode": "G0",
            "input_format": INPUT_FORMAT, "template": by_id[template_id],
            "configs": configs, "inputs": inputs, "groups": GROUPS,
            "sequence_count": n, "output": str(folder),
            "expect_gpu": expect_gpu,
        }
        put(folder / "request.json", request)
        argv = [sys.executable, "-m", "o2_sequence.g0_window_pilot",
                "worker", "--request", str(folder / "request.json")]
        put(folder / "command.json", {"argv": argv, "cwd": str(ROOT)})
        print(f"ABLATION {i}/{len(jobs)} {template_id} n={n} repeat={repeat}", flush=True)
        if dry_run:
            put(folder / "execution.json", {"status": "planned_only"})
            continue
        try:
            with (folder / "worker.log").open("w") as f:
                code = subprocess.run(
                    argv, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                    timeout=600, check=False,
                ).returncode
            quality = read(folder / "quality.json")
            if code != 0 or quality.get("status") != "completed":
                raise RuntimeError(f"Worker return {code}; check worker.log")
            put(folder / "execution.json", {"status": "completed"})
        except Exception as exc:
            failure = {**job, "error": f"{type(exc).__name__}: {exc}"}
            plan["failures"].append(failure)
            put(folder / "execution.json", {"status": "failed", "error": failure["error"]})
            print(f"ERROR: {failure}", file=sys.stderr)
    plan["status"] = "planned_only" if dry_run else (
        "completed_with_failures" if plan["failures"] else "completed"
    )
    plan["failure_count"] = len(plan["failures"])
    put(out / "plan.json", plan)
    print(f"FINISHED: {len(jobs)-len(plan['failures'])}/{len(jobs)}; "
          f"failed={len(plan['failures'])}")
    return 1 if plan["failures"] else 0


def summarize(root):
    root = Path(root).expanduser().resolve()
    plan = read(root / "plan.json")
    if plan["design_protocol"] != DESIGN_PROTOCOL:
        raise RuntimeError("Wrong experiment design")
    if plan["status"] != "completed" or plan["failure_count"] != 0:
        raise RuntimeError("Cannot accept incomplete collection")
    expected_jobs = 60 * len(COUNTS) * REPEATS
    if len(plan["jobs"]) != expected_jobs:
        raise RuntimeError(f"Expected {expected_jobs} jobs")
    if len(plan["template_ids"]) != 60 or len(set(plan["template_ids"])) != 60:
        raise RuntimeError("Wrong template count")

    analysis = root / "analysis"
    analysis.mkdir(exist_ok=False)
    rows = []
    for template_id in plan["template_ids"]:
        for n in COUNTS:
            medians, group_windows, intrarun_spreads = [], [], []
            for repeat in range(1, REPEATS + 1):
                folder = root / "runs" / template_id / f"n{n:04d}" / f"run{repeat:02d}"
                if read(folder / "execution.json")["status"] != "completed":
                    raise RuntimeError(f"Failed run: {folder}")
                quality = read(folder / "quality.json")
                if quality["status"] != "completed":
                    raise RuntimeError(f"Failed worker quality: {folder}")
                expected_correctness = ("passed" if template_id.endswith("_dependent_tail")
                                        else "not_applicable")
                if quality["dependent_correctness"]["status"] != expected_correctness:
                    raise RuntimeError(f"Failed dependent correctness: {folder}")
                with (folder / "groups.csv").open() as f:
                    groups = list(csv.DictReader(f))
                if len(groups) != GROUPS or {int(x["group"]) for x in groups} != set(range(1,GROUPS+1)):
                    raise RuntimeError(f"Wrong groups: {folder}")
                if any(int(g["sequence_count"]) != n for g in groups):
                    raise RuntimeError(f"Wrong sequence count: {folder}")
                times = [float(g["host_us_per_sequence"]) for g in groups]
                windows = [float(g["host_sequence_batch_us"]) / 1000 for g in groups]
                if not all(x > 0 for x in times + windows):
                    raise RuntimeError(f"Invalid group timing: {folder}")
                medians.append(statistics.median(times))
                group_windows += windows
                intrarun_spreads.append(
                    (max(times)-min(times)) / statistics.median(times) * 100
                )
            repeat_spread = (max(medians) - min(medians))/statistics.median(medians)*100
            rows.append({
                "template": template_id, "topology": ("dependent_tail" if template_id.endswith("_dependent_tail") else "independent"),
                "sequences_per_group": n,
                "repeat_spread_pct": round(repeat_spread, 5),
                "stable_le_5pct": repeat_spread <= 5,
                "run1_us": round(medians[0], 5),
                "run2_us": round(medians[1], 5),
                "run3_us": round(medians[2], 5),
                "median_sequence_us": round(statistics.median(medians),5),
                "median_group_ms": round(statistics.median(group_windows),5),
                "min_group_ms": round(min(group_windows),5),
                "intrarun_group_spread_pct_median": round(statistics.median(intrarun_spreads),5),
            })
    with (analysis / "window_quality.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = []
    for n in COUNTS:
        subset = [r for r in rows if r["sequences_per_group"] == n]
        report.append({
            "sequences_per_group": n,
            "stable": sum(r["stable_le_5pct"] for r in subset),
            "total": len(subset),
            "median_repeat_spread_pct": statistics.median(r["repeat_spread_pct"] for r in subset),
            "min_group_ms": min(r["min_group_ms"] for r in subset),
            "median_group_ms": statistics.median(r["median_group_ms"] for r in subset),
        })
    by_template = {}
    for r in rows:
        by_template.setdefault(r["template"], {})[r["sequences_per_group"]] = r
    paired = {}
    for n in (256,512):
        improved = sum(not x[8]["stable_le_5pct"] and x[n]["stable_le_5pct"]
                       for x in by_template.values())
        worsened = sum(x[8]["stable_le_5pct"] and not x[n]["stable_le_5pct"]
                       for x in by_template.values())
        paired[str(n)] = {"fail_to_pass": improved, "pass_to_fail": worsened}
    result = {
        "design_protocol": DESIGN_PROTOCOL,
        "measurement_protocol": pilot.PROTOCOL,
        "completed": expected_jobs,
        "group_policy": GROUPS,
        "stability_threshold": "3-run spread <=5% (exploratory quality threshold)",
        "window_summary": report, "paired_vs_8": paired,
        "note": "Fixed window-length comparison on 60 templates; not V3 trace validation.",
    }
    put(analysis / "summary.json", result)
    print("===== RTX3090 O2 G0 60-TEMPLATE WINDOW ABLATION =====")
    print(f"COLLECTION: {expected_jobs}/{expected_jobs} PASS")
    for x in report:
        n = x["sequences_per_group"]
        print(f"n={n}: stable={x['stable']}/60 "
              f"median_spread={x['median_repeat_spread_pct']:.2f}% "
              f"group_median={x['median_group_ms']:.2f}ms "
              f"group_min={x['min_group_ms']:.2f}ms")
        if n in (256,512):
            print("   vs n=8:", paired[str(n)])
    print("Review:", analysis)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("run")
    q.add_argument("--out", required=True)
    q.add_argument("--expect-gpu", default="3090")
    q.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("review")
    q.add_argument("--root", required=True)
    a = p.parse_args()
    if a.cmd == "run":
        return execute(a.out, a.expect_gpu, a.dry_run)
    summarize(a.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
