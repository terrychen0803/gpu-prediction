from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

MODES = ("G", "A", "P0", "P1")
PHASES = ("stem", "layer1", "layer2", "layer3", "layer4", "head")
DEFAULTS = {
    "seed": 42, "cpu_threads": 1, "warmup_min": 50, "warmup_window": 20,
    "warmup_max": 500, "stability_threshold": 0.05, "min_iterations": 100,
    "min_seconds": 1.0, "max_seconds": 30.0, "transition_iterations": 5,
    "rtol": 1e-2, "atol": 1e-3,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows, fields=None):
    rows = list(rows)
    if fields is None:
        fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_hash():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def quantile(values, q):
    if not values:
        return None
    xs = sorted(values)
    idx = (len(xs) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)


def stats(values):
    if not values:
        return {k: None for k in ("median", "mean", "p10", "p90", "min", "max")}
    return {"median": statistics.median(values), "mean": statistics.mean(values),
            "p10": quantile(values, .1), "p90": quantile(values, .9), "min": min(values), "max": max(values)}


def union_length(intervals):
    intervals = sorted((a, b) for a, b in intervals if b > a)
    if not intervals:
        return 0
    start, end = intervals[0]
    total = 0
    for a, b in intervals[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start


def command_info(command):
    try:
        r = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=20)
        return {"returncode": r.returncode, "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"unavailable": str(exc)}


def load_settings(path=None):
    settings = dict(DEFAULTS)
    if path:
        supplied = read_json(path)
        unknown = supplied.keys() - DEFAULTS.keys()
        if unknown:
            raise ValueError(f"Unknown settings: {sorted(unknown)}")
        settings.update(supplied)
    for key in ("cpu_threads", "warmup_min", "warmup_window", "warmup_max", "min_iterations", "transition_iterations"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["warmup_max"] < settings["warmup_min"] + 3 * settings["warmup_window"]:
        raise ValueError("warmup_max must allow warmup_min plus three stability windows")
    if not 0 < settings["min_seconds"] <= settings["max_seconds"]:
        raise ValueError("Require 0 < min_seconds <= max_seconds")
    if not 0 < settings["stability_threshold"] < 1:
        raise ValueError("stability_threshold must be in (0, 1)")
    if settings["rtol"] < 0 or settings["atol"] < 0:
        raise ValueError("Correctness tolerances must be nonnegative")
    return settings
