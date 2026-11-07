"""Optional matplotlib plots; tables and trace parsing need only the standard library."""
from __future__ import annotations

import collections
from pathlib import Path

from .common import MODES, PHASES, read_csv, read_json
from .report import collect, summarize


def plot(roots, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError("Plots require matplotlib; install requirements-analysis.txt") from exc
    out = Path(out)
    summarize(roots, out)
    rows, _ = collect(roots)
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["protocol"], row["label"], row["batch"])].append(row)
    for (signature, label, batch), runs in groups.items():
        prefix = f"{signature}_{label}_b{batch}"
        g_runs = [r for r in runs if r["mode"] == "G"]
        if g_runs:
            fig, ax = plt.subplots(figsize=(9, 4))
            for run in g_runs:
                directory = Path(run["run_dir"])
                warm = read_csv(directory / "warmup.csv")
                measured = [r for r in read_csv(directory / "latency.csv") if r["stage"] == "measure"]
                ys = [float(r["latency_us"]) for r in warm + measured]
                ax.plot(range(len(ys)), ys, alpha=.65, label=f"run {run['repeat']}")
                ax.axvline(len(warm), color="gray", linestyle="--", alpha=.4)
            ax.set(xlabel="Iteration (warm-up, then measured)", ylabel="Host service time (us)", title=f"{label}, batch {batch}, G")
            ax.legend(); fig.tight_layout(); fig.savefig(out / f"{prefix}_latency.png", dpi=160); plt.close(fig)
        labels, distributions = [], []
        for mode in MODES:
            subset = [r for r in runs if r["mode"] == mode]
            if subset:
                labels.append(mode)
                distributions.append([float(s["latency_us"]) for r in subset for s in read_csv(Path(r["run_dir"]) / "latency.csv") if s["stage"] == "measure"])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.boxplot(distributions, tick_labels=labels, showfliers=False)
        for i, mode in enumerate(labels, 1):
            ax.scatter([i] * sum(r["mode"] == mode for r in runs), [r["median_us"] for r in runs if r["mode"] == mode], color="red", zorder=3)
        ax.set(ylabel="Host service time (us)", title=f"{label}, batch {batch}; red = independent run medians")
        fig.tight_layout(); fig.savefig(out / f"{prefix}_modes.png", dpi=160); plt.close(fig)
        p1 = next((r for r in runs if r["mode"] == "P1"), None)
        if p1:
            directory = Path(p1["run_dir"])
            mappings = [r for r in read_csv(directory / "phase_mapping.csv") if r["stage"] == "measure" and r["iteration"] == "0"]
            if mappings:
                zero = min(float(r["api_start_ns"]) for r in mappings)
                fig, ax = plt.subplots(figsize=(11, 5))
                colors = dict(zip(PHASES, plt.get_cmap("tab10").colors))
                for r in read_csv(directory / "phase_ranges.csv"):
                    if r["stage"] == "measure" and r["iteration"] == "0":
                        ax.broken_barh([((float(r["start_ns"]) - zero) / 1000, (float(r["end_ns"]) - float(r["start_ns"])) / 1000)], (2.1, .5), facecolors=colors[r["phase"]])
                seen_apis = set()
                for r in mappings:
                    color = colors.get(r["phase"], "gray")
                    if r["api_id"] not in seen_apis:
                        ax.broken_barh([((float(r["api_start_ns"]) - zero) / 1000, (float(r["api_end_ns"]) - float(r["api_start_ns"])) / 1000)], (1.1, .5), facecolors=color)
                        seen_apis.add(r["api_id"])
                    ax.broken_barh([((float(r["kernel_start_ns"]) - zero) / 1000, float(r["duration_us"]))], (.1, .5), facecolors=color)
                ax.set(yticks=[.35, 1.35, 2.35], yticklabels=["GPU kernels", "CUDA APIs", "CPU NVTX phases"], xlabel="Trace time relative to first launch (us)", title=f"{label}, batch {batch}, P1, iteration 0; phase colors")
                ax.legend(handles=[Patch(facecolor=colors[p], label=p) for p in PHASES], loc="upper right", ncol=3, fontsize=8)
                fig.tight_layout(); fig.savefig(out / f"{prefix}_timeline.png", dpi=160); plt.close(fig)
            totals = collections.defaultdict(float)
            counts = collections.defaultdict(int)
            for r in read_csv(directory / "phase_summary.csv"):
                totals[r["phase"]] += float(r["kernel_duration_sum_us"])
                counts[r["phase"]] += int(r["kernel_count"])
            names = list(PHASES) + ["unassigned"]
            total = sum(totals.values())
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].bar(names, [100 * totals[n] / total if total else 0 for n in names])
            axes[1].bar(names, [counts[n] for n in names])
            for ax in axes:
                ax.tick_params(axis="x", rotation=30)
            axes[0].set(ylabel="Share of kernel duration sum (%)", title="P1 diagnostic (not host service decomposition)")
            axes[1].set(ylabel="Kernel count", title=f"{label}, batch {batch}")
            fig.tight_layout(); fig.savefig(out / f"{prefix}_phases.png", dpi=160); plt.close(fig)
    ratios = read_csv(out / "device_ratios.csv")
    if ratios:
        fig, ax = plt.subplots(figsize=(8, 4))
        groups = collections.defaultdict(list)
        for r in ratios:
            groups[(r["protocol"], r["source"], r["target"])].append(r)
        for (signature, source, target), rs in groups.items():
            rs.sort(key=lambda r: int(r["batch"]))
            ax.plot([int(r["batch"]) for r in rs], [float(r["target_over_source"]) for r in rs], marker="o", label=f"{source} -> {target} ({signature[:6]})")
        ax.set(xlabel="Batch size", ylabel="Target / source service time", title="Unprofiled G medians")
        ax.legend(); fig.tight_layout(); fig.savefig(out / "device_ratios.png", dpi=160); plt.close(fig)
    print(f"Plots: {out.resolve()}")
