# RTX3090 O2 G0 Timing Window Pilot (V4 exploratory; 2026-11-01)

This is a **new, independent diagnostic protocol**, **not** a modification of O2 V3. It does not relabel the existing O2 G0/P/R data. Base code commit: [77ccb5d](https://github.com/terrychen0803/gpu-prediction/commit/77ccb5db59a359cdba1ef61bc54c48c33ffc28bb). Base V3 code hash is pinned by the pilot runner.

## Why

On RTX3090, historical O2 G0 stability (3-repeat, <=5% spread) was 22/60; recollected clean O2 was 42/60; a G0-only control was 39/60. Both topologies still show time variation, and V3's **5 groups × 8 sequences** means each group is usually only around 1.6–2.8 ms.

A **100 ms minimum window** is a **hypothesis to test**, not evidence already established. This pilot compares different numbers of complete sequences while preserving original V3 workload shapes, input bytes, CPU/GPU execution policy, synchronization at group boundaries, warmup count and time-label definition. Large counts change amortization, cache/power conditions and measurement policy: their outputs are **not V3 G0 labels**.

## Frozen choices

- Independent protocol: `o2_g0_window_pilot_v4_exploratory_20261101`
- Frozen V3 source protocol: `o2_mixed_sequence_gb10_v3_chain_norm_checked_20261024`
- Unmodified O2 V3 code-hash guard: `d9ca6fdd0442f386629c2b5ab459dd6f4162a72277907ed205ad16fb63524f1f`
- Same formal template generator, O1 split, SHAKE256/FP16 inputs and one CUDA stream.
- **5 selected diagnostic templates × 4 window sizes × 3 independent repeat processes = 60 jobs**.
- Template IDs:
  - `fold0_v00_dependent_tail`
  - `fold0_v04_dependent_tail`
  - `fold0_v04_independent`
  - `fold3_v03_independent`
  - `fold4_v01_dependent_tail`
- Sequence counts per group: **8, 64, 256, 512**.
- 5 groups per run; each group has one `torch.cuda.synchronize()` before and after the batch; time label is `perf_counter_ns()` whole-group host duration divided by sequence count. No per-step sync, marker, Nsight or CUDA Event.
- The 60 jobs are shuffled deterministically using seed 20261101, to reduce confounding between window size and time-of-day. This does not eliminate environmental variation.
- Time quality threshold: repeat median spread <=5%, **exploratory only**; report absolute group durations too.

## Use on RTX3090

Use a fresh checkout so that older O1 and O2 sessions are untouched:

```bash
cd ~/RTchen
git clone --branch o2-g0-window-pilot-20261101 \
  https://github.com/terrychen0803/gpu-prediction.git gpu-prediction-o2-g0-v4
conda activate gpu-prediction-o1
cd gpu-prediction-o2-g0-v4
export CUDA_VISIBLE_DEVICES=0
python -m unittest discover -s tests -q
python -m o2_sequence.g0_window_pilot --help
```

Before collection verify the RTX3090 is idle and, on Kubernetes, unscheduled for other users. This runner checks one visible CUDA GPU, expected device, source code hash, correctness, unique session output and each individual job status; it does not continuously monitor co-tenants.

```bash
OUT="$HOME/RTchen/o2-runs/rtx3090_o2_g0_window_pilot_v4_20261101_001"
python -m o2_sequence.g0_window_pilot run \
  --expect-gpu 3090 --out "$OUT"
python -m o2_sequence.g0_window_pilot review --root "$OUT"
```

For a plan-only preflight use `--dry-run` **with a different, nonexistent** `--out` directory: each output path is intentionally create-only.

## Output and interpretation

Each job has `request.json`, `command.json`, `worker.log`, `execution.json`, `quality.json`, `environment.json`, and `groups.csv`. The root has `plan.json`, `input_inventory.json` and generated `inputs/`. Review creates `analysis/window_quality.csv` and `analysis/summary.json`.

Report for each sequence count:

- stable templates out of 5, under the same three-repeat 5% criterion;
- median and **minimum** group duration in milliseconds;
- per-template repeat median spread;
- intra-run group-spread metric (descriptive only);
- environmental/system load evidence and anomalous groups.

**Do not claim 100 ms achieved simply because 512 sequences are requested.** Check the observed `min_group_ms` and `median_group_ms` fields.

**Do not promote these five diagnostically selected templates into a general unbiased estimate of all 60 templates.** If increased windows improve time stability, the next study is a separately versioned **60-template calibration** with a preregistered target and cap, frozen per-template sequences/group, validation on independent repeats, and clearly differentiated labels.

**Do not delete** any V3 O2 raw data, the G0 control, O1 data, or archives as part of this pilot. Packaging and cleanup are separate, after validation.

## Known limitations

- A larger G0 measurement group may alter cache, clock and workload batching behavior; the measured time is not an identical estimand to V3's 8-sequence batch.
- Single-node CPU load and GPU clocks could drift despite idle preflight. GPU snapshots at run boundaries do not establish per-group clocks.
- Host timing includes launch/submission overhead and synchronization, not just GPU kernel execution.
- O2 V3's `verified_steps=0` due to NVTX warnings remains unresolved; this G0-only pilot does **not** repair reference completeness.
