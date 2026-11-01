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


## Observed RTX 3090 pilot results (2026-11-01)

The user-reported on-node run finished with **60/60 completed, 0 failures**. Analysis was generated under `o2-runs/rtx3090_o2_g0_window_pilot_v4_20261101_001/analysis`.

| Sequences per group | Stable under <=5% repeat spread | Median group duration | Minimum group duration |
|---:|---:|---:|---:|
| 8 | 3/5 | 1.992 ms | 1.710 ms |
| 64 | 3/5 | 15.291 ms | 13.369 ms |
| 256 | **5/5** | 61.937 ms | 53.676 ms |
| 512 | 4/5 | 125.411 ms | 107.439 ms |

Full per-template **repeat spread (%)**, in ascending sequence-count order 8 / 64 / 256 / 512:

| Template | 8 | 64 | 256 | 512 |
|---|---:|---:|---:|---:|
| `fold0_v00_dependent_tail` | 4.01 | 1.72 | 1.56 | 2.33 |
| `fold0_v04_independent` | 4.92 | 2.67 | 4.73 | 1.77 |
| `fold0_v04_dependent_tail` | 7.99 | 3.01 | 3.96 | **5.09** |
| `fold3_v03_independent` | 1.81 | **5.28** | 3.33 | 4.84 |
| `fold4_v01_dependent_tail` | 7.93 | **5.39** | 2.86 | 0.39 |

### Interpretation and experiment decision

- Longer groups tend to show lower spread for some templates, supporting a **measurement-window sensitivity hypothesis**, but improvement is **not monotonic**.
- The observed 256-sequence window meets the 5% stability criterion on all five **diagnostically selected** templates despite its minimum group being **below 100 ms**. Thus **100 ms is not yet established as a necessary threshold**.
- At 512 sequences all sampled groups exceeded 100 ms, but one template failed 5% by a small margin (5.09%). Thus >=100 ms is **not sufficient to guarantee** stability.
- Sample size **n=5 templates** and **three repeats per cell** limit confidence. These five were not randomly selected from the 60 formal templates; avoid extrapolating 5/5 to a 100% population acceptance rate.
- The existing V3 G0-only control had 39/60 stable and V3 clean O2 had 42/60; no V3 collection, trace or per-step verified label is changed by this pilot.
- **Next recommended test:** preregister an independent, randomized **60-template G0-only window-length comparison** with 8, 256 and 512 sequences/group (60 x 3 x 3 = 540 G0-only runs), calculate repeat-spread, stable proportions and measured group durations per count. This is a **confirmatory ablation proposal**, not a claim of success. Only afterward select a calibration target/cap for a separately frozen adaptive protocol.
- **Do not discard** any old or clean O2 runs until the new intended use is validated and a verified remote archive exists.


## Implemented next step: 60-template randomized window comparison

The separate module [`o2_sequence/g0_window_ablation.py`](../o2_sequence/g0_window_ablation.py) implements the preregistered 540-run G0-only comparison; it **reuses the exact V4 pilot measurement worker** and logs both the independent `design_protocol` and the worker `measurement_protocol`. It does **not** edit or rerun V3 P/R.

- Full 60 frozen V3 O2 formal templates, 30 independent and 30 dependent-tail.
- 8, 256 and 512 sequences/group, five groups/run, three independent repeats; **540 G0-only jobs**.
- Deterministically shuffled global job order (seed 20261101), with no profiler.
- Session-safe output path: refuses existing directories.
- Outputs `analysis/window_quality.csv` and `analysis/summary.json` with stable count and median repeat spread by count, minimum/median observed group duration, and paired FAIL→PASS/PASS→FAIL versus n=8.

**RTX3090 example (from existing branch checkout):**

```bash
conda activate gpu-prediction-o1
cd ~/RTchen/gpu-prediction-o2-g0-v4
git status --short
git pull --ff-only origin o2-g0-window-pilot-20261101
python -m unittest discover -s tests -p 'test_o2_g0_window_ablation.py' -v
export CUDA_VISIBLE_DEVICES=0
OUT="$HOME/RTchen/o2-runs/rtx3090_o2_g0_window_ablation_20261101_001"
python -m o2_sequence.g0_window_ablation run --expect-gpu 3090 --out "$OUT"
python -m o2_sequence.g0_window_ablation review --root "$OUT"
```

The source checkout must be clean, and the node's GPU should be idle before the run. Use tmux for persistence. Archive data and verify SHA-256 on the intended backup destination **before** deleting any old O2 or other data. This is a G0 timing study, **not** a fix for O2 V3's 0 verified per-step reference labels.

**Expected result semantics:** The runner can report collection 540/540 PASS and then print stability per window size; stable counts and minimum windows are empirical and **not** guaranteed. The confirmatory output should drive a subsequent decision about whether to freeze an adaptive `>=100 ms` calibration policy or adopt a fixed length for timing-only labels.


## RTX3090 60-template confirmation results (2026-11-01)

**Source:** On-node terminal output provided by the experiment operator. The raw `window_quality.csv` and `summary.json` are stored on RTX3090, not committed to GitHub. Raw path: `/home/iccl/RTchen/o2-runs/rtx3090_o2_g0_window_ablation_20261101_001/analysis`.

**Collection:** 540/540 completed; 0 failed. The test compares 60 frozen formal O2 templates, 3 G0 group-window settings (8, 256, 512 sequences/group), and 3 independent repetitions each. Each run has 5 synchronized groups. The original O2 V3 G0/P/R data and P/R labels are not changed.

| Sequences/group | Stable templates (3-repeat spread <=5%) | Pass rate | Median repeat spread | Median group duration | Minimum observed group duration |
|---:|---:|---:|---:|---:|---:|
| 8 | **34/60** | **56.7%** | 4.29% | 2.02 ms | 1.45 ms |
| 256 | **54/60** | **90.0%** | 2.47% | 63.28 ms | 45.18 ms |
| 512 | **55/60** | **91.7%** | 2.37% | 126.30 ms | 90.26 ms |

### Paired changes versus n=8

| Compared window | FAIL -> PASS | PASS -> FAIL | Net stable-template change |
|---|---:|---:|---:|
| n=256 | 24 | 4 | +20 |
| n=512 | 24 | 3 | +21 |

### Decision and limitations

1. The full-template experiment **supports the hypothesis that longer G0 batch windows improve host timing repeat stability** on this RTX3090 measurement environment. This is an association under the experimental protocol, not isolated proof that CPU scheduler jitter or a specific hardware mechanism caused the previous variability.
2. **The additional gain from n=256 to n=512 is small:** +1/60 stable template and median repeat spread 2.47% -> 2.37%, whereas median group duration is nearly doubled (63.28 -> 126.30 ms). Treat 256 as a sensible efficiency/stability *candidate*, not a final frozen choice.
3. **Neither setting is universally stable:** n=256 failed 6 templates; n=512 failed 5. Minimum group duration at n=512 was **90.26 ms**, so the original proposed >=100 ms target is *not met for every measured group*.
4. Repeating longer batches changes measurement policy and can change thermal, frequency, cache, submission/queue effects. These measurements have a distinct protocol and are **not drop-in replacements** for O2 V3 8-sequence G0 data or V3 P/R trace reference labels.
5. Stability is evaluated from **three independent repeats per template per window**, a small sample. The independent/related outcomes are paired by template but run jobs were shuffled in time; full per-template inspection and repeated session checks are appropriate before claiming a generalizable calibration rule.
6. The original O2 V3 strict per-step reference completeness remains unresolved (`verified_steps=0` because all 180 P/R pairs had profiler warnings). This window-only experiment neither improves nor invalidates V3 P/R structural evidence.

**Proposed next step, not yet executed:** develop a separate, versioned *template-specific G0 calibration* protocol. Use a calibration phase to choose calls/group based on observed host group duration (e.g., target 100-125 ms and an explicit maximum), freeze counts, then run **new** three-repeat formal validation and evaluate the same threshold and raw group windows. Do not use formal validation repeats to tune the selected counts. An alternative is to formally adopt fixed n=256 for an efficiency-centered timing endpoint, with the observed 54/60 repeat-spread result disclosed rather than described as a complete pass.

**Data lifecycle:** keep both old and new V3 O2, the G0-only control, the five-template pilot and this 60-template ablation until the chosen G0 target is finalized, the experiment artifacts have been fully packaged, and a remote SHA-256 check has succeeded. Never delete O2 P/R traces in a G0-only cleanup.
