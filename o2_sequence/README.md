# O2 mixed sequence collection (GB10 protocol v3)

For RTX 3090/4090 and GB10 node commands, frozen hashes, and quality gates,
see the [O2 node runbook](../docs/O2_NODE_RUNBOOK.md). The protocol ID records
the first measured node; use `--expect-gpu` for each device and keep the same
collection code and catalog when comparing nodes.

This package collects the O2 **sequence data**, not the T-v1 model or a claim of
validated per-operator labels. It uses the O1 fixed FP16 input generator, O1
five-fold split, one CUDA stream, eager execution, preallocated inputs and
three disjoint modes:

- **G0**: whole sequence host batch time, no Nsight profiler. Each of five
  groups contains eight complete sequences and one synchronization at the end.
- **P**: CUDA runtime/kernel trace with no per-step NVTX marker, event timer or
  synchronization. The predictor-facing `events_allowlist.jsonl` contains only
  capture-local event IDs, timings, kernel names, launch geometry and API
  correlations.
- **R**: same workload with an NVTX scope around each logical step. This is
  evaluator-only reference data; it is never substituted for a P trace.

There are 60 frozen formal templates in `formal_templates_v1.json`, 12 per
O1 fold. Every template has 16 independent calls; half add a four-step
GEMM/BMM tensor-dependent tail. Template composition was fixed from O1's
catalog and split, without GB10 timing selection. The two pilot templates
use the O1 pilot configurations and do not enter formal testing. Each formal
template runs G0/P/R three times, for 540 processes on one GPU. A run has five
groups of eight sequences. The one-process-per-cell policy preserves failures
and prevents CUDA state leaking between modes/repeats.

## Commands

Run from the repository root using an environment with PyTorch CUDA and
Nsight Systems. Create a **new** output directory each time:

```sh
python -m o2_sequence run --matrix pilot --repeats 1 --out /absolute/path/pilot
python -m o2_sequence review --root /absolute/path/pilot --out /absolute/path/pilot_review
python -m o2_sequence run --matrix formal --repeats 3 --out /absolute/path/formal
python -m o2_sequence review --root /absolute/path/formal --out /absolute/path/formal_review
```

`--template ID` and `--mode G0|P|R` select subsets for diagnostics. `--dry-run`
materializes a plan and requests without touching CUDA. `--expect-gpu GB10`
checks the visible device. Collection never uses sudo or GPU performance
counters. On machines where CUDA tracing requires root, execute the generated
worker/profiler command with appropriate local privileges; do not copy
credentials into run artifacts.

## Raw data and review

`plan.json` records protocol, source hash, input inventory, split/catalog
hashes, versions and every intended job. Each `runs/<template>/<mode>/runNN/`
contains request, command, environment, quality, group timing and execution
status. P/R also preserve the original `.nsys-rep`, exported SQLite,
profiler/export logs, diagnostics and parsed trace audit. R keeps
`reference_operations.jsonl` with true step IDs/classes. P keeps only
`events_allowlist.jsonl`. Raw traces, inputs and review outputs belong under
`artifacts/runs` or `analysis_runs` and should not be committed.

`review` writes evaluator-only `rp_alignment.jsonl`, one row per logical step,
plus pair/aggregate summaries. It accepts a P join only when the entire
kernel sequence, launch geometry, stream and R GPU order match exactly;
neither absolute timestamps nor close durations are used for matching.
Nsight diagnostic warnings force `unverifiable` provenance even when signatures
match. Missing/changed paths remain in the coverage denominator. The
`exact_ordered_signature` flag is structural evidence, not an assertion that
every GPU event was captured. An R trace with NVTX collection warnings still
requires manual validation before calling P labels verified.

CUDA event timestamps in SQLite and the P allowlist are integer nanoseconds
relative to the first captured API/kernel event. Group CSV wall clock fields
are UTC ISO 8601 and host durations use `perf_counter_ns`. P and R have
independent clocks/captures and are compared only by ordered signatures.

## Known limitations

- G0 is a batch-amortized whole-sequence host target, not a single-call device
  event target. O2 does not collect per-operator GE labels.
- R NVTX warnings on GB10 may leave every P join at `unverifiable` until a
  separate manual or tool-level completeness check is available. Preserve the
  warning text and raw trace.
- The initial taxonomy covers GEMM, BMM and Conv. Add/LayerNorm, unknown and
  ambiguous-class challenges need separate preregistered collection.
- O2 collection does not implement segmentation training, five-fold model
  evaluation, inference costs or results for RTX 3090/4090.

Dependent tails are checked against a CPU FP32-then-FP16 reference at every
edge. The gate records relative L2 error and maximum absolute error divided
by the maximum reference magnitude; both must be at most 1%. A per-element
absolute tolerance is unsuitable for long FP16 chains near cancellation.
