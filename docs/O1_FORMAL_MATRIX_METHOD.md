# O1 formal 139-config method

This branch extends the published v4 pilot with a separate formal protocol. The new material contains code, the predefined configuration catalog, and a split manifest. New formal calibration manifests and trace data stay on each node; historical v4 pilot material is inherited from the base branch.

The catalog contains GEMM 27, BMM 24, Conv2D 64, Add 12, and LayerNorm 12 configurations. The first three families form the 115-config core. All inputs use the existing seed-42 SHAKE256-to-FP16 format. The execution policy is FP16 eager on one CUDA stream, with no autograd, autocast, CUDA Graph, input transfer, or allocation setup inside measured groups. CPU FP32 results converted to FP16 provide the correctness reference.

The split manifest is frozen before measurement. It groups all captures, modes, repeats, and devices of the same canonical configuration. The pilot-overlap Conv configuration is development-only and excluded from every formal test fold. Apply the same manifest on every device; do not make a new split after seeing labels.

## Per-node sequence

Use a checkout of this branch and a new output path for each command. Replace `GB10` with the GPU name fragment actually discovered on that node. Each node must build its **own** calibration file; do not copy another node's call counts.

```sh
python -m unittest discover -s tests -v
python -m o1_operator run --matrix formal --bootstrap --modes G0 GE P --repeats 1 \
  --expect-gpu GB10 --out /path/to/new_formal_bootstrap
python -m o1_operator calibrate --matrix formal \
  --g0ge-root /path/to/new_formal_bootstrap \
  --p-root /path/to/new_formal_bootstrap \
  --out /path/to/new_formal_calibration.json
python -m o1_operator run --matrix formal --modes G0 GE P R --repeats 3 \
  --expect-gpu GB10 --calibration-manifest /path/to/new_formal_calibration.json \
  --out /path/to/new_formal_full
python -m o1_operator summarize --root /path/to/new_formal_full \
  --out /path/to/new_formal_analysis
```

Bootstrap protocol: `o1_formal_139_bootstrap_8192_v2_20261022`. It runs five groups of 8192 calls for each G0, GE, and P cell. The calibrated protocol is `o1_formal_139_calibrated_v2_20261022`; it has 139 × four modes × three fresh processes = 1668 runs. Each mode's calls per group derive from that node's shortest bootstrap group as `ceil(8192 × 125 ms / minimum_group_ms / 256) × 256`, capped at 65536. R uses the same fixed calls as P. Warm-up requires at least 100 calls and 100 ms, with a 100000-call and 45-second limit. Every measured run has five groups and targets at least 100 ms per group.

Check correctness, group durations, and G0/GE three-run spread `(max−min)/median ≤ 5%`. Review P/R fixed-call equality, CUDA launch-to-kernel mapping, R reference scope inventory, and trace warnings. The summary preserves failures and marks warning-bearing P/R pairs `manual_review_required_event_loss`. Continuing collection with warnings does not validate reference labels or event completeness. Keep G0 host completion, GE single-call events, GE batch events, and P/R trace proxies separate. Do not combine this protocol with v2/v4 pilot timing data.

The raw session includes the copied calibration manifest, SHA-256 provenance, fixed inputs, per-run logs, timings, traces, and audits. Keep raw traces, node-specific calibration, and analysis outputs in the node's artifact area. Compare devices only after each node passes its own quality review; report missing or incompatible cells rather than changing shapes or tolerances silently.
