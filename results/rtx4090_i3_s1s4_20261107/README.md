# RTX4090 O4 I3 S1–S4 Engineering Collection — 2026-11-07

This branch records the RTX4090 engineering collection **metadata** for the frozen
`o4-i3-s4-scale0-dis-dev-v1-20261107` source. Measurements and their
raw Nsight `.nsys-rep`/`.sqlite` files remain on the RTX4090 host until
separately archived and transferred; **this branch does not contain the raw files**.

## Reported collection results

| Stage | Selected complete formal cells | Notable conditions |
|---|---:|---|
| S1 | 27/27 | Three base configurations |
| S2 | 297/297 | Core 33 configurations |
| S3 | 324/324 | The original capture has 316 completed, 2 failed and 6 deferred; a separate 9-cell Groups4 recovery supersedes one original duplicate GI repeat |
| S4-A | 126/126 | CBR 45; Residual 81 |
| S4-BC | 234/234 | Depthwise-Pointwise 54; SE Gate 72; Resize-Concat-Conv 54; Attention Eager 54 |
| **Total** | **1,008/1,008** | **40,320 measured iteration samples**, selected without duplicates |

S4 P/R ordered signatures: **120/120 exact pairs**. S4 mode-group stability:
**91 stable / 29 review_required**. The fact that P/R ordered signatures match
does not establish complete CUDA-workground-truth coverage.

## S3 Groups4 recovery and interpretation

Original `I3_conv_00_groups4` P/R capture contained 180 GPU records across four
CUDA streams, zero correlation errors and zero unscoped GPU records. The
original single-stream engineering gate rejected those captures.
A separate, strictly scoped `groups4_multistream_conditional_v2` collector
completed 9/9 cells, with P/R 3/3 exact, and all three GI/P/R mode groups stable.
The original failed/raw files are retained. See:

- `s3_lite_20261107T075927Z/collections/train_controls_formal`
- `s3_lite_20261107T075927Z/collections/groups4_multistream_v2_formal`
- `s3_lite_20261107T075927Z/analysis/S3_groups4_versioned_recovery_manifest.json`
- `s3_lite_20261107T075927Z/groups4_multistream_v2_source`

**Caveats:** Groups4 P/R are conditional, multistream recorded-structure evidence.
GI's all-stream completion is **not verified**. Protocol versions are mixed for
this controlled exception. All experiments remain engineering/development data,
`strict_reference_verified=false` and `formal_training_eligible=false`.

## Archive and GB10 analysis plan

Source RTX4090 collection roots:

- `/home/iccl_resilient/RTchen/i3-data-4090/core36_20261107T072354Z`
- `/home/iccl_resilient/RTchen/i3-data-4090/s3_lite_20261107T075927Z`
- `/home/iccl_resilient/RTchen/i3-data-4090/s4_scale0_20261107T092142Z`

Target GB10 archive directory:
`/home/good_egg/RTchen/RTX4090_archives` on `good_egg@140.113.179.4`.

Make four `.tar.gz` archives: Core36; S3 including original failure and
versioned recovery; S4 with per-device locally derived assets; and frozen
source/assets together with `fix_i3_groups4_4090.sh`. Copy
`UNIFIED_DATASET_INDEX.json`, `PROVENANCE.txt`, and `SHA256SUMS` alongside.
Transfer over SCP, and verify `sha256sum -c SHA256SUMS` on GB10 before extracting.

**Reconstruction:** Preserve complete raw artifacts and select 1,008 unique
`(config_id, repeat, mode)` cells. For `I3_conv_00_groups4`, prefer all nine
versioned recovery cells, including its GI Repeat 1, over the original duplicate.
Do not erase the original trace or failure diagnostics. Do not infer unseen
cross-device kernel equivalence from name matches alone.

## Publication status

- This GitHub branch stores the **reported metadata**, not the raw archive.
- SCP and archive checksum verification require executing commands on the RTX4090.
- Exact archive SHA256 values must be written from the generated files.
- Large raw captures are better suited to GB10 archival storage or GitHub
  Release assets, not normal Git commits.
