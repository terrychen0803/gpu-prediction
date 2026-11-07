# Publishing the RTX4090 raw archives

The GitHub connector has already written `SUMMARY.json` and `README.md` on
this result branch. The raw files must be uploaded **from the machine that
holds them**, after the archive and SHA256 validation are complete.

## Step 1 — Archive on RTX4090 and verify SCP

Run the supplied `package_scp_rtx4090_i3_s1s4_20261107.sh` from the RTX4090
host. It builds four tar.gz archives plus `SHA256SUMS`,
`PROVENANCE.txt`, `README_TRANSFER.txt` and
`UNIFIED_DATASET_INDEX.json`. The script's SCP step creates a timestamped
subdirectory beneath:

`good_egg@140.113.179.4:/home/good_egg/RTchen/RTX4090_archives`

A remote `sha256sum -c SHA256SUMS` must pass before analysis.

## Step 2 — Upload raw archives as GitHub Release assets (optional)

Git commits are for the small metadata and reproducible code; large raw
`.nsys-rep` and SQLite datasets should be uploaded as **Release assets**,
not as ordinary Git objects.

On GB10, using a separately installed/authenticated GitHub CLI:

```bash
gh auth status
ARCHIVE_DIR=/home/good_egg/RTchen/RTX4090_archives/RTX4090_I3_S1S4_<UTC_TIMESTAMP>
cd "$ARCHIVE_DIR"
sha256sum -c SHA256SUMS

gh release create rtx4090-o4-i3-s1s4-engineering-20261107 \
  RTX4090_I3_*.tar.gz SHA256SUMS \
  UNIFIED_DATASET_INDEX.json PROVENANCE.txt README_TRANSFER.txt \
  --repo terrychen0803/gpu-prediction \
  --target results/rtx4090-o4-i3-s1-s4-20261107 \
  --prerelease \
  --title "RTX4090 I3 S1–S4 engineering data (Groups4 v2 recovery)" \
  --notes "Engineering collection only; strict reference unverified. Groups4 multistream conditional P/R; GI all-stream completion unverified."
```

Do not put personal access tokens into scripts, console logs, or the repository;
authenticate `gh` separately. If the release tag already exists, use
`gh release upload TAG ... --clobber` only after intentional checksum/version
review, or use a new immutable release tag. Do not mark a release created until
its assets are visible and SHA256-verified.

## Selection and limitations

The original Groups4 failed and deferred raw data are preserved. The recovery
batch provides all nine cells; the original GI repeat 1 is intentionally
superseded in the canonical 1,008-cell dataset index. The original capture
must not be silently modified. Versioned conditional multistream evidence is
not equivalent to verified GI all-stream completion.
