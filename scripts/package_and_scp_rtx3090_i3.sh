#!/usr/bin/env bash
# RTX3090 O4/I3 Core36 + S3 + S4 reproducible archival and verified SCP.
# Runs on the RTX3090 node. Never modifies/erases source collections.
# Example:
#   RTX3090_I3_REMOTE='user@host' RTX3090_I3_REMOTE_DIR='/path' bash scripts/package_and_scp_rtx3090_i3.sh
set -euo pipefail

: "${RTX3090_I3_REMOTE:?Set RTX3090_I3_REMOTE=user@host (never commit host credentials)}"
: "${RTX3090_I3_REMOTE_DIR:?Set RTX3090_I3_REMOTE_DIR=/absolute/destination}"

ROOT="${RTX3090_I3_DATA_ROOT:-$HOME/RTchen/i3-data-3090}"
OUTROOT="$ROOT/exports/RTX3090_O4_I3_20261107"
REMOTE="$RTX3090_I3_REMOTE"
REMOTE_DIR="$RTX3090_I3_REMOTE_DIR"
S4_REPO="${RTX3090_I3_S4_REPO:-$HOME/RTchen/gpu-prediction-i3-s4}"

if [[ "${RTX3090_I3_S4_ONLY:-0}" == "1" ]]; then
  DATASETS=(S4_scale0_3090_01)
else
  DATASETS=(Core36_S1_smoke_PBfZ0zgK S3_lite_B3FiyVTe S4_scale0_3090_01)
fi

for cmd in tar gzip sha256sum ssh scp python3 df du awk; do
  command -v "$cmd" >/dev/null || { echo "Missing dependency: $cmd" >&2; exit 1; }
done

# Confirm the released code and actual on-node S4 final audit before archiving.
test "$(git -C "$S4_REPO" rev-parse HEAD)" = "3db5871d43abd87fefbdadc21371beee59ac6b88"
python3 - "$ROOT/S4_scale0_3090_01/analysis/final_s4_scale0_3090_v1/SUMMARY.json" <<'PY'
import json, sys
from pathlib import Path
d=json.loads(Path(sys.argv[1]).read_text())
for field,value in {
  "status":"PASS","work_items":40,"smoke_cells":120,
  "formal_cells":360,"formal_measured":14400,
  "formal_exact_PR_pairs":120,"R_parameter_mismatches":0,"batch_issues":0
}.items():
    assert d.get(field)==value,(field,d.get(field),value)
print("RTX3090 S4 AUDIT: PASS")
PY

if [[ "$REMOTE_DIR" != /* ]]; then echo "Remote destination must be absolute" >&2; exit 1; fi
echo "===== VERIFY SSH DESTINATION ====="
ssh "$REMOTE" 'printf "SSH CONNECTION OK\\n"'

echo "===== SOURCE DATASET SIZES ====="
for name in "${DATASETS[@]}"; do
  test -d "$ROOT/$name" || { echo "Missing dataset: $ROOT/$name" >&2; exit 1; }
  du -sh "$ROOT/$name"
done
df -h "$ROOT"

mkdir -p "$OUTROOT"

for name in "${DATASETS[@]}"; do
  prefix="RTX3090_${name}_20261107"
  pkg="$OUTROOT/$prefix"
  source_dir="$ROOT/$name"
  lite="$prefix.analysis.tar.gz"
  full="$prefix.full.tar.gz"

  echo "===== PACKAGE $name ====="

  # Fail closed on possible credential material; the operator can review and
  # prepare a new, separately documented safe archive without editing raw data.
  suspect="$(find "$source_dir" -type f \( -name '.env' -o -name 'id_rsa' -o -name '*.pem' -o -name 'credentials.json' \) -print -quit)"
  if [[ -n "$suspect" ]]; then
    echo "STOP: Possible private material in archive: $suspect" >&2
    exit 1
  fi

  if [[ -e "$pkg" ]]; then
    echo "STOP: Existing package path (no overwrites): $pkg" >&2
    exit 1
  fi

  # Conservative space guard. Archives are compressed but do not assume any
  # particular compression ratio. Never delete existing data to make space.
  source_kb="$(du -sk "$source_dir" | awk '{print $1}')"
  free_kb="$(df -Pk "$ROOT" | awk 'END {print $4}')"
  needed_kb="$((source_kb + 1048576))"
  if (( free_kb < needed_kb )); then
    echo "STOP: Insufficient conservative local free-space budget for $name." >&2
    echo "Source KiB=$source_kb, available KiB=$free_kb. Use a reviewed streaming workflow instead." >&2
    exit 1
  fi

  mkdir "$pkg"

  echo "===== CREATE ANALYSIS-LITE ARCHIVE: $name ====="
  # This contains inventories, audit, R positions, source snapshots, assets,
  # normalized traces, quality and summaries; raw binary traces are in FULL.
  tar -C "$ROOT" --exclude='*.nsys-rep' --exclude='*.sqlite' \
    --exclude='*.sqlite-shm' --exclude='*.sqlite-wal' \
    -czf "$pkg/$lite.partial" "$name"
  mv "$pkg/$lite.partial" "$pkg/$lite"

  echo "===== CREATE FULL RAW ARCHIVE: $name ====="
  tar -C "$ROOT" -czf "$pkg/$full.partial" "$name"
  mv "$pkg/$full.partial" "$pkg/$full"

  echo "===== VERIFY LOCAL ARCHIVES ====="
  tar -tzf "$pkg/$lite" >/dev/null
  tar -tzf "$pkg/$full" >/dev/null

  (
    cd "$pkg"
    sha256sum "$lite" "$full" > SHA256SUMS
    sha256sum -c SHA256SUMS
  )

  python3 - "$pkg" "$name" "$lite" "$full" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);name,lite,full=sys.argv[2:]
sha={}
for line in (p/"SHA256SUMS").read_text().splitlines():
    digest,filename=line.split(None,1)
    sha[filename.lstrip("*")]=digest
manifest={
    "dataset":name,
    "source_node_gpu":"NVIDIA GeForce RTX 3090",
    "source_release":"o4-i3-s4-scale0-dis-dev-v1-20261107" if name.startswith("S4_") else "see original collection's source_snapshot and protocol",
    "archive_policy":"FULL includes raw Nsight/SQLite; ANALYSIS excludes binary traces but retains metadata and summaries",
    "files":{n:{"bytes":(p/n).stat().st_size,"sha256":sha[n]} for n in (lite,full)},
    "raw_immutable":True,
    "contents_root":name,
    "model_test_scores_not_generated":True
}
(p/"MANIFEST.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
print("Manifest:",p/"MANIFEST.json")
PY

  # Never overwrite a remote completed archive or a prior failed upload.
  echo "===== PREPARE REMOTE UPLOAD STAGING: $name ====="
  ssh "$REMOTE" bash -s -- "$REMOTE_DIR" "$prefix" <<'REMOTE'
set -euo pipefail
root="$1";name="$2"
mkdir -p "$root"
test ! -e "$root/$name" || { echo "STOP: Remote final archive already exists" >&2; exit 1; }
test ! -e "$root/$name.uploading" || { echo "STOP: Remote partial upload exists; inspect before retry" >&2; exit 1; }
mkdir "$root/$name.uploading"
REMOTE

  echo "===== SCP TO GB10: $name ====="
  scp "$pkg/$lite" "$pkg/$full" "$pkg/SHA256SUMS" "$pkg/MANIFEST.json" \
    "$REMOTE:$REMOTE_DIR/$prefix.uploading/"

  echo "===== VERIFY REMOTE SHA256 AND FINALIZE: $name ====="
  ssh "$REMOTE" bash -s -- "$REMOTE_DIR" "$prefix" <<'REMOTE'
set -euo pipefail
root="$1";name="$2"
partial="$root/$name.uploading"
final="$root/$name"
test -d "$partial"
test ! -e "$final"
(
  cd "$partial"
  sha256sum -c SHA256SUMS
  test -s MANIFEST.json
)
mv -T "$partial" "$final"
printf 'REMOTE VERIFIED: %s\n' "$final"
REMOTE

  echo "TRANSFER PASS: $name"
done

echo "===== RTX3090 ARCHIVE AND VERIFIED SCP COMPLETE ====="
echo "LOCAL ARCHIVE ROOT: $OUTROOT"
echo "GB10 ARCHIVE ROOT: $REMOTE:$REMOTE_DIR"
echo "Original RTX3090 data has not been altered or removed."
