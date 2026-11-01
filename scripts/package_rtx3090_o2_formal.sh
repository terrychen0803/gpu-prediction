#!/usr/bin/env bash
# RTX3090: pack verified clean O2 V3 and formal G0 V4 including calibration/analysis.
# No deletion.
set -Eeuo pipefail
ROOT="$HOME/RTchen"
OUT="$ROOT/transfer_archives"
REMOTE="good_egg@140.113.179.4"
DEST="/home/good_egg/RTchen/RTX3090_archives"
NAME="RTX3090_O2_CleanV3_AdaptiveV4_Formal_20261101"
mkdir -p "$OUT"
SOURCES=(
 o2-runs/rtx3090_o2_clean_20261101_001_pilot
 o2-runs/rtx3090_o2_clean_20261101_001_formal
 o2-review/rtx3090_o2_clean_20261101_001_pilot
 o2-review/rtx3090_o2_clean_20261101_001_formal
 o2-review/rtx3090_o2_clean_20261101_001_quality
 o2-review/rtx3090_o2_clean_20261101_001_collection.log
 o2-runs/rtx3090_o2_g0_window_ablation_20261101_001
 o2-runs/rtx3090_o2_g0_adaptive_125ms_20261101_001
)
for p in "${SOURCES[@]}"; do
  test -e "$ROOT/$p" && test ! -L "$ROOT/$p" || { echo "ERROR missing $p"; exit 1; }
done
if command -v zstd >/dev/null; then
  ARCH="$OUT/$NAME.tar.zst"
  COMPRESSION=(--zstd)
else
  ARCH="$OUT/$NAME.tar.gz"
  COMPRESSION=(-z)
fi
SUM="$ARCH.sha256"
MANIFEST="$OUT/$NAME.manifest.txt"
LIST="$OUT/$NAME.contents.txt"
for p in "$ARCH" "$SUM" "$MANIFEST" "$LIST"; do
  test ! -e "$p" || { echo "ERROR output exists $p"; exit 1; }
done

python3 - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
root=Path(sys.argv[1])
def read(x):return json.loads(x.read_text())
def check(x,msg):
 if not x:raise SystemExit("ERROR: "+msg)
v=root/"o2-runs/rtx3090_o2_clean_20261101_001_formal"
r=root/"o2-review/rtx3090_o2_clean_20261101_001_formal"
a=root/"o2-runs/rtx3090_o2_g0_window_ablation_20261101_001"
d=root/"o2-runs/rtx3090_o2_g0_adaptive_125ms_20261101_001"
p=read(v/"plan.json");s=read(r/"summary.json")
check(p["status"]=="completed" and p["failure_count"]==0 and len(p["jobs"])==540,"V3 Formal")
check(s["pairs"]==180 and s["exact_signature_pairs"]==180 and s["missing_pairs"]==0,"V3 P/R")
check(len(list((v/"runs").rglob("trace.nsys-rep")))==360,"V3 missing Nsight reports")
check(len(list((v/"runs").rglob("trace.sqlite")))==360,"V3 missing SQLite")
p=read(a/"plan.json");s=read(a/"analysis/summary.json")
check(p["status"]=="completed" and len(p["jobs"])==540 and p["failure_count"]==0,"V4 ablation")
check(s["completed"]==540,"V4 ablation review")
bp=read(d/"bootstrap_plan.json");fp=read(d/"formal_plan.json");s=read(d/"analysis/summary.json")
check(bp["status"]=="completed" and len(bp["jobs"])==120 and bp["failure_count"]==0,"Adaptive bootstrap")
check(fp["status"]=="completed" and len(fp["jobs"])==180 and fp["failure_count"]==0,"Adaptive formal")
check(s["stable_5pct"]==56 and s["all_groups_ge_100ms_templates"]==60,"Adaptive quality")
print("All required V3/V4 formal results and calibration evidence verified.")
PY
(cd "$ROOT/o2-runs/rtx3090_o2_g0_adaptive_125ms_20261101_001" && sha256sum -c calibration.sha256)

{
 echo "RTX3090 O2 full formal data: Clean V3, V4 60-template ablation and Adaptive G0"
 echo "V3: 540/540 complete; P/R 180/180; 0 verified steps due to NVTX warnings"
 echo "V4 ablation: 540/540, stable 34/60 54/60 55/60 at 8/256/512"
 echo "V4 adaptive: bootstrap 120/120; formal 180/180; stable 56/60; >=100ms 60/60"
 echo "V3 commit: 77ccb5db59a359cdba1ef61bc54c48c33ffc28bb"
 echo "V4 branch: o2-g0-window-pilot-20261101"
 echo "UTC: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
 echo "Included:"
 printf '%s\n' "${SOURCES[@]}"
} > "$MANIFEST"

tar "${COMPRESSION[@]}" -cf "$ARCH" -C "$ROOT" "${SOURCES[@]}" -C "$OUT" "$(basename "$MANIFEST")"
tar "${COMPRESSION[@]}" -tf "$ARCH" > "$LIST"
for p in "${SOURCES[@]}"; do
  grep -Fxq "$p" "$LIST" || grep -Fxq "$p/" "$LIST" || { echo "Archive missing $p"; exit 1; }
done
(cd "$OUT" && sha256sum "$(basename "$ARCH")" > "$(basename "$SUM")" && sha256sum -c "$(basename "$SUM")")
ls -lh "$ARCH"
ssh "$REMOTE" "mkdir -p '$DEST'"
BASE="$(basename "$ARCH")"
HASH="$(cut -d' ' -f1 "$SUM")"
if ssh "$REMOTE" "test -e '$DEST/$BASE'"; then
  R_HASH="$(ssh "$REMOTE" "sha256sum '$DEST/$BASE'" | awk '{print $1}')"
  test "$R_HASH" = "$HASH" || { echo "Remote archive exists with different hash"; exit 1; }
else
  TEMP="$DEST/.$BASE.upload.$$.part"
  scp -p "$ARCH" "$REMOTE:$TEMP"
  R_HASH="$(ssh "$REMOTE" "sha256sum '$TEMP'" | awk '{print $1}')"
  test "$R_HASH" = "$HASH" || { echo "Remote upload checksum mismatch"; exit 1; }
  ssh "$REMOTE" "test ! -e '$DEST/$BASE' && mv '$TEMP' '$DEST/$BASE'"
fi
scp -p "$SUM" "$MANIFEST" "$REMOTE:$DEST/"
ssh "$REMOTE" "cd '$DEST' && sha256sum -c '$(basename "$SUM")'"
echo "BACKUP VERIFIED. No local or remote data deleted."
