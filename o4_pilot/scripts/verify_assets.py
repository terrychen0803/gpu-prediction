"""Portable read-only asset and source verification; no CUDA import."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from o4_iteration.common import ROOT,read,sha,code_manifest
assets=read(ROOT/'manifests/assets.json');items=list(assets['resnet'].values())
for files in assets['operators'].values():items.extend(files.values())
for item in items:
 path=ROOT/item['path']
 if not path.is_file() or sha(path)!=item['sha256']:raise RuntimeError('Missing/corrupted shared asset: '+str(path))
unified=ROOT/'manifests/unified_release.json'
if unified.exists():
 from o4_research.provenance import verify_release
 verify_release()
expected=ROOT/'manifests/code_release.json'
if expected.exists() and not unified.exists():
 manifest=read(expected)
 for relative,digest in manifest['files'].items():
  if sha(ROOT/relative)!=digest:raise RuntimeError('Release file mismatch: '+relative)
print(f'PASS: {len(items)} shared asset hashes; release code checked when available; no CUDA work')
