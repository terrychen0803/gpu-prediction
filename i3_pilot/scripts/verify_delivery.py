"""Verify immutable source files before invoking the portable collector."""
import hashlib,json
from pathlib import Path

def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'manifests/FROZEN_SOURCE_SHA256.json').read_text())
    seen=set()
    for row in manifest['files']:
        rel=Path(row['path']);f=(root/rel).resolve()
        if rel.is_absolute() or root not in f.parents or row['path'] in seen:
            raise ValueError('Invalid/duplicate source path: '+row['path'])
        seen.add(row['path'])
        if not f.is_file() or hashlib.sha256(f.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('Frozen I3 source changed or missing: '+str(f))
    actual={str(f.relative_to(root)) for f in (root/'source').rglob('*')
            if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
    if actual!=seen:raise ValueError('Unexpected or missing files in frozen source/')
    return {'frozen_source_files_verified':len(seen),'GPU_execution':False}

if __name__=='__main__':
    print(json.dumps(verify(Path(__file__).resolve().parents[1]),indent=2))
