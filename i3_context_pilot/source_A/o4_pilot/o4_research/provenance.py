"""Full distribution hashes and safe environment discovery; no environment dump."""
from pathlib import Path
import importlib.metadata,platform,subprocess,sys
from .common import sha
ROOT=Path(__file__).resolve().parents[1]
SOURCE_SUFFIXES={'.py','.sh','.cu','.cpp','.h','.md','.json','.txt','.csv'}
EXCLUDED={'assets','analysis_runs','artifacts','results','.git','__pycache__','.pytest_cache'}

def source_hashes(root=ROOT):
    return {str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*'))
            if p.is_file() and not (set(p.relative_to(root).parts)&EXCLUDED)
            and p.suffix in SOURCE_SUFFIXES and p.name!='unified_release.json'}

def query(argv):
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=20)
        return {'command':argv,'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as e:return {'command':argv,'unavailable':type(e).__name__}

def snapshot():
    packages={}
    for name in ('torch','torchvision','transformers','ultralytics','numpy','safetensors','tokenizers','huggingface-hub'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]='unavailable'
    return {'python':sys.version,'machine':platform.machine(),'platform':platform.platform(),
            'packages':packages,'git_commit':query(['git','-C',str(ROOT),'rev-parse','HEAD']),
            'git_status':query(['git','-C',str(ROOT),'status','--porcelain']),
            'nsys':query(['nsys','--version']),'nvcc':query(['nvcc','--version']),
            'gpu':query(['nvidia-smi','--query-gpu=name,driver_version','--format=csv,noheader']),
            'environment_variables_dumped':False,'GPU_work_launched':False}

def verify_release(root=ROOT):
    import json
    manifest=json.loads((root/'manifests/unified_release.json').read_text())
    actual=source_hashes(root)
    if actual!=manifest['files']:
        changed=sorted(k for k in actual.keys()|manifest['files'].keys() if actual.get(k)!=manifest['files'].get(k))
        raise ValueError('Unified source release mismatch: '+', '.join(changed))
    return {'release_id':manifest['release_id'],'files_verified':len(actual),'source_tree_sha256':sha(root/'manifests/unified_release.json')}
