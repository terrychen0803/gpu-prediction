"""One node entry point: offline verification/planning, explicit bounded collection."""
import argparse,os,subprocess,sys
from pathlib import Path
from .common import read,write,sha,gpu_guard
from .provenance import ROOT,verify_release,snapshot
from .verify_assets import verify
ASSET_SHA='55300ee1eb51a481421a7318ec8fbf9c1b2bffdce9e31d77d37f7383cf69cec4'

def verify_all_assets(manifest):
    manifest=Path(manifest).resolve()
    verify(manifest,ASSET_SHA)
    index=read(ROOT/'manifests/a_inventory_file_sha256_20261106.json')
    for name,digest in index.items():
        p=manifest.parent/name
        if not p.is_file() or sha(p)!=digest:raise ValueError('Full asset index mismatch: '+name)
    return {'manifest_sha256':ASSET_SHA,'full_asset_files_verified':len(index)}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['verify','preflight','plan','smoke','collect','native','replay'])
    p.add_argument('--gpu',choices=['GB10','3090','4090'],required=True)
    p.add_argument('--assets-manifest',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--workload',nargs='+',default=['RN-A'])
    p.add_argument('--runs',nargs='+')
    a=p.parse_args();out=Path(a.out).resolve()
    if out.exists():raise FileExistsError('Use a new output directory: '+str(out))
    if ROOT.parent==out or ROOT.parent in out.parents:raise ValueError('Keep node outputs outside the source checkout')
    release=verify_release();assets=verify_all_assets(a.assets_manifest)
    if a.action in ('verify','preflight'):
        out.mkdir(parents=True)
        report={'source_release':release,'assets':assets,'environment':snapshot(),'GPU_execution':False,'strict_verified':0}
        write(out/'PREFLIGHT.json',report)
        if a.action=='preflight':gpu_guard(out,a.gpu)
        print('PASS source/assets; no CUDA workload launched:',out);return 0
    base=[sys.executable,'-m']
    if a.action=='native':
        cmd=['bash',str(ROOT/'native_control/run_native_node.sh'),str(out),a.gpu]
    elif a.action=='replay':
        if not a.runs:p.error('replay needs --runs')
        cmd=base+['o4_research.replay','--runs',*a.runs,'--out',str(out)]
    else:
        module='o4_research.run_smoke' if a.action=='smoke' else 'o4_research.run_formal'
        cmd=base+[module,'--out',str(out),'--expect-gpu',a.gpu,'--assets-manifest',str(Path(a.assets_manifest).resolve()),'--workload',*a.workload]
        if a.action!='smoke':cmd+=['--expected-manifest-sha256',ASSET_SHA]
        if a.action=='plan':cmd+=['--plan-only']
    proc=subprocess.run(cmd,cwd=ROOT,env={**os.environ,"PYTHON":sys.executable})
    if proc.returncode:return proc.returncode
    if a.action=='smoke':
        from .audit_v5 import audit
        reviews=[]
        for job in read(out/'inventory.json')['jobs']:
            if job['mode']=='GI':continue
            folder=out/job['folder']
            reviews.append({'cell':job['folder'],'acceptance':audit(folder,folder/'unified_audit')['engineering_acceptance']})
        write(out/'UNIFIED_SMOKE_REVIEW.json',{'protocol':'additive_engineering_overlay_v1','reviews':reviews,'strict_verified':0})
        if any(not x['acceptance']['accepted'] for x in reviews):return 1
    return 0

if __name__=='__main__':raise SystemExit(main())
