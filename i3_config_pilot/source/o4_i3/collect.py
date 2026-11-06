"""Portable bounded smoke collector. No formal 40x3 or model training interface."""
import argparse,os,platform,sys,shutil
from pathlib import Path
from .controlled import verify_assets,sha
from .worker import PROTOCOL
from o4_research.common import read,write,now,command,gpu_guard

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--assets-manifest',required=True)
    p.add_argument('--config',nargs='+',required=True);p.add_argument('--mode',nargs='+',choices=['GI','P','R'],default=['GI','P','R'])
    p.add_argument('--expect-gpu',required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
    manifest=Path(a.assets_manifest).resolve();verified=verify_assets(manifest);configs={c['id'] for c in read(manifest)['configurations']}
    if not set(a.config)<=configs or len(set(a.config))!=len(a.config) or len(set(a.mode))!=len(a.mode):raise ValueError('Invalid/duplicate config or mode')
    out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'ASSET_VERIFICATION.json',verified)
    original_source=Path(__file__).resolve().parents[1]
    source=out/'source_snapshot';source.mkdir()
    for package in ('o4_i3','o4_pilot'):
        shutil.copytree(original_source/package,source/package,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files=[]
    for base in (source/'o4_i3',source/'o4_pilot'):
        for f in sorted(base.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py','.json','.sh','.cpp'):
                files.append({'path':str(f.relative_to(source)),'sha256':sha(f)})
    write(out/'CODE_SHA256.json',files)
    jobs=[{'config_id':c,'mode':m,'status':'planned_only' if a.plan_only else 'not_started'} for c in a.config for m in a.mode]
    inv={'protocol':PROTOCOL,'assets_manifest_sha256':sha(manifest),'jobs':jobs,'utc_start':now(),
         'status':'planned_only' if a.plan_only else 'running','models_trained':0,'strict_verified':0,'formal_training_eligible':False,
         'python':sys.version,'machine':platform.machine(),'expect_gpu':a.expect_gpu}
    write(out/'inventory.json',inv)
    if a.plan_only:return
    env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join([str(source),str(source/'o4_pilot')]),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    # Preserve every attempted cell. A guard refusal never terminates others' work.
    for j in jobs:
        folder=out/(j['config_id']+'_'+j['mode']);folder.mkdir();j['folder']=folder.name
        req={'config_id':j['config_id'],'mode':j['mode'],'assets_manifest':str(manifest),'protocol':PROTOCOL,
             'expect_gpu':a.expect_gpu,'output':str(folder),'trace_backend':'cuda-sw'};write(folder/'request.json',req)
        try:
            gpu_guard(folder,a.expect_gpu,idle_wait_seconds=30,check_interval_seconds=3,idle_checks_required=2)
            argv=[sys.executable,'-m','o4_i3.worker','--request',str(folder/'request.json')]
            if j['mode']!='GI':argv=['nsys','profile','--trace=cuda-sw,nvtx','--sample=none','--cpuctxsw=none',
                '--gpu-metrics-devices=none','--discard-environment=true','--capture-range=cudaProfilerApi','--capture-range-end=stop',
                '--output='+str(folder/'trace')]+argv
            command(argv,folder,'collect',timeout=600,env=env,cwd=source)
            if read(folder/'quality.json')['status']!='completed':raise RuntimeError('Worker incomplete')
            j['GPU_execution_completed']=True
            if j['mode']!='GI':
                command(['nsys','export','--type=sqlite','--output='+str(folder/'trace.sqlite'),str(folder/'trace.nsys-rep')],folder,'export',timeout=120)
                from o4_research.audit_v5 import audit
                summary=audit(folder,folder/'audit');j['engineering_acceptance']=summary['engineering_acceptance']['status']
                if not summary['engineering_acceptance']['accepted']:raise RuntimeError('Trace engineering acceptance blocked')
            j['status']='completed'
        except Exception as e:
            j.update(status='failed',error=f'{type(e).__name__}: {e}');write(folder/'failure.json',j)
        finally:write(out/'inventory.json',inv)
        print(j,flush=True)
    pairs=[]
    for c in a.config:
        by={j['mode']:j for j in jobs if j['config_id']==c and j['status']=='completed'}
        if 'P' in by and 'R' in by:
            try:
                from o4_research.audit import pair
                result=pair(out/by['P']['folder']/'audit',out/by['R']['folder']/'audit',out/(c+'_pair'))
                pairs.append({'config_id':c,'windows':len(result),'exact':sum(x['ordered_signature_exact'] for x in result)})
            except Exception as e:pairs.append({'config_id':c,'error':str(e)})
    inv.update(utc_end=now(),status='completed_development_smoke' if all(j['status']=='completed' for j in jobs) else 'failed_or_incomplete',P_R_pairs=pairs)
    write(out/'inventory.json',inv)
    if inv['status']!='completed_development_smoke':raise SystemExit(1)

if __name__=='__main__':main()
