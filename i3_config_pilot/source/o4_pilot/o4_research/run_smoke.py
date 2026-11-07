"""One independent process/mode; at most seven A configurations, never a grid."""
import argparse,os,sys,statistics
from pathlib import Path
from .adapters import SPECS,require_a
from .common import *
from .worker import PROTOCOL
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--assets-manifest',required=True);p.add_argument('--workload',nargs='+',default=list(SPECS));p.add_argument('--mode',nargs='+',choices=['GI','P','R'],default=['GI','P','R']);p.add_argument('--expect-gpu',default='GB10');p.add_argument('--plan-only',action='store_true');a=p.parse_args()
 for name in a.workload:require_a(name)
 if len(set(a.workload))!=len(a.workload):raise ValueError('Duplicate workloads')
 out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);manifest=Path(a.assets_manifest).resolve();assets=read(manifest);jobs=[{'workload':w,'mode':m,'status':'planned_only' if a.plan_only else 'not_started'} for w in a.workload for m in a.mode];inventory={'protocol':PROTOCOL,'jobs':jobs,'utc':now(),'assets_manifest_sha256':sha(manifest),'status':'planned_only' if a.plan_only else 'running','models_trained':0,'strict_verified':0};write(out/'inventory.json',inventory)
 if a.plan_only:return
 for j in jobs:
  folder=out/(j['workload']+'_'+j['mode']);folder.mkdir();j['folder']=str(folder.relative_to(out));req={'workload':j['workload'],'mode':j['mode'],'assets_manifest':str(manifest),'protocol':PROTOCOL,'expect_gpu':a.expect_gpu,'output':str(folder),'trace_backend':'cuda-sw'};write(folder/'request.json',req)
  try:
   if assets['workloads'][j['workload']]['status']!='ready':raise RuntimeError('pending_assets; no substitute')
   gpu_guard(folder,a.expect_gpu);cmd=[sys.executable,'-m','o4_research.worker','--request',str(folder/'request.json')]
   if j['mode']!='GI':cmd=['nsys','profile','--trace=cuda-sw,nvtx','--sample=none','--cpuctxsw=none','--gpu-metrics-devices=none','--discard-environment=true','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(folder/'trace')]+cmd
   env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_CONFIG_DIR=str(out/'ultralytics_settings'),TOKENIZERS_PARALLELISM='false');command(cmd,folder,'collect',timeout=900,env=env,cwd=Path(__file__).resolve().parents[1]);q=read(folder/'quality.json')
   if q['status']!='completed':raise RuntimeError('worker not completed')
   if j['mode']!='GI':
    command(['nsys','export','--type=sqlite','--output='+str(folder/'trace.sqlite'),str(folder/'trace.nsys-rep')],folder,'export',timeout=120)
    from .audit import audit
    summary=audit(folder,folder/'audit');j['trace_structure']=summary['status']
   j['status']='completed';j['GPU_execution_completed']=True
  except Exception as e:j.update(status='failed',error=f'{type(e).__name__}: {e}');write(folder/'failure.json',j)
  finally:write(out/'inventory.json',inventory)
  print(j,flush=True)
 for name in a.workload:
  by={j['mode']:j for j in jobs if j['workload']==name and j['status']=='completed'}
  if 'P' in by and 'R' in by:
   from .audit import pair
   try:pair(out/by['P']['folder']/'audit',out/by['R']['folder']/'audit',out/(name+'_pair'))
   except Exception as e:write(out/(name+'_pair_failure.json'),{'error':str(e)})
 inventory['status']='completed' if all(x['status']=='completed' for x in jobs) else 'failed_or_incomplete';inventory['utc_end']=now();write(out/'inventory.json',inventory)
 if inventory['status']!='completed':raise SystemExit(1)
if __name__=='__main__':main()
