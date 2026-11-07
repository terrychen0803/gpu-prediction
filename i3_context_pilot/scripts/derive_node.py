"""Portable guarded local D-input derivation, outside trace and formal GI."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['A','BC'],required=True);p.add_argument('--base',required=True);p.add_argument('--expect-gpu',required=True);p.add_argument('--out',required=True);p.add_argument('--plan-only',action='store_true');a=p.parse_args()
 root=Path(__file__).resolve().parents[1];src=root/('source_'+a.stage);sys.path[:0]=[str(src),str(src/'o4_pilot')]
 from o4_research.common import write,now,sha,gpu_guard
 from o4_i3_context.assets import verify_base
 from o4_i3_v2.monitor import Monitor
 expected=json.loads((root/'manifests'/('SOURCE_'+a.stage+'_SHA256.json')).read_text());actual={str(f.relative_to(src)):sha(f) for f in src.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc'}
 if actual!={x['path'].removeprefix('source/'):x['sha256'] for x in expected}:raise ValueError('Source does not match frozen release; do not regenerate hash index')
 base=Path(a.base).resolve();verify_base(base);out=Path(a.out).resolve();assets=out/'assets_context_local'
 argv=[sys.executable,'-m','o4_i3_context.assets','derive','--base',str(base),'--path',str(assets),'--expect-gpu',a.expect_gpu]
 if a.plan_only:print(json.dumps({'status':'plan_only_no_GPU','stage':a.stage,'argv':argv,'source_files_verified':len(actual)},indent=2));return 0
 out.mkdir(parents=True,exist_ok=False);gpu_guard(out,a.expect_gpu,idle_wait_seconds=30,check_interval_seconds=3,idle_checks_required=2)
 env=dict(os.environ,PYTHONPATH=str(src)+os.pathsep+str(src/'o4_pilot'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 record={'argv':argv,'utc_start':now(),'monotonic_start_ns':time.perf_counter_ns(),'role':'local native D input derivation outside formal timing/capture','stage':a.stage,'base_manifest_sha256':sha(base)};child=None;code=None
 try:
  with (out/'derive.log').open('w') as log:
   child=subprocess.Popen(argv,cwd=src,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);write(out/'environment.json',{'pid':child.pid})
   with Monitor(out,5) as monitor:
    try:code=child.wait(timeout=600)
    finally:
     if child.poll() is None:
      os.killpg(child.pid,signal.SIGTERM)
      try:child.wait(timeout=10)
      except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   write(out/'monitor_summary.json',{'samples':len(monitor.rows),'foreign_GPU_work_observed':monitor.foreign,'errors':monitor.errors,'note':'5s sampled only; not proof of no interference between samples'})
  record.update(returncode=code,status='completed' if code==0 and not monitor.foreign and not monitor.errors else 'failed_or_review')
 except BaseException as e:
  record.update(status='interrupted_or_failed',error=f'{type(e).__name__}: {e}');raise
 finally:record.update(utc_end=now(),monotonic_end_ns=time.perf_counter_ns());write(out/'command.json',record)
 if record['status']!='completed':return 1
 print('Local assets:',assets/'ASSETS.json');return 0
if __name__=='__main__':raise SystemExit(main())
