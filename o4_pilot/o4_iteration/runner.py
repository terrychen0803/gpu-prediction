from __future__ import annotations
import os,random,shutil,subprocess,sys,time,uuid
from pathlib import Path
from .common import ROOT,read,write,now,code_manifest,snapshot,sha
from .manifest import POLICY,IDS

def run_logged(cmd,folder,name,timeout=300):
 write(folder/(name+'_command.json'),{'argv':cmd,'utc_start':now()});t0=time.perf_counter()
 with (folder/(name+'.log')).open('w') as f:
  proc=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:code=proc.wait(timeout=timeout)
  except BaseException:
   import signal
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:
    os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   raise
 cost=time.perf_counter()-t0;write(folder/(name+'_execution.json'),{'returncode':code,'elapsed_seconds':cost,'utc_end':now()});return code

def execute(args):
 out=Path(args.out).resolve()
 if out.exists():raise FileExistsError('Refusing to overwrite existing run: '+str(out))
 workloads=args.workload or list(IDS[:3]);modes=args.mode or POLICY['modes']
 if len(set(workloads))!=len(workloads) or any(x not in IDS for x in workloads):raise ValueError('Unsupported/duplicate workloads')
 if len(set(modes))!=len(modes) or any(x not in POLICY['modes'] for x in modes):raise ValueError('Unsupported/duplicate modes')
 if args.repeats not in (1,2,3):raise ValueError('repeats must be 1,2,3')
 assets=read(ROOT/'manifests/assets.json');jobs=[{'workload':w,'mode':m,'repeat':r,'status':'planned_only' if args.dry_run else 'not_started','opaque_id':uuid.uuid4().hex[:12]} for w in workloads for m in modes for r in range(1,args.repeats+1)]
 random.Random(POLICY['schedule_seed']).shuffle(jobs);out.mkdir(parents=True)
 inventory={'utc_start':now(),'status':'planned_only' if args.dry_run else 'running','protocol':POLICY,'workloads':workloads,'modes':modes,'repeats':args.repeats,'jobs':jobs,'code_manifest':code_manifest(),'expected_gpu':args.expect_gpu,'trace_backend':args.trace_backend,'formal_reference_verified':False}
 commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True)
 inventory['collector_git_commit']=commit.stdout.strip() if commit.returncode==0 else None
 write(out/'inventory.json',inventory);write(out/'protocol.json',POLICY)
 if args.dry_run:print('planned_only:',len(jobs),'cells; no CUDA process started');return 0
 if any(x in modes for x in ('P','R')) and not shutil.which('nsys'):raise RuntimeError('nsys required for P/R')
 for index,job in enumerate(jobs):
  folder=out/'raw'/f'cell_{index:03d}';folder.mkdir(parents=True);job['folder']=str(folder.relative_to(out));job['status']='started';write(out/'inventory.json',inventory)
  try:
   state=snapshot();write(folder/'host_before.json',state)
   if state['processes']['returncode']!=0 or state['processes']['stdout'].strip():raise RuntimeError('Cannot establish GPU availability: compute processes/query error')
   if code_manifest()!=inventory['code_manifest']:raise RuntimeError('Collector source changed during run')
   a=assets['resnet'] if job['workload']=='RN-A' else assets['operators'][job['workload']]
   req={**job,'protocol':POLICY,'assets':a,'expect_gpu':args.expect_gpu,'trace_backend':args.trace_backend,'output':str(folder),'planned_only':False,'code_manifest':inventory['code_manifest']}
   write(folder/'request.json',req)
   cmd=[sys.executable,'-m','o4_iteration','worker','--request',str(folder/'request.json')]
   if job['mode']!='GI':cmd=['nsys','profile','--trace='+args.trace_backend+',nvtx','--sample=none','--cpuctxsw=none','--gpu-metrics-devices=none','--capture-range=cudaProfilerApi','--capture-range-end=stop','--output='+str(folder/'trace'),*cmd]
   code=run_logged(cmd,folder,'collect')
   if code!=0 or read(folder/'quality.json')['status']!='completed':raise RuntimeError('Worker/profile failed; inspect preserved collect.log and quality.json')
   if job['mode']!='GI':
    code=run_logged(['nsys','export','--type=sqlite','--output='+str(folder/'trace.sqlite'),str(folder/'trace.nsys-rep')],folder,'export')
    if code!=0:raise RuntimeError('Nsight export failed')
    from .trace import audit
    q=audit(folder,out/'quality'/f'cell_{index:03d}');job['trace_status']=q['status']
    if q['status']=='failed':raise RuntimeError('Trace audit failed: '+str(q['errors']))
   job['status']='completed'
  except Exception as exc:
   job['status']='failed';job['error']=f'{type(exc).__name__}: {exc}'
   write(folder/'attempt_failure.json',{'error':job['error'],'utc':now()})
  finally:
   write(folder/'host_after.json',snapshot());write(out/'inventory.json',inventory)
  print(f"{index+1}/{len(jobs)} {job['workload']} {job['mode']} repeat{job['repeat']}: {job['status']}",flush=True)
  if job['status']=='failed' and not args.keep_going:break
 inventory.update(status='completed' if all(j['status']=='completed' for j in jobs) else 'failed_or_incomplete',utc_end=now());write(out/'inventory.json',inventory)
 return 0 if inventory['status']=='completed' else 1
