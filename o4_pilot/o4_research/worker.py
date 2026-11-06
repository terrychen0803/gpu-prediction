"""Bounded A-smoke; new protocol, software-only, never formal training."""
import argparse,contextlib,os,platform,sys,time,traceback
from pathlib import Path
from .adapters import Adapter
from .common import *
PROTOCOL={'protocol_id':'o4_A_inventory_smoke_v1_20261106','warmup':8,'transition_iterations':1,'measured_iterations':3,'modes':['GI','P','R'],'independent_process_repeats':1,'repeat_stability_evaluated':False,'dtype':'float16','eager':True,'autocast':False,'compile':False,'cuda_graph':False,'trace_backend':'cuda-sw','scope':'inventory development only; not frozen40-iteration pilot','resnet_relative_l2_max':.02,'resnet_max_abs_over_scale_max':.02,'finalization':'release adapter/events; synchronize; own-context reset; ProfilerStop; no delay','strict_verified':0}
def run(reqfile):
 req=read(reqfile);out=Path(req['output']);quality={'status':'started','utc_start':now(),'formal_reference_verified':False,'errors':[]};rows=[];inventory=None;capturing=False
 try:
  import torch,torchvision
  from o4_iteration import common as legacy_bootstrap  # initialize vendored legacy import path
  from inference_pilot.benchmark import precision_policy
  torch.set_num_threads(1);torch.set_num_interop_threads(1)
  if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required; no fallback')
  torch.cuda.set_device(0);policy=precision_policy(torch);stream=torch.cuda.current_stream();props=torch.cuda.get_device_properties(0)
  if req['expect_gpu'].lower() not in props.name.lower():raise RuntimeError('GPU mismatch')
  write(out/'environment.json',{'utc':now(),'pid':os.getpid(),'python':sys.version,'machine':platform.machine(),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':props.name,'precision':policy})
  with torch.inference_mode(),torch.autocast('cuda',enabled=False):
   adapter=Adapter(torch,req['workload'],req['assets_manifest']);correct=adapter.correctness();quality['correctness']=correct;write(out/'adapter.json',adapter.metadata)
   if req['mode']=='R':
    from o4_iteration.adapters import dispatch_inventory
    inventory=dispatch_inventory(torch)
    adapter.prepare();plain=adapter.forward().detach();adapter.prepare()
    with inventory:marked=adapter.forward().detach()
    torch.cuda.synchronize()
    if not torch.equal(plain,marked):raise RuntimeError('TorchDispatch changes native output')
    quality['instrumented_output_exact']=True;del plain,marked
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record(stream);end.record(stream);end.synchronize();torch.cuda.nvtx.range_push('o4.setup_nvtx');torch.cuda.nvtx.range_pop()
   for _ in range(PROTOCOL['warmup']):adapter.prepare();adapter.forward();torch.cuda.synchronize()
   if req['mode']!='GI':torch.cuda.profiler.start();capturing=True
   with inventory if inventory is not None else contextlib.nullcontext():
    for stage,count in [('transition',1),('measure',3)]:
     for i in range(count):
      if req['mode']!='GI':torch.cuda.nvtx.range_push(f'o4.prepare/{stage}/{i:06d}')
      adapter.prepare();torch.cuda.synchronize() # preparation/reset excluded, never per op
      if req['mode']!='GI':torch.cuda.nvtx.range_pop()
      label=f'o4.iter/{stage}/{i:06d}';scope=req['mode']!='GI'
      if scope:torch.cuda.nvtx.range_push(label)
      if inventory:inventory.begin(stage,i)
      utc_start=now();tick=time.perf_counter_ns();start.record(stream);result=adapter.forward();end.record(stream);end.synchronize();elapsed=start.elapsed_time(end)*1000;finish=time.perf_counter_ns();utc_end=now()
      if inventory:inventory.end()
      if scope:torch.cuda.nvtx.range_pop()
      rows.append({'stage':stage,'iteration':i,'utc_start':utc_start,'utc_end':utc_end,'start_monotonic_ns':tick,'end_monotonic_ns':finish,'elapsed_us':elapsed,'target_kind':'iteration_device_event_us' if req['mode']=='GI' else 'profiled_iteration_event_us','cache_before':adapter.last_cache_before,'cache_after':adapter.last_cache_after,'status':'completed'});del result
   if capturing:
    del start,end,adapter;torch.cuda.synchronize()
    from o1_operator.worker import reset_cuda_context_for_trace
    quality['own_context_reset_returncode']=reset_cuda_context_for_trace(torch.version.cuda);torch.cuda.profiler.stop();capturing=False
   quality.update(status='completed',utc_end=now(),measured_iterations=3,trace_integrity='not_applicable' if req['mode']=='GI' else 'pending_audit',reference_status='development_unverified')
 except Exception as e:
  quality.update(status='failed',utc_end=now());quality['errors'].append(f'{type(e).__name__}: {e}');(out/'worker_error.txt').write_text(traceback.format_exc());print(quality['errors'][-1],file=sys.stderr)
 finally:
  if capturing:
   try:
    import torch
    torch.cuda.profiler.stop()
   except Exception:pass
  if rows:csvwrite(out/'iterations.csv',rows)
  if inventory is not None:
   with (out/'reference_dispatch.jsonl').open('w') as f:
    for row in inventory.rows:f.write(__import__('json').dumps(row)+'\n')
  write(out/'quality.json',quality)
 return 0 if quality['status']=='completed' else 1
def main():
 p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args();raise SystemExit(run(a.request))
if __name__=='__main__':main()
