from __future__ import annotations
import contextlib,os,platform,sys,time,traceback
from pathlib import Path
from .common import read,write,csvwrite,jsonl,now,code_manifest
from .manifest import validate_request,stable
from .adapters import Adapter,dispatch_inventory
from .timer import measure

def run(path):
 req=validate_request(read(path));out=Path(req['output']);p=req['protocol'];quality={'status':'started','utc_start':now(),'errors':[],'formal_reference_verified':False};rows=[];warm=[];inventory=None;capturing=False
 write(out/'quality.json',quality)
 try:
  import torch,torchvision
  from inference_pilot.benchmark import precision_policy
  if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible CUDA GPU required; no CPU fallback')
  torch.cuda.set_device(0);torch.set_num_threads(1);torch.set_num_interop_threads(1)
  props=torch.cuda.get_device_properties(0)
  if req['expect_gpu'].lower() not in props.name.lower():raise RuntimeError('GPU identity mismatch: '+props.name)
  precision=precision_policy(torch);stream=torch.cuda.current_stream()
  write(out/'environment.json',{'utc':now(),'pid':os.getpid(),'python':sys.version,'machine':platform.machine(),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu_name':props.name,'gpu_uuid':str(getattr(props,'uuid','unavailable')),'stream_cuda_handle':stream.cuda_stream,'precision':precision,'code_manifest':code_manifest()})
  with torch.inference_mode(),torch.autocast(device_type='cuda',enabled=False):
   adapter=Adapter(torch,req);correct,plain=adapter.correctness();quality['correctness']=correct;write(out/'adapter.json',adapter.metadata)
   inventory=dispatch_inventory(torch) if req['mode']=='R' else None
   # Verify dispatch does not change the native FP16 output before any timing.
   if inventory is not None:
    with inventory:marked=adapter.forward()
    torch.cuda.synchronize()
    if not torch.equal(plain,marked):raise RuntimeError('Dispatch instrumentation changed output')
    quality['instrumented_output_exact']=True
    del marked
   del plain
   start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
   start.record(stream);end.record(stream);end.synchronize()  # event initialization outside capture
   torch.cuda.nvtx.range_push('o4.setup_nvtx');torch.cuda.nvtx.range_pop()
   for i in range(p['warmup_max']):
    row=measure(torch,adapter,start,end,stream,'GI','warmup',i);warm.append(row)
    if (i+1)%p['warmup_window']==0 and stable([r['elapsed_us'] for r in warm],p):break
   else:raise RuntimeError('Warmup failed stability gate within fixed cap')
   quality['warmup']={'passed':True,'iterations':len(warm)}
   if req['mode'] in ('P','R'):torch.cuda.profiler.start();capturing=True
   with inventory if inventory is not None else contextlib.nullcontext():
    for stage,count in [('transition',p['transition_iterations']),('measure',p['measured_iterations'])]:
     for i in range(count):rows.append(measure(torch,adapter,start,end,stream,req['mode'],stage,i,inventory))
   if capturing:
    # Destroy initialized CUDA events while their own context is still valid.
    del start,end,adapter
    torch.cuda.synchronize()
    time.sleep(1)
    from o1_operator.worker import reset_cuda_context_for_trace
    quality['own_context_reset_returncode']=reset_cuda_context_for_trace(torch.version.cuda)
    torch.cuda.profiler.stop();capturing=False
   quality.update(status='completed',utc_end=now(),measured_iterations=sum(r['stage']=='measure' for r in rows),stream_check='current_stream_unchanged; full-device-work stream requires trace audit',trace_integrity='not_applicable' if req['mode']=='GI' else 'pending_audit')
 except Exception as exc:
  quality.update(status='failed',utc_end=now());quality['errors'].append(f'{type(exc).__name__}: {exc}');(out/'worker_error.txt').write_text(traceback.format_exc());print(quality['errors'][-1],file=sys.stderr)
 finally:
  if capturing:
   try:
    import torch
    torch.cuda.profiler.stop()
   except Exception:pass
  if rows:csvwrite(out/'iterations.csv',rows)
  if warm:csvwrite(out/'warmup.csv',warm)
  if inventory is not None:jsonl(out/'reference_dispatch.jsonl',inventory.rows)
  write(out/'quality.json',quality)
 return 0 if quality['status']=='completed' else 1
