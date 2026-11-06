"""Independent CUDA API argument evidence; MUST NOT run under Nsight."""
import argparse,collections,csv,ctypes,ctypes.util,os,platform,sys,time
from pathlib import Path
from .common import *
def validate_rows(data):
 enters=[r for r in data if r['site']=='0'];exits=[r for r in data if r['site']=='1'];assert len(enters)==len(exits)>0
 exit_by={r['correlation_id']:r for r in exits};assert len(exit_by)==len(exits)
 for row in enters:
  other=exit_by[row['correlation_id']];assert other['return_code']=='0';assert other['scope']==row['scope'] and other['count_bytes']==row['count_bytes']
 return enters
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--library',required=True);p.add_argument('--assets-manifest');p.add_argument('--assets-root');p.add_argument('--cudart-library');p.add_argument('--expect-gpu',default='GB10');p.add_argument('--case',choices=['control','resnet'],required=True);a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
 status={'status':'started','protocol_id':'o4_memset_argument_observer_v2_20261106','utc':now(),'nsight_present':False,'training':False,'strict_verified':0,'observer_python_sha256':sha(__file__)}
 try:
  if os.getenv('IS_PROCESS_PROFILED_BY_NSYS'):raise RuntimeError('Do not combine subscriber with Nsight')
  gpu_guard(out,a.expect_gpu);import torch,torchvision
  torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.cuda.set_device(0)
  from o4_iteration import common as legacy_bootstrap
  from inference_pilot.benchmark import precision_policy
  precision=precision_policy(torch)
  lib=ctypes.CDLL(str(Path(a.library).resolve()));lib.observer_start.restype=ctypes.c_int;lib.observer_stop.argtypes=[ctypes.c_char_p];lib.observer_scope.argtypes=[ctypes.c_char_p]
  write(out/'environment.json',{'utc':now(),'pid':os.getpid(),'python':sys.version,'machine':platform.machine(),'torch':torch.__version__,'vision':torchvision.__version__,'cuda':torch.version.cuda,'precision':precision,'CUPTI_version':lib.observer_version(),'observer_library_sha256':sha(a.library),'no_Nsight_no_activity_tracing':True})
  with torch.inference_mode(),torch.autocast('cuda',enabled=False):
   if a.case=='control':
    x=torch.empty(64,dtype=torch.uint8,device='cuda');torch.cuda.synchronize();rc=lib.observer_start()
    if rc:raise RuntimeError(f'CUPTI subscribe failed{rc}; no multi-subscriber fallback')
    cudart_path=a.cudart_library or ctypes.util.find_library('cudart')
    if not cudart_path:raise RuntimeError('Cannot discover cudart; pass --cudart-library /absolute/path/to/libcudart.so')
    cudart=ctypes.CDLL(cudart_path);call=cudart.cudaMemsetAsync;call.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_size_t,ctypes.c_void_p];call.restype=ctypes.c_int
    for count in (64,0):
     lib.observer_scope(f'control/count{count}'.encode());rc=call(ctypes.c_void_p(x.data_ptr()),0,count,ctypes.c_void_p(torch.cuda.current_stream().cuda_stream));assert rc==0
    torch.cuda.synchronize();assert lib.observer_stop(str(out/'API_arguments.csv').encode())==0
   else:
    if not a.assets_manifest:raise ValueError('--assets-manifest required for RN-A')
    asset_data=read(a.assets_manifest)
    if 'workloads' in asset_data:
     from .adapters import Adapter
     adapter=Adapter(torch,'RN-A',a.assets_manifest);correct=adapter.correctness()
    else:
     from o4_iteration.adapters import Adapter
     assets=asset_data['resnet'];root=Path(a.assets_root)
     for item in assets.values():item['path']=str(root/item['path'])
     protocol=read(Path(__file__).resolve().parents[1]/'manifests/protocol.json');adapter=Adapter(torch,{'workload':'RN-A','assets':assets,'protocol':protocol});correct,plain=adapter.correctness();del plain
    for _ in range(8):adapter.forward()
    torch.cuda.synchronize();write(out/'adapter.json',adapter.metadata);write(out/'correctness.json',correct)
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten
    rows=[]
    class Observe(TorchDispatchMode):
     iteration=0;index=0
     def __torch_dispatch__(self,func,types,args=(),kwargs=None):
      i=self.index;self.index+=1;label=f'iteration{self.iteration}/op{i}/{func}';tensors=[x for x in tree_flatten((args,kwargs or {}))[0] if isinstance(x,torch.Tensor)];rows.append({'iteration':self.iteration,'op_index':i,'aten':str(func),'scope':label,'shapes':[list(x.shape) for x in tensors],'dtypes':[str(x.dtype) for x in tensors]});lib.observer_scope(label.encode())
      try:return func(*args,**(kwargs or {}))
      finally:lib.observer_scope(b'outside')
    observer=Observe();rc=lib.observer_start()
    if rc:raise RuntimeError(f'CUPTI subscribe failed{rc}; no Nsight combination attempted')
    try:
     for i in range(3):
      observer.iteration=i;observer.index=0
      with observer:result=adapter.forward()
      torch.cuda.synchronize();del result
    finally:stop=lib.observer_stop(str(out/'API_arguments.csv').encode());assert stop==0,stop
    write(out/'ATen_operations.json',rows)
  data=list(csv.DictReader((out/'API_arguments.csv').open()));enters=validate_rows(data)
  if a.case=='control':assert sorted(int(x['count_bytes']) for x in enters)==[0,64]
  status.update(status='completed',utc_end=now(),direct_argument_calls=len(enters),count_distribution=dict(collections.Counter(x['count_bytes'] for x in enters)),historical_RN_A_relabelled=False,submitted_work_coverage_historical='unverified')
  if a.case=='resnet':
   selected=[x for x in enters if '/op140/' in x['scope']];status.update(op140_direct_calls=len(selected),op140_count_bytes=[int(x['count_bytes']) for x in selected],op140_scopes=[x['scope'] for x in selected]);assert len(selected)==3,'op140 count differs; investigate rather than guess'
 except BaseException as exc:status.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
 finally:write(out/'STATUS.json',status)
 print(status)
if __name__=='__main__':main()
