"""One initialized event pair per serial iteration, no per-operation events."""
def measure(torch,adapter,start,end,stream,mode,stage,index,inventory=None):
 from .common import now
 import time
 marked=mode in ('P','R')
 if marked:torch.cuda.nvtx.range_push(f'o4.iter/{stage}/{index:06d}')
 utc0=now();t0=time.perf_counter_ns()
 try:
  if torch.cuda.current_stream()!=stream:raise RuntimeError('Current stream changed before forward')
  start.record(stream)
  if inventory is not None:inventory.begin(stage,index)
  try:output=adapter.forward()
  finally:
   if inventory is not None:inventory.end()
  end.record(stream)
  end.synchronize()
  t1=time.perf_counter_ns();utc1=now()
  if torch.cuda.current_stream()!=stream:raise RuntimeError('Current stream changed in adapter')
  us=float(start.elapsed_time(end))*1000.0
  if us<=0:raise RuntimeError('Nonpositive event interval')
 finally:
  if marked:torch.cuda.nvtx.range_pop()
 row={'stage':stage,'iteration':index,'status':'completed','utc_start':utc0,'utc_end':utc1,'monotonic_start_ns':t0,'monotonic_end_ns':t1,'elapsed_us':us,'target_kind':'iteration_device_event_us' if mode=='GI' else 'profiled_iteration_event_us','host_forward_plus_wait_us':(t1-t0)/1000.0}
 del output
 return row
