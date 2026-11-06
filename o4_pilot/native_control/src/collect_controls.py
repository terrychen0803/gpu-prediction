"""Bounded, process-local native CUDA control collector. No service changes."""
from __future__ import annotations
import argparse,datetime,json,os,signal,subprocess,time
from pathlib import Path
from quality import read,write,sha
RUN=Path(__file__).resolve().parents[1]
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def command(argv,out,timeout=60):
 out=Path(out);started=utc();tick=time.perf_counter_ns()
 record={'argv':[str(x) for x in argv],'start_utc':started,'start_monotonic_ns':tick}
 with out.open('wb') as f:
  proc=subprocess.Popen([str(x) for x in argv],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=proc.wait(timeout=timeout)
  except (subprocess.TimeoutExpired,KeyboardInterrupt):
   # Only the process group created by this collector is owned by this run.
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   record.update(status='timeout_or_interrupted',returncode=proc.returncode,end_utc=utc(),end_monotonic_ns=time.perf_counter_ns());write(out.with_suffix(out.suffix+'.command.json'),record);raise
 record.update(status='completed' if rc==0 else 'failed',returncode=rc,end_utc=utc(),end_monotonic_ns=time.perf_counter_ns());write(out.with_suffix(out.suffix+'.command.json'),record)
 if rc:raise RuntimeError(f'Command exited {rc}; evidence: {out}')
 return record
def gpu_snapshot(dest):
 command(['nvidia-smi','--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],dest/'gpu.csv')
 command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],dest/'compute_processes.csv')
def binary_for(cell):return RUN/'native'/('serial_mre_drain_v2' if cell.startswith('D') or cell=='correctness_drain' else 'serial_mre')
def preflight(cell,expect_gpu):
 from runtime_guard import gpu_guard
 dest=RUN/'preflight'/cell;dest.mkdir(exist_ok=False);gpu_snapshot(dest);gpu_guard(dest,expect_gpu)
 for name,cmd in [('uname.txt',['uname','-srm']),('nsys_status.txt',['nsys','status','-e'])]:
  try:command(cmd,dest/name)
  except RuntimeError:pass
 # Restrict identification to GPU and tool properties; do not dump environment/secrets.
 write(dest/'PREFLIGHT.json',{'utc':utc(),'device_index':0,'expected_device_name_contains':expect_gpu,'scope':'native diagnostics only; not GI benchmark','background_policy':'preserve other contexts; record nonexclusive host; no service or Kubernetes changes','gpu_launch_allowed':'only bounded controls, no performance claims','source_sha256':sha(binary_for(cell).with_suffix('.cu')),'binary_sha256':sha(binary_for(cell))})
 gpu_lines=[x for x in (dest/'gpu.csv').read_text().splitlines() if x.strip()]
 if len(gpu_lines)!=1 or expect_gpu.lower() not in gpu_lines[0].lower():raise RuntimeError('Expected one matching physical GPU; no control executed')
 (dest/'preflight_report.md').write_text('# Read-only native control preflight\n\nGPU/tool discovery saved beside this file. RustDesk background context is preserved; measurements are nonexclusive diagnostics, not performance labels. No K3s/Pod/Netdata commands or changes. Each child process has a timeout and process-group cleanup.\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--cell',choices=['correctness','A1','A2','A3','A4','B1','B3','C3','D1','D3','D3_repeat','correctness_drain'],required=True);p.add_argument('--expect-gpu',default='GB10');a=p.parse_args();preflight(a.cell,a.expect_gpu)
 folder=RUN/('native_'+a.cell if a.cell.startswith('correctness') else 'captures/'+a.cell);folder.mkdir(exist_ok=False)
 cmd=[str(binary_for(a.cell)),'--measure','8','--warmup','8'];backend=None;mode='P'
 if a.cell in ('A2','A4'):cmd+=['--reference'];mode='R'
 if not a.cell.startswith('correctness'):
  backend='cuda-sw' if a.cell in ('A1','A2','B1','D1') else 'cuda'
  cmd=['nsys','profile',f'--trace={backend},nvtx','--discard-environment=true','--sample=none','--cpuctxsw=none','--gpu-metrics-devices=none','--capture-range=cudaProfilerApi','--capture-range-end=stop',f'--output={folder}/trace']+cmd
 if a.cell=='C3':cmd.insert(2,'--flush-on-cudaprofilerstop=true')
 request={'protocol_id':'o4_native_software_handoff_v3_20261106','cell':a.cell,'protocol_variant':'explicit_flush_true' if a.cell=='C3' else 'baseline','mode':mode,'requested_backend':backend,'measured':8,'warmup':8,'source_sha256':sha(binary_for(a.cell).with_suffix('.cu')),'binary_sha256':sha(binary_for(a.cell)),'full_forward_GI':False,'one_process_per_cell':True,'PDL_launch_attributes':'none; ordinary <<<>>> launches in source','correctness':'all256 float32 outputs exactly3.0; successful exit required','background':'existing contexts preserved; actual process snapshot in preflight'};request['protocol_id']='o4_native_trace_host_drain_v2_20261106' if a.cell.startswith('D') or a.cell=='correctness_drain' else ('o4_native_trace_explicit_flush_v1_20261106' if a.cell=='C3' else request['protocol_id']);request['host_drain_ms']=200 if a.cell.startswith('D') or a.cell=='correctness_drain' else 0;write(folder/'request.json',request)
 try:
  command(cmd,folder/'collect.log',timeout=90)
  text=(folder/'collect.log').read_text();header='iteration,event_elapsed_us,kernel_calls,positive_memset_bytes,zero_memset_bytes,reference'
  if header not in text:raise RuntimeError('Missing complete stdout ledger')
  data=text.split(header,1)[1].splitlines();records=[x for x in data if x and x[0].isdigit() and len(x.split(','))==6]
  if len(records)!=8:raise RuntimeError(f'Expected8 rows, got{len(records)}')
  (folder/'iterations.csv').write_text(header+'\n'+'\n'.join(records)+'\n')
  if not a.cell.startswith('correctness'):command(['nsys','export','--type=sqlite',f'--output={folder}/trace.sqlite',str(folder/'trace.nsys-rep')],folder/'export.log',timeout=90)
  write(folder/'execution.json',{'status':'completed','utc':utc(),'correctness_passed':True,'known_submission_ledger_complete':True,'raw_trace_collected':not a.cell.startswith('correctness')})
 except BaseException as e:
  write(folder/'execution.json',{'status':'failed','utc':utc(),'error':str(e)});raise
 finally:
  # Never reset the GPU or stop other users' contexts. Native resets only itself.
  command(['nvidia-smi','--query-gpu=name,utilization.gpu','--format=csv,noheader'],folder/'gpu_after.csv')
  command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],folder/'processes_after.csv')
 print('Completed:',a.cell,'evidence:',folder)
if __name__=='__main__':main()
