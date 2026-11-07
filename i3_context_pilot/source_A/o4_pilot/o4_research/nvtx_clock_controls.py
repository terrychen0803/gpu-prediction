"""Three bounded SW/CPU controls; no hardware tracing or repeat drain tests."""
import argparse,collections,csv,sqlite3,shutil,sys
from pathlib import Path
from .common import *
def inspect(folder,kind):
 c=sqlite3.connect('file:'+str(folder/'trace.sqlite')+'?mode=ro',uri=True);c.execute('pragma query_only=ON');c.row_factory=sqlite3.Row;tables={x[0] for x in c.execute("select name from sqlite_master where type='table'")}
 def rows(t):return [dict(x) for x in c.execute('select rowid as _rowid,* from '+t)] if t in tables else []
 strings={x['id']:x['value'] for x in rows('StringIds')};nvtx=rows('NVTX_EVENTS');diags=rows('DIAGNOSTIC_EVENT');write(folder/'diagnostics.json',diags);calls=rows('CUPTI_ACTIVITY_KIND_RUNTIME');event=rows('CUPTI_ACTIVITY_KIND_CUDA_EVENT');gpu=[]
 for t in ['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMSET','CUPTI_ACTIVITY_KIND_MEMCPY']:gpu.extend([dict(x,table=t) for x in rows(t)])
 counts={t:len(rows(t)) for t in ['CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_MEMSET','CUPTI_ACTIVITY_KIND_MEMCPY']};warnings=[x['text'] for x in diags if x.get('severity',0)>=2]
 named=[dict(x,name=x['text'] or strings.get(x['textId'],'')) for x in nvtx];scope=[x for x in named if (x['name'] or '').startswith('diag.iter/')];api={x['correlationId']:x for x in calls};anchoraudit=[]
 for s in scope:
  inside=[e for e in event if e['correlationId'] in api and s['start']<=api[e['correlationId']]['start'] and api[e['correlationId']]['end']<=s['end']];inside.sort(key=lambda e:api[e['correlationId']]['start']);ev=[e for e in gpu if e['correlationId'] in api and s['start']<=api[e['correlationId']]['start'] and api[e['correlationId']]['end']<=s['end']]
  row={'scope':s['name'],'CUDA_EVENT_records':len(inside),'nonzero_device_completion':sum(bool(x['timestamp']) for x in inside),'recorded_work':len(ev)}
  if len(inside)==2 and all(x['timestamp'] for x in inside) and ev:row.update(start_completion_ns=inside[0]['timestamp'],end_completion_ns=inside[1]['timestamp'],device_event_interval_ns=inside[1]['timestamp']-inside[0]['timestamp'],first_work_minus_start_completion_ns=min(x['start'] for x in ev)-inside[0]['timestamp'],end_completion_minus_last_work_ns=inside[1]['timestamp']-max(x['end'] for x in ev))
  anchoraudit.append(row)
 csvwrite(folder/'event_anchor_audit.csv',anchoraudit);write(folder/'CUDA_EVENT_records.json',event)
 summary={'kind':kind,'known_CPU_NVTX_ranges_observed':sum((x['name'] or '').startswith('audit.cpu/') for x in named),'measured_iteration_scopes':len(scope),'all_NVTX_rows':len(nvtx),'GPU_counts':counts,'CUDA_EVENT_rows':len(event),'nonzero_device_completion_timestamps':sum(bool(x.get('timestamp')) for x in event),'warnings':warnings,'backend_evidence':[x['text'] for x in diags if 'trace was collected' in x['text'] or 'tracing used' in x['text']],'strict_verified':0,'raw_timestamps_changed':False,'interpretation':'comparison changes capture/instrumentation; not a proof of warning root cause or independent clock calibration'};write(folder/'SUMMARY.json',summary);c.close();return summary
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--native-binary',required=True);p.add_argument('--CPU-binary',required=True);p.add_argument('--expect-gpu',default='GB10');a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);results=[]
 settings=[('CPU_NVTX_wholeprocess',a.CPU_binary,['--trace=nvtx'],False),('SW_native_wholeprocess',a.native_binary,['--trace=cuda-sw,nvtx'],True),('SW_device_event_anchors',a.native_binary,['--trace=cuda-sw,nvtx','--capture-range=cudaProfilerApi','--capture-range-end=stop','--cuda-event-trace=true'],True)]
 for name,binary,options,gpu in settings:
  f=out/name;f.mkdir();write(f/'request.json',{'protocol_id':'o4_nvtx_clock_control_v1_20261106','case':name,'binary_sha256':sha(binary),'additional_hardware_tests':0,'development_only':True})
  if gpu:gpu_guard(f,a.expect_gpu)
  cmd=['nsys','profile',*options,'--sample=none','--cpuctxsw=none','--gpu-metrics-devices=none','--discard-environment=true','--output='+str(f/'trace'),binary]
  if gpu:cmd+=['--measure','8','--warmup','8']
  try:command(cmd,f,'collect',timeout=120);command(['nsys','export','--type=sqlite','--output='+str(f/'trace.sqlite'),str(f/'trace.nsys-rep')],f,'export',timeout=90);results.append(inspect(f,name))
  except Exception as e:write(f/'failure.json',{'error':str(e)});raise
 write(out/'SUMMARY.json',{'cases':results,'hardware_GPU_experiments':0,'strict_verified':0});print(results)
if __name__=='__main__':main()
