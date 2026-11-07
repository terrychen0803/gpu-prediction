"""Strict schema-aware read-only native control audit; no torch/CUDA imports."""
from __future__ import annotations
import argparse,collections,re,sqlite3
from pathlib import Path
from quality import *
from nsys_schema import validate_export_schema
from trace_contract import diagnostic_policy
RUN=Path(__file__).resolve().parents[1]
ITER=re.compile(r'^diag\.iter/(\d{6})$');OP=re.compile(r'^diag\.op/(\d{6})/(first|second|memset64|memset0)$')
def table_rows(c,t,tables):return [dict(x) for x in c.execute('SELECT rowid AS _rowid,* FROM "'+t+'"')] if t in tables else []
def scope_for(a,scopes):return [s for s in scopes if s['globalTid']==a['globalTid'] and s['start']<=a['start'] and a['end']<=s['end']]
def inspect(cell,out):
 folder=RUN/'captures'/cell;req=read(folder/'request.json');out.mkdir(exist_ok=False)
 c=sqlite3.connect('file:'+str(folder/'trace.sqlite')+'?mode=ro',uri=True);c.execute('PRAGMA query_only=ON');c.row_factory=sqlite3.Row
 tables={x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table'")};schema={t:[dict(x) for x in c.execute('pragma table_info("'+t+'")')] for t in sorted(tables)};write(out/'schema.json',schema)
 meta=table_rows(c,'META_DATA_EXPORT',tables);schema_version=next(x['value'] for x in meta if x['name']=='EXPORT_SCHEMA_VERSION')
 compatibility=validate_export_schema(schema_version,schema);write(out/'schema_compatibility.json',compatibility)
 integrity=c.execute('pragma quick_check').fetchone()[0];strings={x['id']:x['value'] for x in table_rows(c,'StringIds',tables)}
 contexts=table_rows(c,'TARGET_INFO_CUDA_CONTEXT_INFO',tables);pids={x['processId'] for x in contexts}
 if len(pids)!=1:raise ValueError('Expected exactly one native process/context set')
 pid=next(iter(pids));diagnostics=table_rows(c,'DIAGNOSTIC_EVENT',tables);warnings=[x for x in diagnostics if x.get('severity',0)>=2];write(out/'diagnostics.json',diagnostics)
 # Raw Nsight metadata may include the inherited environment; never export it.
 metadata={t:table_rows(c,t,tables) for t in ('META_DATA_EXPORT','TARGET_INFO_CUDA_CONTEXT_INFO','TARGET_INFO_CUDA_STREAM','ENUM_CUDA_KERNEL_LAUNCH_TYPE','ENUM_DIAGNOSTIC_SEVERITY_LEVEL')};write(out/'safe_metadata.json',metadata)
 engineering=diagnostic_policy(diagnostics,'cuda-sw');write(out/'engineering_diagnostics.json',engineering)
 backend_messages=[x['text'] for x in diagnostics if any(s in x['text'].lower() for s in ('software instrumented trace','hardware','fallback','hw trace','incomplete cupti'))]
 actual='software' if any('legacy/software instrumented trace was collected' in s for s in backend_messages) else 'unverified'
 if any('Hardware tracing used for CUDA tracing.' in s for s in backend_messages):actual='hardware'
 nvtx=[];iters=[];ops=[]
 for n in table_rows(c,'NVTX_EVENTS',tables):
  if ((n['globalTid']>>24)&0xffffff)!=pid:continue
  n['name']=n['text'] or strings.get(n['textId'],'');nvtx.append(n)
  if ITER.fullmatch(n['name']):n['iteration']=int(ITER.fullmatch(n['name']).group(1));iters.append(n)
  elif OP.fullmatch(n['name']):m=OP.fullmatch(n['name']);n.update(iteration=int(m.group(1)),operation=m.group(2));ops.append(n)
 expected_iters={f'diag.iter/{i:06d}' for i in range(8)};expected_ops={f'diag.op/{i:06d}/{op}' for i in range(8) for op in ('first','second','memset64','memset0')} if req['mode']=='R' else set()
 nvtx_exact=len(iters)==8 and len(ops)==len(expected_ops) and {x['name'] for x in iters}==expected_iters and {x['name'] for x in ops}==expected_ops and all(isinstance(x['end'],int) and x['end']>=x['start'] for x in iters+ops)
 apis=[]
 for t in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
  for a in table_rows(c,t,tables):
   if ((a['globalTid']>>24)&0xffffff)!=pid:continue
   a.update(api_name=strings.get(a['nameId'],''),source_table=t,api_key=f'{t}:{a["_rowid"]}')
   its=scope_for(a,iters);os=scope_for(a,ops);a.update(iteration=its[0]['iteration'] if len(its)==1 else None,operation=os[0]['operation'] if len(os)==1 else None)
   apis.append(a)
 index=collections.defaultdict(list)
 for a in apis:index[(a['globalTid']>>24,a['correlationId'])].append(a)
 events=[];errors=[]
 for t,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMSET','memset'),('CUPTI_ACTIVITY_KIND_MEMCPY','copy')]:
  for e in table_rows(c,t,tables):
   if ((e['globalPid']>>24)&0xffffff)!=pid:continue
   matches=index[(e['globalPid']>>24,e['correlationId'])]
   if len(matches)!=1:errors.append({'table':t,'rowid':e['_rowid'],'reason':'ambiguous_or_missing_API','count':len(matches)});continue
   a=matches[0]
   e.update(uid=f'{t}:{e["_rowid"]}',kind=kind,source_table=t,api_name=a['api_name'],api_key=a['api_key'],api_start=a['start'],api_end=a['end'],api_return=a.get('returnValue'),iteration=a['iteration'],operation=a['operation'],kernel_name=strings.get(e.get('demangledName'),''),gpu_end_minus_API_entry_ns=e['end']-a['start'],gpu_start_minus_API_entry_ns=e['start']-a['start'])
   if e['end']<=e['start']:errors.append({'uid':e['uid'],'reason':'nonpositive_duration'})
   if e['iteration'] is None:errors.append({'uid':e['uid'],'reason':'unassigned_iteration'})
   events.append(e)
 events.sort(key=lambda e:(e['start'],e['end'],e['uid']));pairs=overlapping_pairs(events);csvwrite(out/'events.csv',events);csvwrite(out/'correlation_errors.csv',errors,['uid','reason','count','table','rowid'])
 seen=collections.Counter(e['api_key'] for e in events);reverse=[];iterations=[];signatures=[];opclosures=[]
 for i in range(8):
  ev=[e for e in events if e['iteration']==i];aa=sorted([a for a in apis if a['iteration']==i and any(s in a['api_name'].lower() for s in ('launchkernel','memsetasync','memcpy'))],key=lambda a:(a['start'],a['end']))
  memset_apis=[a for a in aa if 'memsetasync' in a['api_name'].lower()];kernel_apis=[a for a in aa if 'launchkernel' in a['api_name'].lower()];reverse_ok=len(memset_apis)==2 and len(kernel_apis)==2
  for a in aa:
   expected='kernel' if a in kernel_apis else ('positive64' if memset_apis and a is memset_apis[0] else 'zero0')
   # P uses the exact compiled source's fixed call order, not guessed duration.
   if req['mode']=='R':expected={'first':'kernel','second':'kernel','memset64':'positive64','memset0':'zero0'}.get(a['operation'],'unknown')
   mapped=[e for e in ev if e['api_key']==a['api_key']];ok=a.get('returnValue')==0
   if expected=='kernel':ok &= len(mapped)==1 and mapped[0]['kind']=='kernel'
   elif expected=='positive64':ok &= len(mapped)==1 and mapped[0]['kind']=='memset' and mapped[0]['bytes']==64
   elif expected=='zero0':ok &= len(mapped)==0 or all(e['kind']=='memset' and e.get('bytes')==0 for e in mapped)
   else:ok=False
   reverse_ok &= ok
   reverse.append({'iteration':i,'api_name':a['api_name'],'api_key':a['api_key'],'correlationId':a['correlationId'],'return_value':a.get('returnValue'),'operation':a['operation'],'known_source_call':expected,'mapped_GPU_records':len(mapped),'known_zero_byte_explanation':expected=='zero0' and len(mapped)==0,'passed':bool(ok),'provenance':'exact compiled source + R NVTX' if req['mode']=='R' else 'exact compiled source + fixed within-iteration API order'})
  intervals=[(e['start'],e['end']) for e in ev];q=interval_closure(intervals) if intervals else None
  ids=[e['operation'] for e in ev] if req['mode']=='R' else [e['api_key'] for e in ev]
  pp=overlapping_pairs(ev)
  iterations.append({'iteration':i,'kernel_records':sum(e['kind']=='kernel' for e in ev),'memset_records':sum(e['kind']=='memset' for e in ev),'GPU_events':len(ev),'kernel_work_APIs':len(kernel_apis),'memset_work_APIs':len(memset_apis),'submitted_work_accounted':bool(reverse_ok),'end_before_API_entry':sum(e['end']<e['api_start'] for e in ev),'overlap_pairs':len(pp),'event_index_partition_supported':partition_supported(ids),'GPU_end_exceeds_iteration_scope':sum(e['end']>next(s['end'] for s in iters if s['iteration']==i) for e in ev),'GPU_interval_closure':q})
  signatures.append({'iteration':i,'signature':[[e['kind'],e['kernel_name'],[e.get('grid'+x) for x in 'XYZ'],[e.get('block'+x) for x in 'XYZ'],e.get('bytes')] for e in ev]})
  if req['mode']=='R' and ev:
   groups=collections.defaultdict(list)
   for e in ev:groups[e['operation']].append(e)
   if None not in groups:opclosures.append({'iteration':i,**interval_closure([(min(e['start'] for e in g),max(e['end'] for e in g)) for g in groups.values()])})
 inv=[e for e in events if e['end']<e['api_start']];start_inv=[e for e in events if e['start']<e['api_start']];streams={(e['deviceId'],e['contextId'],e['streamId']) for e in events};eventrows=table_rows(c,'CUPTI_ACTIVITY_KIND_CUDA_EVENT',tables)
 csvwrite(out/'reverse_API_coverage.csv',reverse);write(out/'iterations.json',iterations);write(out/'ordered_signatures.json',signatures);write(out/'operator_interval_closure.json',opclosures)
 csvwrite(out/'clock_inversions.csv',inv,['uid','kind','iteration','api_name','api_start','api_end','start','end','correlationId','gpu_end_minus_API_entry_ns'])
 csvwrite(out/'clock_start_before_entry.csv',start_inv,['uid','kind','iteration','api_name','api_start','api_end','start','end','correlationId','gpu_start_minus_API_entry_ns'])
 csvwrite(out/'overlap_pairs.csv',[{'left_uid':a['uid'],'right_uid':b['uid'],'intersection_ns':v,'kind_pair':a['kind']+'->'+b['kind'],'iteration':a['iteration']} for a,b,v in pairs],['left_uid','right_uid','intersection_ns','kind_pair','iteration'])
 gates={}
 gates['recorded_event_integrity']=gate('passed' if integrity=='ok' and not errors and len(streams)==1 and len(events)>0 else 'failed',['events.csv','correlation_errors.csv','schema.json'],'unique process-scoped recorded GPU→API and valid scope/intervals only; expected submission count is checked independently')
 covered=all(i['submitted_work_accounted'] for i in iterations) and len(reverse)==32
 source_name='serial_mre_drain_v2.cu' if req.get('host_drain_ms') else 'serial_mre.cu'
 gates['submitted_work_coverage']=gate('passed' if covered else 'failed',['reverse_API_coverage.csv','../../../native/'+source_name],'known 2 kernels +64B memset +0B no-op each iteration, exact source bytes; native control only')
 sem_status='failed' if not nvtx_exact or (req['mode']=='R' and not covered) else ('warning_accepted' if warnings else 'passed')
 gates['reference_semantic_alignment']=gate(sem_status,['diagnostics.json','iterations.json','safe_metadata.json'],'native source/NVTX known operations; no ATen proof and warnings remain' if warnings else 'native known-source scopes only; no ATen/RN-A reference upgrade')
 gates['timestamp_consistency']=gate('failed' if inv or pairs else ('unverified' if start_inv or not covered else 'passed'),['clock_inversions.csv','clock_start_before_entry.csv','overlap_pairs.csv'],'ordinary non-PDL serial control has end-before-entry or overlapping work' if inv or pairs else ('start-before-entry observed; no end-before-entry, clock/callback origin still uncalibrated' if start_inv else ('no measured causal inversion or serial overlap; not absolute clock calibration' if covered else 'missing known positive work prevents full temporal check')))
 gates['event_index_partition_supported']=gate('passed' if covered and all(i['event_index_partition_supported'] for i in iterations) else 'failed',['iterations.json'],'known native source partitions require complete expected positive work')
 gates['nonoverlap_sum_supported']=gate('failed' if pairs else ('passed' if covered else 'unverified'),['overlap_pairs.csv','operator_interval_closure.json'],'recorded native intervals only; incomplete work prevents full nonoverlap acceptance')
 gates['overlap_aware_closure_supported']=gate('passed' if all(i['GPU_interval_closure'] and i['GPU_interval_closure']['closure_error_ns']==0 for i in iterations) else 'failed',['iterations.json','operator_interval_closure.json'],'union arithmetic of observed intervals only')
 gates['strict_reference_verified']=strict_gate(gates,['quality.json']);write(out/'quality.json',{'version':VERSION,'gates':gates,'strict_verified':0,'formal_training_eligible':False,'timestamp_mutations':0,'scope':'native control; never transfers strict eligibility to RN-A'})
 result={'cell':cell,'mode':req['mode'],'requested_backend':req['requested_backend'],'actual_backend':actual,'backend_evidence':backend_messages,'worker_pid':pid,'schema':schema_version,'sqlite_sha256':sha(folder/'trace.sqlite'),'report_sha256':sha(folder/'trace.nsys-rep'),'correctness_passed':read(folder/'execution.json')['correctness_passed'],'GPU_events':len(events),'GPU_kind_counts':dict(collections.Counter(e['kind'] for e in events)),'GPU_to_API_errors':errors,'NVTX_exact':nvtx_exact,'NVTX_ranges':len(nvtx),'end_before_API_entry':len(inv),'clock_inversion_ns':quant([e['api_start']-e['end'] for e in inv]),'overlap_pairs':len(pairs),'known_zero_byte_APIs_without_GPU':sum(x['known_zero_byte_explanation'] for x in reverse),'all_known_submitted_work_accounted':covered,'warnings':[x['text'] for x in warnings],'CUDA_EVENT_records':len(eventrows),'CUDA_EVENT_nonzero_completion_timestamps':sum(bool(x.get('timestamp')) for x in eventrows),'strict_verified':0,'formal_training_eligible':False}
 result.update(start_before_API_entry=len(start_inv),start_before_API_entry_ns=quant([e['api_start']-e['start'] for e in start_inv]))
 write(out/'summary.json',result);c.close();return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--cells',nargs='+',default=['A1','A2','A3','A4']);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False)
 results=[inspect(cell,out/cell) for cell in a.cells];comparisons=[]
 for pcell,rcell in [('A1','A2'),('A3','A4')]:
  if not {pcell,rcell}.issubset(a.cells):continue
  ps=read(out/pcell/'ordered_signatures.json');rs=read(out/rcell/'ordered_signatures.json');comparisons.append({'P':pcell,'R':rcell,'iterations':8,'exact_signature_iterations':sum(p['signature']==r['signature'] for p,r in zip(ps,rs)),'strict_upgrade':False})
 backends={r['actual_backend'] for r in results};write(out/'SUMMARY.json',{'version':VERSION,'captures':results,'P_R_comparisons':comparisons,'actual_backends':sorted(backends),'backend_contrast_distinct':len(backends)>1 and 'unverified' not in backends,'strict_verified':0,'no_RN_A_reference_upgrade':True,'no_models_trained':True})
 print([{k:r[k] for k in ('cell','actual_backend','GPU_events','end_before_API_entry','overlap_pairs','known_zero_byte_APIs_without_GPU','warnings')} for r in results])
if __name__=='__main__':main()
