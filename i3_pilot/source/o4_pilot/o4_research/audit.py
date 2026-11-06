"""Read-only v4 audit with explicit out-of-iteration preparation accounting."""
import bisect,collections,json,re,sqlite3
from pathlib import Path
from .common import read,write,csvwrite,sha
from .quality_v4 import interval_closure,overlapping_pairs,partition_supported,gate,strict_gate,VERSION
from .nsys_schema import validate_export_schema
ITER=re.compile(r'^o4\.iter/(transition|measure)/(\d{6})$');PREP=re.compile(r'^o4\.prepare/(transition|measure)/(\d{6})$');OP=re.compile(r'^o4\.op/(transition|measure)/(\d{6})/(\d{4})$')
SAFE={'event_kind','start_ns','end_ns','api_start_ns','api_end_ns','grid_xyz','block_xyz','dynamic_shared_memory_bytes','bytes','copy_kind','memory_kind','memset_value','event_id','api_event_id','correlation_id','stream_id','context_id','process_id'}
def safe(e,i):
 x={k:v for k,v in e.items() if k in SAFE};x.update(event_id=f'e{i:06d}',api_event_id=f'a{i:06d}',correlation_id=f'c{i:06d}',stream_id='s0',context_id='c0',process_id='p0');return x
def signature(e):return [e.get(k) for k in ('event_kind','kernel_name','grid_xyz','block_xyz','dynamic_shared_memory_bytes','bytes','copy_kind','memory_kind','memset_value')]
class ScopeLookup:
 """Exact single-owner time lookup; nested scopes rejected, never guessed."""
 def __init__(self,scopes):
  grouped=collections.defaultdict(list)
  for scope in scopes:grouped[scope['globalTid']].append(scope)
  self.groups={}
  for tid,values in grouped.items():
   values.sort(key=lambda x:(x['start'],x['end']))
   if any(a['end']>b['start'] for a,b in zip(values,values[1:])):
    raise ValueError('Overlapping/nested scopes of same class; cannot use unique ownership lookup')
   self.groups[tid]=([x['start'] for x in values],values)
 def contains(self,api):
  group=self.groups.get(api['globalTid'])
  if group is None:return []
  starts,values=group
  idx=bisect.bisect_right(starts,api['start'])-1
  if idx<0:return []
  x=values[idx]
  return [x] if x['start']<=api['start'] and api['end']<=x['end'] else []
def contains(a,scopes):
 return scopes.contains(a) if isinstance(scopes,ScopeLookup) else [s for s in scopes if s['globalTid']==a['globalTid'] and s['start']<=a['start'] and a['end']<=s['end']]
def audit(folder,out):
 folder=Path(folder);out=Path(out);out.mkdir(exist_ok=False);req=read(folder/'request.json');env=read(folder/'environment.json');pid=env['pid'];c=sqlite3.connect('file:'+str(folder/'trace.sqlite')+'?mode=ro',uri=True);c.execute('pragma query_only=ON');c.row_factory=sqlite3.Row
 try:
  tables={x[0] for x in c.execute("select name from sqlite_master where type='table'")}
  def rows(t):return [dict(x) for x in c.execute('select rowid as _rowid,* from "'+t+'"')] if t in tables else []
  schema={t:[dict(x) for x in c.execute('pragma table_info("'+t+'")')] for t in sorted(tables)};write(out/'schema.json',schema);export=rows('META_DATA_EXPORT');version=next(x['value'] for x in export if x['name']=='EXPORT_SCHEMA_VERSION')
  compat=validate_export_schema(version,schema)
  integrity=c.execute('pragma quick_check').fetchone()[0]
  if integrity!='ok':raise ValueError('Nsight SQLite integrity check failed: '+str(integrity))
  compat['sqlite_integrity']=integrity;write(out/'schema_compatibility.json',compat)
  strings={x['id']:x['value'] for x in rows('StringIds')};diags=rows('DIAGNOSTIC_EVENT');write(out/'diagnostics.json',diags);warnings=[x['text'] for x in diags if x.get('severity',0)>=2];iters=[];preps=[];ops=[]
  for n in rows('NVTX_EVENTS'):
   if ((n['globalTid']>>24)&0xffffff)!=pid:continue
   n['name']=n['text'] or strings.get(n['textId'],'')
   for rx,dest in [(ITER,iters),(PREP,preps),(OP,ops)]:
    match=rx.fullmatch(n['name'])
    if match:
     if n['end'] is None or n['end']<n['start']:raise ValueError('Unclosed/invalid scope')
     n.update(stage=match[1],iteration=int(match[2]));dest.append(n)
  dispatch=[json.loads(x) for x in (folder/'reference_dispatch.jsonl').read_text().splitlines()] if req['mode']=='R' else []
  byname={x['scope']:x for x in dispatch}
  p=req.get('protocol',{})
  transition=p.get('transition_iterations',1);measured=p.get('measured_iterations',3)
  if not isinstance(transition,int) or not isinstance(measured,int) or not 0<transition<=20 or not 0<measured<=100:
   raise ValueError('Unexpected or unbounded iteration protocol')
  expected={f'o4.iter/{stage}/{i:06d}' for stage,n in [('transition',transition),('measure',measured)] for i in range(n)}
  scope_exact=len(iters)==len(expected) and {x['name'] for x in iters}==expected
  reference_exact=len(ops)==len(dispatch)==len(byname) and {x['name'] for x in ops}==set(byname)
  if not scope_exact:raise ValueError('Iteration NVTX scope set inconsistent with request protocol')
  if req['mode']=='R' and not reference_exact:raise ValueError('R op NVTX/dispatch scope sets differ')
  iter_lookup=ScopeLookup(iters);prep_lookup=ScopeLookup(preps);op_lookup=ScopeLookup(ops)
  apis=[];idx=collections.defaultdict(list)
  for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
   for a in rows(table):
    if ((a['globalTid']>>24)&0xffffff)!=pid:continue
    a.update(name=strings.get(a['nameId'],''),table=table,key=f'{table}:{a["_rowid"]}');ins=contains(a,iter_lookup);prep=contains(a,prep_lookup);op=contains(a,op_lookup);a.update(iter_scope=ins[0]['name'] if len(ins)==1 else None,prep_scope=prep[0]['name'] if len(prep)==1 else None,op_scope=op[0]['name'] if len(op)==1 else None);apis.append(a);idx[(a['globalTid']>>24,a['correlationId'])].append(a)
  events=[];errors=[]
  for table,kind in [('CUPTI_ACTIVITY_KIND_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL','kernel'),('CUPTI_ACTIVITY_KIND_MEMSET','memset'),('CUPTI_ACTIVITY_KIND_MEMCPY','copy')]:
   for e in rows(table):
    if ((e['globalPid']>>24)&0xffffff)!=pid:continue
    matches=idx[(e['globalPid']>>24,e['correlationId'])]
    if len(matches)!=1:errors.append({'table':table,'rowid':e['_rowid'],'correlationId':e['correlationId'],'matches':len(matches)});continue
    a=matches[0]
    if e['end']<=e['start']:raise ValueError('Nonpositive GPU interval')
    if not a['iter_scope'] and not a['prep_scope']:errors.append({'table':table,'rowid':e['_rowid'],'reason':'unscoped_recorded_GPU'})
    events.append({'uid':f'{table}:{e["_rowid"]}','start':e['start'],'end':e['end'],'event_kind':kind,'start_ns':e['start'],'end_ns':e['end'],'api_start_ns':a['start'],'api_end_ns':a['end'],'api_key':a['key'],'API_name':a['name'],'correlation_id_raw':e['correlationId'],'iteration_scope':a['iter_scope'],'preparation_scope':a['prep_scope'],'op_scope':a['op_scope'],'kernel_name':strings.get(e.get('demangledName'),''),'grid_xyz':[e.get('grid'+k) for k in 'XYZ'] if kind=='kernel' else None,'block_xyz':[e.get('block'+k) for k in 'XYZ'] if kind=='kernel' else None,'dynamic_shared_memory_bytes':e.get('dynamicSharedMemory'),'bytes':e.get('bytes'),'copy_kind':e.get('copyKind'),'memory_kind':e.get('memKind'),'memset_value':e.get('value') if kind=='memset' else None,'device':e['deviceId'],'context':e['contextId'],'stream':e['streamId']})
  events.sort(key=lambda e:(e['start'],e['end'],e['uid']))
  csvwrite(out/'events_diagnostic.csv',events);write(out/'correlation_errors.json',errors)
  keys={e['api_key'] for e in events};orphans=[a for a in apis if any(k in a['name'].lower() for k in ('launchkernel','memsetasync','memcpy')) and a['key'] not in keys];csvwrite(out/'unmatched_work_APIs.csv',orphans);per=[];references=[]
  by_iteration=collections.defaultdict(list)
  by_op_by_stage=collections.defaultdict(list)
  for e in events:
   if e['iteration_scope']:by_iteration[e['iteration_scope']].append(e)
  for op in ops:by_op_by_stage[(op['stage'],op['iteration'])].append(op)
  for it in sorted(iters,key=lambda n:n['start']):
   ev=by_iteration[it['name']];owns=[e['op_scope'] for e in ev];parts=[];seq=[]
   ids_by_op=collections.defaultdict(list)
   for j,e in enumerate(ev):
    if e['op_scope']:ids_by_op[e['op_scope']].append(j)
   for op in by_op_by_stage[(it['stage'],it['iteration'])]:
    ids=ids_by_op[op['name']];span=(max(ev[j]['end'] for j in ids)-min(ev[j]['start'] for j in ids)) if ids else None;ref=byname.get(op['name'],{})
    references.append({**ref,'scope':op['name'],'event_indices':ids,'contiguous':not ids or ids==list(range(ids[0],ids[-1]+1)),'recorded_GPU_event_count':len(ids),'recorded_contextual_span_ns':span,'zero_recorded_GPU_work':not ids,'label_status':'development_unverified'})
    if ids:parts.append((min(ev[j]['start'] for j in ids),max(ev[j]['end'] for j in ids)))
   closure=interval_closure(parts) if parts else None
   record={'scope':it['name'],'stage':it['stage'],'iteration':it['iteration'],'events':len(ev),'recorded_GPU_kind_counts':dict(collections.Counter(e['event_kind'] for e in ev)),'end_before_API_entry':sum(e['end']<e['api_start_ns'] for e in ev),'start_before_API_entry':sum(e['start']<e['api_start_ns'] for e in ev),'overlap_pairs':len(overlapping_pairs(ev)),'event_index_partition_supported':partition_supported(owns) if req['mode']=='R' else None,'operator_union_closure':closure};per.append(record)
   if ev:
    predictor=out/'predictor'/f'{it["stage"]}_{it["iteration"]:06d}.jsonl';predictor.parent.mkdir(exist_ok=True)
    with predictor.open('w') as f:
     for i,e in enumerate(ev):f.write(json.dumps(safe(e,i))+'\n')
  write(out/'iteration_audit.json',per);write(out/'reference_operations.json',references)
  start_inv=sum(e['start']<e['api_start_ns'] for e in events);end_inv=sum(e['end']<e['api_start_ns'] for e in events);streams={(e['device'],e['context'],e['stream']) for e in events};gs={}
  gs['recorded_event_integrity']=gate('passed' if not errors and scope_exact and len(streams)==1 else 'failed',['events_diagnostic.csv','correlation_errors.json'],'recorded unique correlations + explicit iteration/preparation scopes; not submitted-work completeness')
  gs['submitted_work_coverage']=gate('unverified' if orphans else 'passed',['unmatched_work_APIs.csv'] if orphans else ['events_diagnostic.csv'],'unmatched API parameters absent from this raw; standalone op140 evidence cannot relabel historical or this capture')
  gs['reference_semantic_alignment']=gate(('failed' if not reference_exact else 'warning_accepted' if warnings else 'unverified') if req['mode']=='R' else 'unverified',['reference_operations.json','diagnostics.json'],'ATen reference only in R; P requires exact same-device ordered-signature transfer; warnings never auto-upgrade')
  gs['timestamp_consistency']=gate('unverified' if start_inv or end_inv else 'passed',['events_diagnostic.csv'],'raw timestamp values preserved; start/end inversions retained; no fixed offset')
  gs['event_index_partition_supported']=gate('passed' if req['mode']=='R' and all(x['event_index_partition_supported'] for x in per) else 'unverified',['iteration_audit.json'],'recorded R partitions, preparation excluded by explicit known scope; P not yet transferred')
  closures=[x['operator_union_closure'] for x in per if x['operator_union_closure']]
  gs['nonoverlap_sum_supported']=gate('failed' if any(x['union_overlap_ns'] for x in closures) else ('passed' if closures else 'unverified'),['iteration_audit.json'],'observed operator spans only; not strict reference qualification')
  gs['overlap_aware_closure_supported']=gate('passed' if closures and all(x['closure_error_ns']==0 for x in closures) else 'unverified',['iteration_audit.json'],'generalized union arithmetic only')
  gs['strict_reference_verified']=gate('unverified',['diagnostics.json','iteration_audit.json'],'no formal reference upgrade; all engineering inventory')
  write(out/'v4_quality.json',{'version':VERSION,'gates':gs,'strict_verified':0,'formal_training_eligible':False})
  summary={'status':'recorded_structure_complete' if gs['recorded_event_integrity']['status']=='passed' else 'failed','mode':req['mode'],'schema':version,'sqlite_sha256':sha(folder/'trace.sqlite'),'GPU_records':len(events),'iteration_GPU_records':sum(bool(e['iteration_scope']) for e in events),'preparation_GPU_records':sum(bool(e['preparation_scope']) for e in events),'GPU_kind_counts':dict(collections.Counter(e['event_kind'] for e in events)),'iteration_count':len(iters),'reference_op_count':len(references),'zero_recorded_GPU_ops':sum(x['zero_recorded_GPU_work'] for x in references),'maximum_op_event_count':max((x['recorded_GPU_event_count'] for x in references),default=None),'operations_over_L4':sum(x['recorded_GPU_event_count']>4 for x in references),'unmatched_work_APIs':len(orphans),'clock_start_before_API_entry':start_inv,'clock_end_before_API_entry':end_inv,'overlap_pairs':len(overlapping_pairs(events)),'warnings':warnings,'actual_backend':[x['text'] for x in diags if 'trace was collected' in x['text'] or 'tracing used' in x['text']],'strict_verified':0,'schema_compatibility':'structural_contract_passed_semantics_unverified'};write(out/'SUMMARY.json',summary);return summary
 finally:c.close()
def pair(paudit,raudit,out):
 paudit=Path(paudit);raudit=Path(raudit);out=Path(out);out.mkdir(exist_ok=False);p=list(__import__('csv').DictReader((paudit/'events_diagnostic.csv').open()));r=list(__import__('csv').DictReader((raudit/'events_diagnostic.csv').open()));refs=read(raudit/'reference_operations.json');results=[]
 for scope in [x['scope'] for x in read(paudit/'iteration_audit.json')]:
  pe=[e for e in p if e['iteration_scope']==scope];re=[e for e in r if e['iteration_scope']==scope];same=[signature(e) for e in pe]==[signature(e) for e in re];parts=[]
  if same:
   for ref in [x for x in refs if x['stage']==ITER.fullmatch(scope)[1] and int(x['iteration'])==int(ITER.fullmatch(scope)[2])]:
    ev=[pe[i] for i in ref['event_indices']]
    if ev:parts.append((min(int(e['start']) for e in ev),max(int(e['end']) for e in ev)))
  results.append({'scope':scope,'P_events':len(pe),'R_events':len(re),'ordered_signature_exact':same,'P_contextual_union_closure':interval_closure(parts) if parts else None,'semantic_reference_status':'warning_accepted_conditional' if same else 'unmatched','strict_verified':False})
 write(out/'P_R_ALIGNMENT.json',results);return results
