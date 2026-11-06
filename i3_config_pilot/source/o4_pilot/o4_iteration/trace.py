"""Read-only SQLite audit; safe P export and evaluator-only ATen inventory."""
from __future__ import annotations
import collections,re,sqlite3
from pathlib import Path
from .common import read,write,jsonl,sha
from o1_operator.trace_audit import pid_of,require,rows,table_columns
ITER=re.compile(r'^o4\.iter/(transition|measure)/(\d{6})$')
OP=re.compile(r'^o4\.op/(transition|measure)/(\d{6})/(\d{4})$')
SAFE={'event_id','event_kind','start_ns','end_ns','api_start_ns','api_end_ns','stream_id','context_id','process_id','api_event_id','correlation_id','grid_xyz','block_xyz','dynamic_shared_memory_bytes','bytes','copy_kind','memory_kind','memset_value'}
def safe_event(event,origin,index):
 out={k:v for k,v in event.items() if k in SAFE}
 for k in ('start_ns','end_ns','api_start_ns','api_end_ns'):out[k]-=origin
 out.update(event_id=f'e{index:06d}',api_event_id=f'a{index:06d}',correlation_id=f'c{index:06d}',stream_id='s0',context_id='c0',process_id='p0')
 return out

def union_ns(intervals):
 total=0;end=None
 for a,b in sorted(intervals):
  if b<=a:raise ValueError('Invalid interval')
  total+=b-a if end is None or a>=end else max(0,b-end);end=b if end is None else max(end,b)
 return total

def closure(groups):
 groups=[g for g in groups if g]
 spans=[(g[0]['start_ns'],g[-1]['end_ns']) for g in groups]
 if not spans:raise ValueError('No GPU work in scope')
 if any(a[1]>b[0] for a,b in zip(spans,spans[1:])):raise ValueError('Overlapping/noncontiguous operation groups')
 span=spans[-1][1]-spans[0][0];parts=sum(b-a for a,b in spans);gaps=sum(b[0]-a[1] for a,b in zip(spans,spans[1:]))
 return {'iteration_trace_span_us':span/1000,'operation_span_sum_us':parts/1000,'inter_operation_gap_us':gaps/1000,'closure_error_ns':span-parts-gaps}

def closure_diagnostic(groups):
 groups=[g for g in groups if g]
 if not groups:raise ValueError('No GPU work')
 spans=[(min(e['start_ns'] for e in g),max(e['end_ns'] for e in g)) for g in groups]
 span=max(b for a,b in spans)-min(a for a,b in spans);parts=sum(b-a for a,b in spans)
 overlap=sum(max(0,a[1]-b[0]) for a,b in zip(spans,spans[1:]));gaps=sum(max(0,b[0]-a[1]) for a,b in zip(spans,spans[1:]))
 supported=not overlap and all(a[0]<=b[0] for a,b in zip(spans,spans[1:]))
 return {'iteration_trace_span_us':span/1000,'operation_span_sum_us':parts/1000,'inter_operation_gap_us':gaps/1000,'inter_operation_overlap_us':overlap/1000,'composition_supported':supported,'closure_error_ns':span-parts-gaps if supported else None}

def audit(folder,dest):
 folder=Path(folder);dest=Path(dest);dest.mkdir(parents=True,exist_ok=False)
 req=read(folder/'request.json');env=read(folder/'environment.json');quality=read(folder/'quality.json')
 result={'status':'failed','mode':req['mode'],'errors':[],'warnings':[],'formal_reference_verified':False,'sqlite_sha256':sha(folder/'trace.sqlite'),'report_sha256':sha(folder/'trace.nsys-rep')}
 c=sqlite3.connect(f'file:{folder/"trace.sqlite"}?mode=ro',uri=True);c.row_factory=sqlite3.Row
 try:
  if quality['status']!='completed':raise ValueError('Worker not completed')
  if c.execute('pragma quick_check').fetchone()[0]!='ok':raise ValueError('SQLite integrity failure')
  require(c,'StringIds',['id','value']);strings={r['id']:r['value'] for r in rows(c,'StringIds')}
  diagnostics=rows(c,'DIAGNOSTIC_EVENT') if table_columns(c,'DIAGNOSTIC_EVENT') else []
  write(dest/'diagnostics.json',diagnostics)
  result['warnings']=[r.get('text','') for r in diagnostics if r.get('severity',0)>=2]
  require(c,'CUPTI_ACTIVITY_KIND_RUNTIME',['start','end','globalTid','correlationId','nameId'])
  apis=[r for r in rows(c,'CUPTI_ACTIVITY_KIND_RUNTIME') if pid_of(r['globalTid'])==env['pid']]
  api_index=collections.defaultdict(list)
  for a in apis:
   if a['end']<a['start']:raise ValueError('Invalid API interval')
   name=strings.get(a['nameId'],'').lower()
   if any(x in name for x in ('launch','memcpy','memset')):api_index[a['correlationId']].append(a)
  populated=[t for t in ('CUPTI_ACTIVITY_KIND_KERNEL','CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL') if table_columns(c,t) and c.execute(f'SELECT 1 FROM {t} LIMIT 1').fetchone()]
  if len(populated)!=1:raise ValueError('Expected one populated kernel table')
  events=[];devices=set();contexts=set();streams=set();schema={}
  for table,kind in [(populated[0],'kernel'),('CUPTI_ACTIVITY_KIND_MEMCPY','copy'),('CUPTI_ACTIVITY_KIND_MEMSET','memset')]:
   columns=table_columns(c,table);schema[table]=sorted(columns)
   if not columns:continue
   require(c,table,['start','end','globalPid','correlationId','deviceId','contextId','streamId'])
   if kind=='kernel':require(c,table,['demangledName','gridX','gridY','gridZ','blockX','blockY','blockZ','dynamicSharedMemory'])
   for e in rows(c,table):
    if pid_of(e['globalPid'])!=env['pid']:continue
    devices.add(e['deviceId']);contexts.add(e['contextId']);streams.add(e['streamId'])
    # Exact process-scoped correlation is authoritative. CPU/GPU timestamps
    # may differ at sub-us scale; do not discard a unique correlation by clocks.
    candidates=api_index[e['correlationId']]
    if len(candidates)!=1:raise ValueError('Ambiguous/missing runtime correlation')
    a=candidates[0]
    if e['end']<=e['start']:raise ValueError('Nonpositive GPU work interval')
    item={'event_kind':kind,'start_ns':e['start'],'end_ns':e['end'],'api_start_ns':a['start'],'api_end_ns':a['end'],'api_tid':a['globalTid'],'kernel_name':strings.get(e.get('demangledName'),''),'grid_xyz':[e['grid'+x] for x in 'XYZ'] if kind=='kernel' else None,'block_xyz':[e['block'+x] for x in 'XYZ'] if kind=='kernel' else None,'dynamic_shared_memory_bytes':e.get('dynamicSharedMemory'),'bytes':e.get('bytes'),'copy_kind':e.get('copyKind'),'memory_kind':e.get('memKind'),'memset_value':e.get('value') if kind=='memset' else None,'raw_correlation_id':e['correlationId']}
    events.append(item)
  write(dest/'trace_schema.json',schema)
  if not events or any(len(x)!=1 for x in (devices,contexts,streams)):raise ValueError('Unsupported missing work/multiple devices/contexts/streams')
  result['gpu_end_before_cpu_api_start_count']=sum(e['end_ns']<e['api_start_ns'] for e in events)
  result['maximum_cpu_gpu_clock_inversion_ns']=max(0,max((e['api_start_ns']-e['end_ns'] for e in events),default=0))
  events.sort(key=lambda x:(x['start_ns'],x['end_ns']))
  result['overlapping_adjacent_work_events']=sum(a['end_ns']>b['start_ns'] for a,b in zip(events,events[1:]))
  result['maximum_adjacent_overlap_ns']=max(0,max((a['end_ns']-b['start_ns'] for a,b in zip(events,events[1:])),default=0))
  result['composition_supported']=result['overlapping_adjacent_work_events']==0
  result['trace_schema_version']='o4_audit_v2_overlap_explicit'
  require(c,'NVTX_EVENTS',['start','end','text','textId','globalTid'])
  iters=[];ops=[]
  for r in rows(c,'NVTX_EVENTS'):
   if pid_of(r['globalTid'])!=env['pid']:continue
   name=r['text'] or strings.get(r['textId'],'')
   if ITER.fullmatch(name or '') or OP.fullmatch(name or ''):
    if r['end'] is None or r['end']<=r['start']:raise ValueError('Unclosed NVTX scope')
    r['name']=name;(iters if ITER.fullmatch(name) else ops).append(r)
  p=req['protocol'];expected={f'o4.iter/{stage}/{i:06d}' for stage,n in [('transition',p['transition_iterations']),('measure',p['measured_iterations'])] for i in range(n)}
  if len(iters)!=len(expected) or {r['name'] for r in iters}!=expected:raise ValueError('Iteration scope inventory missing/duplicate')
  if req['mode']=='P' and ops:raise ValueError('P unexpectedly contains operator truth markers')
  def contained(e,scope):return e['api_tid']==scope['globalTid'] and scope['start']<=e['api_start_ns'] and e['api_end_ns']<=scope['end']
  by_iter={r['name']:[] for r in iters};op_events=collections.defaultdict(list)
  for index,e in enumerate(events):
   e['event_id']=f'e{index:06d}';matches=[s for s in iters if contained(e,s)]
   if len(matches)!=1:raise ValueError('GPU event has ambiguous/missing iteration scope')
   s=matches[0]
   if e['end_ns']>s['end']:raise ValueError('GPU work exceeds iteration completion')
   e['iteration_scope']=s['name'];by_iter[s['name']].append(e)
   if req['mode']=='R':
    matches=[op for op in ops if contained(e,op)]
    if len(matches)!=1:raise ValueError('R GPU event has missing/nested operation scope')
    op_events[matches[0]['name']].append(e)
  metadata={}
  if req['mode']=='R':
   dispatch=[__import__('json').loads(s) for s in (folder/'reference_dispatch.jsonl').read_text().splitlines()]
   metadata={x['scope']:x for x in dispatch}
   if len(metadata)!=len(dispatch) or set(metadata)!={x['name'] for x in ops}:raise ValueError('Dispatch/NVTX inventory mismatch')
  reference=[];scope_rows=[]
  for i,s in enumerate(sorted(iters,key=lambda x:x['start'])):
   ev=by_iter[s['name']]
   if not ev:raise ValueError('Iteration has no GPU work')
   match=ITER.fullmatch(s['name']);stage,idx=match.groups();origin=ev[0]['start_ns'];opaque=f'capture_{req["opaque_id"]}_{i:03d}'
   groups=[]
   if req['mode']=='R':
    subops=sorted([op for op in ops if s['start']<=op['start'] and op['end']<=s['end']],key=lambda x:x['start'])
    used=[]
    for op in subops:
     mapped=op_events[op['name']];ids=[ev.index(e) for e in mapped]
     contiguous=not ids or ids==list(range(ids[0],ids[-1]+1))
     used+=ids
     row={**metadata[op['name']],'event_indices':ids,'event_count':len(ids),'contiguous_event_indices':contiguous,'zero_gpu_work':not bool(mapped),'contextual_trace_span_us':(max(e['end_ns'] for e in mapped)-min(e['start_ns'] for e in mapped))/1000 if mapped else None,'gpu_busy_union_us':union_ns([(x['start_ns'],x['end_ns']) for x in mapped])/1000 if mapped else 0,'reference_status':'warning_accepted' if result['warnings'] else 'structural_only_pending_review'}
     reference.append(row)
     if mapped:groups.append(mapped)
    if sorted(used)!=list(range(len(ev))) or len(set(used))!=len(used):raise ValueError('R work coverage incomplete/double counted')
   else:groups=[[e] for e in ev]
   detail=closure_diagnostic(groups);detail.update(stage=stage,iteration=int(idx),opaque_id=opaque,event_count=len(ev),kernel_count=sum(e['event_kind']=='kernel' for e in ev),copy_count=sum(e['event_kind']=='copy' for e in ev),memset_count=sum(e['event_kind']=='memset' for e in ev),gpu_busy_union_us=union_ns([(e['start_ns'],e['end_ns']) for e in ev])/1000)
   scope_rows.append(detail)
   if req['mode']=='P' and stage=='measure':
    safe=dest/'predictor'/opaque;jsonl(safe/'events.jsonl',[safe_event(e,origin,j) for j,e in enumerate(ev)]);write(safe/'scope.json',{'schema':'o4_p_numeric_v1','known_iteration_scope':True,'event_count':len(ev),'start_ns':0,'end_ns':max(e['end_ns'] for e in ev)-origin,'composition_supported':not any(a['end_ns']>b['start_ns'] for a,b in zip(ev,ev[1:])),'stream_policy':'single_observed_stream','source_sqlite_sha256':result['sqlite_sha256']})
  jsonl(dest/'events_diagnostic.jsonl',events);jsonl(dest/'reference_operations.jsonl',reference);write(dest/'iteration_scopes.json',scope_rows)
  result.update(status='warning_accepted' if result['warnings'] else 'structural_only_pending_review',kernel_count=sum(e['event_kind']=='kernel' for e in events),copy_count=sum(e['event_kind']=='copy' for e in events),memset_count=sum(e['event_kind']=='memset' for e in events),device_count=1,context_count=1,stream_count=1,iteration_count=len(iters),reference_operations=len(reference),zero_gpu_operations=sum(x['zero_gpu_work'] for x in reference),work_event_count=len(events),mapped_work_events=len(events) if req['mode']=='R' else None,all_work_assigned_once=True if req['mode']=='R' else None,maximum_reference_event_count=max((x['event_count'] for x in reference),default=None),reference_contiguous=all(x['contiguous_event_indices'] for x in reference) if reference else None,all_closure_errors_zero=all(x['closure_error_ns']==0 for x in scope_rows),unsupported_composition_iterations=sum(not x['composition_supported'] for x in scope_rows))
 except Exception as exc:result['errors'].append(f'{type(exc).__name__}: {exc}')
 finally:c.close();write(dest/'quality.json',result)
 return result
