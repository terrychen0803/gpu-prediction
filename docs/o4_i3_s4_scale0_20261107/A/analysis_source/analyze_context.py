"""New evaluator-only context analysis; unchanged parser, original timestamps preserved."""
import collections,csv,hashlib,json,statistics,sys
from pathlib import Path
from runtime_check_helpers import check_runtime,semantic,digest
ROOT=Path(__file__).resolve().parent;src=ROOT/'source';sys.path[:0]=[str(src),str(src/'o4_pilot')]
from o4_research.common import read,write,csvwrite,sha
from o4_i3.inventory import union,ownership
from o4_i3_context.assets import plan,work,verify

def spec(shape):
 stride=[];value=1
 for size in reversed(shape):stride.append(value);value*=size
 return {'shape':shape,'stride':list(reversed(stride)),'storage_offset':0,'dtype':'float16'}
def config(n):
 p=n['parameters'];t={'x':spec(n['input_shapes'][0])};kind=n['op']
 if kind=='conv':
  t['weight']=spec(p['weight_shape'])
  if p['bias']:t['bias']=spec([p['weight_shape'][0]])
 if kind=='batchnorm_eval':
  for k in ['weight','bias','running_mean','running_var']:t[k]=spec([n['input_shapes'][0][1]])
 if kind=='add':t['other']=spec(n['input_shapes'][1])
 return {'family':kind,'definition':{'family':kind,'parameters':p,'tensors':t,'output_shape':n['output_shape']}}
def expected_nodes(cid):
 w=work(cid);g=next(g for g in plan()['graphs'] if g['id']==w['graph_id'])
 return [n for n in g['nodes'] if w['kind']!='isolated_position' or n['id']==w['node_id']]
def rows(p):return list(csv.DictReader(p.open()))
def main():
 r=ROOT;state=read(r/'RECOVERY_STATUS.json');assert state['status']!='running'
 out=r/'analysis/summary';out.mkdir(exist_ok=False);verify(r/'assets_context_local/ASSETS.json')
 from o4_i3_v2.report import analyze
 view=r/'analysis/active_run_view';(view/'collections').mkdir(parents=True,exist_ok=False)
 for name in ['cbr_smoke','cbr_formal_recovery','residual_smoke','residual_formal']:
  target=r/'collections'/name
  if target.exists():(view/'collections'/name).symlink_to(target,target_is_directory=True)
 replay=analyze(view,r/'analysis/raw_replay',True)
 modes=[];checks=[];traces=[];owners=[];streams=[];closure=[];samples=collections.defaultdict(list);uses=collections.defaultdict(list);cell_records=[];pids=[];failures=[]
 graphs={g['id']:g for g in plan()['graphs']}
 for invpath in sorted((view/'collections').glob('*/inventory.json')):
  folder=invpath.parent;inv=read(invpath);kind=inv['protocol']['kind'];modes.extend({'collection':folder.name,**g} for g in inv.get('mode_groups',[]))
  for j in inv['jobs']:
   cell=folder/j['folder'];cid=j['config_id'];row={'collection':folder.name,'kind':kind,**j};cell_records.append(row)
   if j['status']!='completed':failures.append(row);continue
   env=read(cell/'environment.json');pids.append({'cell':str(cell),'pid':env['pid']})
   if j['mode']=='GI':continue
   a=cell/'audit';q=read(a/'SUMMARY.json');eng=read(a/'engineering_acceptance.json');traces.append({'collection':folder.name,'config_id':cid,'repeat':j['repeat'],'mode':j['mode'],**q,'actual_backend':eng['actual_backend'],'engineering_blockers':eng['blockers']})
   es=rows(a/'events_diagnostic.csv');streams.append({'cell':str(cell),'streams':sorted({(e['device'],e['context'],e['stream']) for e in es})})
   if j['mode']=='R':
    per=collections.defaultdict(list)
    for op in read(a/'reference_operations.json'):per[(op['stage'],op['iteration'])].append(op)
    for (stage,i),ops in per.items():
     ops=sorted(ops,key=lambda x:x['op_index']);nodes=expected_nodes(cid);scope=f'o4.iter/{stage}/{i:06d}';ev=[e for e in es if e['iteration_scope']==scope]
     own=ownership([op['event_indices'] for op in ops],len(ev));owners.append({'collection':folder.name,'config_id':cid,'stage':stage,'iteration':i,'unassigned':len(own['unassigned']),'multiple':len(own['multiple_owners']),'noncontiguous':sum(not op['contiguous'] for op in ops),'zero_recorded':sum(op['zero_recorded_GPU_work'] for op in ops)})
     if len(ops)!=len(nodes):checks.append({'config_id':cid,'matched':False,'problem':'operation_count','actual':len(ops),'expected':len(nodes)})
     for index,op in enumerate(ops):
      n=nodes[index] if index<len(nodes) else None
      result=check_runtime(config(n),op) if n else {'matched':False,'problem':'extra op'}
      checks.append({'collection':folder.name,'config_id':cid,'repeat':j['repeat'],'stage':stage,'iteration':i,'op_index':op['op_index'],'node_id':n['id'] if n else None,**result})
      if stage=='measure':uses[digest(semantic(op))].append({'config_id':cid,'node_id':n['id'] if n else None,'graph_id':work(cid)['graph_id'],'leakage_group':'context_scale_0','split':'train'})
  if kind!='formal':continue
  for pair in inv.get('P_R_pairs',[]):
   if pair.get('error'):continue
   cid=pair['config_id'];rep=pair['repeat'];pe=rows(folder/f'{cid}_r{rep}_P/audit/events_diagnostic.csv');refs=read(folder/f'{cid}_r{rep}_R/audit/reference_operations.json');by=collections.defaultdict(list)
   for e in pe:by[e['iteration_scope']].append(e)
   refby=collections.defaultdict(list)
   for op in refs:refby[f'o4.iter/{op["stage"]}/{op["iteration"]:06d}'].append(op)
   for a in read(folder/f'{cid}_r{rep}_pair/P_R_ALIGNMENT.json'):
    if not a['ordered_signature_exact'] or '/measure/' not in a['scope']:continue
    scope=a['scope'];ev=by[scope];intervals=[(int(e['start_ns']),int(e['end_ns'])) for e in ev];parts=[]
    for op in refby[scope]:
     ints=[intervals[i] for i in op['event_indices']]
     if not ints:continue
     span=(max(b for a,b in ints)-min(a for a,b in ints))/1000;samples[(cid,op['op_index'],rep)].append(span);parts.append((min(a for a,b in ints),max(b for a,b in ints)))
    if intervals:closure.append({'config_id':cid,'repeat':rep,'scope':scope,'P_operator_span_sum_us':sum(b-a for a,b in parts)/1000,'P_operator_span_union_us':union(parts)/1000,'P_span_overlap_us':(sum(b-a for a,b in parts)-union(parts))/1000,'P_full_gpu_span_us':(max(b for a,b in intervals)-min(a for a,b in intervals))/1000,'P_GPU_event_duration_sum_us':sum(b-a for a,b in intervals)/1000,'P_GPU_busy_union_us':union(intervals)/1000,'union_closure_error_ns':a['P_contextual_union_closure']['closure_error_ns'] if a['P_contextual_union_closure'] else None})
 ctx=[];metrics=[]
 for w in plan()['work_items']:
  cid=w['id'];gs=[g for g in modes if g['config_id']==cid and ('formal' in g['collection'])];by={g['mode']:g for g in gs}
  metrics.append({**w,'leakage_group':'context_scale_0','GI_us':by.get('GI',{}).get('median_of_process_medians_us'),'GI_status':by.get('GI',{}).get('stability_status','not_collected'),'GI_process_medians_us':by.get('GI',{}).get('process_medians_us',[]),'P_event_us':by.get('P',{}).get('median_of_process_medians_us'),'P_event_status':by.get('P',{}).get('stability_status','not_collected'),'R_event_us':by.get('R',{}).get('median_of_process_medians_us'),'R_event_status':by.get('R',{}).get('stability_status','not_collected'),'strict_reference_verified':False})
  for i,n in enumerate(expected_nodes(cid)):
   vals=[samples[(cid,i,rep)] for rep in [1,2,3]];med=[statistics.median(x) for x in vals if x];complete=all(len(x)==40 for x in vals);spread=(max(med)-min(med))/statistics.median(med) if complete else None
   ts=[t for t in traces if t['config_id']==cid and ('formal' in t['collection'])];cc=[c for c in cell_records if c['config_id']==cid and c['kind']=='formal'];oo=[o for o in owners if o['config_id']==cid and ('formal' in o['collection'])];param=[c for c in checks if c['config_id']==cid and ('formal' in c.get('collection',''))];unmatched=sum(t['unmatched_work_APIs'] for t in ts)
   usable=complete and spread<=.05 and len(cc)==9 and all(c['status']=='completed' and c.get('monitor_review') is False for c in cc) and len(ts)==6 and not any(t['engineering_blockers'] for t in ts) and not unmatched and oo and not any(o['unassigned'] or o['multiple'] or o['noncontiguous'] or o['zero_recorded'] for o in oo) and param and all(c['matched'] for c in param)
   ctx.append({'config_id':cid,'graph_id':w['graph_id'],'kind':w['kind'],'op_index':i,'node_id':n['id'],'op':n['op'],'process_medians_us':med,'median_us':statistics.median(med) if med else None,'spread_fraction':spread,'complete_40x3':complete,'stability_status':'stable' if complete and spread<=.05 else 'review_required','unmatched_work_API_count':unmatched,'development_use':'development_usable' if usable else 'limited_use','strict_reference_verified':False,'formal_training_eligible':False})
 comparisons=[];byid={m['id']:m for m in metrics}
 for g in plan()['graphs']:
  d=byid[g['id']];ind=byid[g['independent_control_id']];single=[byid[cid] for cid in g['isolated_control_ids']];complete=all(x['GI_us'] is not None for x in [d,ind]+single)
  comparisons.append({'graph_id':g['id'],'D_GI_us':d['GI_us'],'I_GI_us':ind['GI_us'],'sum_S_GI_us':sum(x['GI_us'] for x in single) if complete else None,'D_minus_I_pct':(d['GI_us']/ind['GI_us']-1)*100 if complete else None,'sum_S_minus_D_pct':(sum(x['GI_us'] for x in single)/d['GI_us']-1)*100 if complete else None,'all_GI_groups_stable':all(x['GI_status']=='stable' for x in [d,ind]+single),'scope':'descriptive same-device medians; not pure dependency causality, not predictor performance'})
 # Exact runtime component cross-check against all prior72 controlled configurations.
 old=ROOT.parent/'o4_i3_s3_lite_gb10_20261106T230230Z';olduses=collections.defaultdict(set)
 for name in ['o4_i3_s1_gb10_20261106T213722Z','o4_i3_s2_gb10_20261106T220336Z',old.name]:
  for p in (ROOT.parent/name/'collections').glob('*/*_R/audit/reference_operations.json'):
   for op in read(p):
    if op['stage']=='measure':olduses[digest(semantic(op))].add(p.parents[1].name.rsplit('_r',1)[0])
 overlap=[{'semantic_sha256':key,'context_uses':list({(x['config_id'],x['node_id']) for x in uses[key]}),'prior_configs':sorted(olduses[key])} for key in uses if key in olduses]
 duplicates=[{'semantic_sha256':key,'positions':list({(u['config_id'],u['node_id']) for u in value}),'leakage_group':'context_scale_0','reason':'expected shared full runtime signature; controls/positions not independent groups'} for key,value in uses.items() if len({(u['config_id'],u['node_id']) for u in value})>1]
 for n,x in [('CELLS',cell_records),('MODE_STABILITY',modes),('WORK_METRICS',metrics),('R_POSITION_CHECKS',checks),('P_CTX_STABILITY',ctx),('TRACE_QUALITY',traces),('OWNERSHIP',owners),('P_CLOSURE',closure),('D_I_S_COMPARISON',comparisons)]:
  if x:csvwrite(out/(n+'.csv'),x)
 write(out/'SHARED_RUNTIME_COMPONENTS.json',duplicates);write(out/'PRIOR72_COMPONENT_OVERLAP.json',overlap);write(out/'STREAMS.json',streams);write(out/'PID_AUDIT.json',{'workers':len(pids),'distinct_PIDs':len({p['pid'] for p in pids}),'rows':pids});write(out/'FAILURES.json',failures)
 s={'collected_work_items':len({c['config_id'] for c in cell_records if c['status']=='completed'}),'completed_smoke_cells':sum(c['status']=='completed' and c['kind']=='smoke' for c in cell_records),'completed_formal_cells':sum(c['status']=='completed' and c['kind']=='formal' for c in cell_records),'raw_replay':read(r/'analysis/raw_replay/REPLAY.json'),'formal_mode_stable':sum(('formal' in g['collection']) and g['stability_status']=='stable' for g in modes),'formal_mode_review':sum(('formal' in g['collection']) and g['stability_status']!='stable' for g in modes),'R_parameter_rows':len(checks),'R_parameter_rows_matched':sum(c['matched'] for c in checks),'P_CTX_positions':len(ctx),'P_CTX_stable':sum(c['stability_status']=='stable' for c in ctx),'P_CTX_review':sum(c['stability_status']!='stable' for c in ctx),'P_CTX_development_usable':sum(c['development_use']=='development_usable' for c in ctx),'prior72_runtime_overlap_keys':len(overlap),'shared_context_component_keys':len(duplicates),'unmatched_work_APIs':sum(t['unmatched_work_APIs'] for t in traces),'NVTX_warning_traces':sum(bool(t['warnings']) for t in traces),'clock_start_before_API':sum(t['clock_start_before_API_entry'] for t in traces),'clock_end_before_API':sum(t['clock_end_before_API_entry'] for t in traces),'overlap_pairs':sum(t['overlap_pairs'] for t in traces),'unassigned_events':sum(o['unassigned'] for o in owners),'multiple_owners':sum(o['multiple'] for o in owners),'noncontiguous_ops':sum(o['noncontiguous'] for o in owners),'zero_recorded_ops':sum(o['zero_recorded'] for o in owners),'max_union_closure_error_ns':max((x['union_closure_error_ns'] for x in closure if x['union_closure_error_ns'] is not None),default=None),'strict_reference_verified':False,'formal_training_eligible':False,'models_trained':0,'independent_context_groups':1,'OOF_ready':False}
 # Avoid duplicating the full replay list in the summary.
 s['raw_replay']={k:v for k,v in s['raw_replay'].items() if k!='checks'};write(out/'SUMMARY.json',s);print(json.dumps(s,indent=2))
if __name__=='__main__':main()
