"""Aggregate completed/failed S3 cells and S1 peers without changing source runs."""
import collections,csv,datetime,gzip,hashlib,json,re,statistics,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
S1=ROOT.parent/'o4_i3_s1_gb10_20261106T213722Z'
S2=ROOT.parent/'o4_i3_s2_gb10_20261106T220336Z'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def csvwrite(p,rows):
    rows=list(rows)
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict,tuple)) else v for k,v in r.items()})
def normalize(x):
    if isinstance(x,dict):return {k:normalize(v) for k,v in x.items() if k not in ('device','requires_grad')}
    if isinstance(x,list):return [normalize(v) for v in x]
    return x
def semantic(op):return {'aten':op['aten'],'arguments':normalize(op.get('bound_arguments',{})),'output':normalize(op.get('output_metadata'))}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def check_runtime(c,op):
    d=c['definition'];family=d['family'];p=d['parameters'];spec=d['tensors'];args=op.get('bound_arguments',{})
    input_key='input' if family in ('conv','depthwise','linear','batchnorm_eval','bilinear_resize') else 'self'
    mapping={'x':input_key}
    scalar={}
    expected={'conv':'aten.conv2d.default','depthwise':'aten.conv2d.default','gemm':'aten.mm.default',
        'bmm':'aten.bmm.default','linear':'aten.linear.default','batchnorm_eval':'aten.batch_norm.default',
        'relu':'aten.relu.default','add':'aten.add.Tensor','maxpool':'aten.max_pool2d.default',
        'adaptive_average_pool':'aten.adaptive_avg_pool2d.default','bilinear_resize':'aten.upsample_bilinear2d.vec','concat':'aten.cat.default'}
    if op['aten']!=expected[family]:return {'matched':False,'reason':'runtime interface differs; manual review required','ATen':op['aten']}
    if family in ('conv','depthwise','linear','batchnorm_eval'):
        mapping.update({n:n for n in spec if n!='x'})
    if family in ('gemm','bmm'):mapping['weight']='mat2'
    if family=='add':mapping['other']='other';scalar={'alpha':p['alpha']}
    if family in ('conv','depthwise'):scalar={k:p[k] for k in ['stride','padding','dilation','groups']}
    if family=='batchnorm_eval':scalar={k:p[k] for k in ['training','eps','momentum']}
    if family=='maxpool':scalar={k:p[k] for k in ['kernel_size','stride','padding','dilation','ceil_mode']}
    if family=='adaptive_average_pool':scalar={'output_size':p['output_size']}
    if family=='bilinear_resize':scalar={'output_size':p['size'],'align_corners':p['align_corners'],'scale_factors':None}
    tensors={n:args.get(key) for n,key in mapping.items()}
    if family=='concat':
        values=args.get('tensors',[])
        tensors={'x':values[0] if len(values)>0 else None,'other':values[1] if len(values)>1 else None};scalar={'dim':p['dim']}
    problems=[]
    for n,s in spec.items():
        actual=tensors.get(n)
        if not isinstance(actual,dict):problems.append(n+':tensor missing');continue
        for key in ['shape','stride','storage_offset']:
            if actual.get(key)!=s[key]:problems.append(n+':'+key)
        if actual.get('dtype')!='torch.float16':problems.append(n+':dtype')
    for key,value in scalar.items():
        if args.get(key)!=value:problems.append('parameter:'+key)
    if family in ('conv','depthwise','linear') and not p.get('bias',False) and args.get('bias') is not None:problems.append('unexpected_bias')
    if op.get('output_metadata',{}).get('shape')!=d['output_shape']:problems.append('output_shape')
    return {'matched':not problems,'problems':problems,'ATen':op['aten']}

def main():
    from o4_i3_v2.assets import verify
    from o4_i3_v2.collect import code_index
    from o4_i3_v2.protocol import measured_values,distribution
    from o4_i3.taxonomy import classify
    from o4_i3.inventory import ownership,union
    out=ROOT/'analysis/summary';out.mkdir(parents=True,exist_ok=False)
    scope=read(ROOT/'SCOPE.json');selected={c['id']:c for c in read(ROOT/'SELECTED_CONFIGS.json')}
    verify(ROOT/'assets_S3/ASSETS.json')
    cfg_assets=read(ROOT/'assets_S3/ASSETS.json')['configurations']+read(S1/'assets_S1/ASSETS.json')['configurations']+read(S2/'assets_S2/ASSETS.json')['configurations']
    configs={e['configuration']['id']:e['configuration'] for e in cfg_assets}
    cells=[];modes=[];pairs=[];failures=[];quality=[];ownerships=[];sequence_records=[];component_index=collections.defaultdict(list)
    sources=[];pids=[];allow_records=0;clockdists=[];overlap_stats=collections.Counter();overlap_samples=[];duration_samples=collections.defaultdict(list);runtime_checks=[]
    all_iterations=[];streams=[];capture_sizes=0;sqlite_sizes=0
    safe={'event_kind','start_ns','end_ns','api_start_ns','api_end_ns','grid_xyz','block_xyz','dynamic_shared_memory_bytes','bytes','copy_kind','memory_kind','memset_value','event_id','api_event_id','correlation_id','stream_id','context_id','process_id'}
    indexes=[('S1_smoke',S1/'collections/smoke'),('S1_formal',S1/'collections/formal')]
    indexes += [('S2_'+p.name,p) for p in sorted((S2/'collections').iterdir()) if (p/'inventory.json').exists()]
    indexes += [('S3_'+p.name,p) for p in sorted((ROOT/'collections').iterdir()) if (p/'inventory.json').exists()]
    for name,folder in indexes:
        inv=read(folder/'inventory.json');p=inv['protocol'];is_s2=name.startswith('S3_')
        sources.append({'path':str(folder/'inventory.json'),'sha256':sha(folder/'inventory.json')})
        if inv['status']=='running':raise RuntimeError('Cannot summarize running collection '+name)
        for g in inv.get('mode_groups',[]):modes.append({'source':name,**g})
        for pair in inv.get('P_R_pairs',[]):pairs.append({'source':name,**pair})
        expected_index=read(folder/'CODE_SHA256.json')
        actual_index=code_index(folder/'source_snapshot')
        assert actual_index==expected_index
        for j in inv['jobs']:
            cell=folder/j['folder'];cid=j['config_id'];row={'source':name,'kind':p['kind'],'config_id':cid,'repeat':j['repeat'],'mode':j['mode'],'status':j['status'],'root':str(cell)}
            if j['status']!='completed':failures.append({**row,'error':j.get('error','not completed')})
            if (cell/'quality.json').exists():
                q=read(cell/'quality.json');row.update(worker_status=q['status'],correctness_passed=q.get('correctness',{}).get('passed'),
                    correctness_relative_l2=q.get('correctness',{}).get('relative_l2'),correctness_scaled_abs=q.get('correctness',{}).get('max_abs_over_scale'),
                    warmup=q.get('warmup',{}).get('iterations'),warmup_passed=q.get('warmup',{}).get('passed'),
                    worker_errors=q.get('errors',[]),R_instrumented_exact=q.get('instrumented_output_exact'),
                    measured=q.get('distribution',{}).get('n',0),median_us=q.get('distribution',{}).get('median_us'),tail_review=q.get('tail_review'))
                if (cell/'environment.json').exists():pids.append({'source':name,'cell':j['folder'],'pid':read(cell/'environment.json')['pid']})
                if (cell/'iterations.csv').exists():
                    it=list(csv.DictReader((cell/'iterations.csv').open()))
                    all_iterations.extend({**r,'source':name,'config_id':cid,'repeat':j['repeat'],'mode':j['mode']} for r in it)
                    if q['status']=='completed':
                        calculated=distribution(measured_values(it,p));assert calculated==q['distribution']
                        for r in it:
                            if r['stage']=='measure':duration_samples[(name,cid,j['mode'])].append(float(r['elapsed_us']))
            mon=read(cell/'monitor_summary.json') if (cell/'monitor_summary.json').exists() else None
            row['monitor_review']=j.get('monitor_review')
            if mon:row.update(foreign_work=mon['foreign_GPU_work_observed'],monitor_errors=mon['errors'])
            for label in ['collect','export']:
                f=cell/(label+'_command.json')
                if f.exists():
                    cmd=read(f);row[label+'_seconds']=(cmd['end_monotonic_ns']-cmd['start_monotonic_ns'])/1e9
            for filename in ['trace.nsys-rep','trace.sqlite']:
                f=cell/filename
                if f.exists():
                    row[filename+'_bytes']=f.stat().st_size
                    if is_s2:
                        if filename.endswith('rep'):capture_sizes+=f.stat().st_size
                        else:sqlite_sizes+=f.stat().st_size
            cells.append(row)
            audit=cell/'audit'
            if not (audit/'SUMMARY.json').exists():continue
            summary=read(audit/'SUMMARY.json');eng=read(audit/'engineering_acceptance.json');gates=read(audit/'v4_quality.json')['gates']
            quality.append({'source':name,'config_id':cid,'repeat':j['repeat'],'mode':j['mode'],**summary,
                'actual_backend_confirmed':eng['actual_backend'],'engineering_status':eng['status'],'engineering_blockers':eng['blockers'],
                'submitted_work_status':gates['submitted_work_coverage']['status'],'clock_status':gates['timestamp_consistency']['status']})
            sources.append({'path':str(cell/'trace.sqlite'),'sha256':sha(cell/'trace.sqlite')})
            events=list(csv.DictReader((audit/'events_diagnostic.csv').open()))
            streamset=sorted({(e['device'],e['context'],e['stream']) for e in events})
            streams.append({'source':name,'config_id':cid,'repeat':j['repeat'],'mode':j['mode'],'observed_streams':streamset})
            for f in (audit/'predictor').glob('*.jsonl'):
                for line in f.read_text().splitlines():assert set(json.loads(line))<=safe;allow_records+=1
            starts=[int(e['start_ns'])-int(e['api_start_ns']) for e in events];ends=[int(e['end_ns'])-int(e['api_start_ns']) for e in events]
            if starts:clockdists.append({'source':name,'config_id':cid,'repeat':j['repeat'],'mode':j['mode'],
                'start_minus_entry_ns_min':min(starts),'start_minus_entry_ns_median':statistics.median(starts),'start_minus_entry_ns_max':max(starts),
                'end_minus_entry_ns_min':min(ends),'end_minus_entry_ns_median':statistics.median(ends),'end_minus_entry_ns_max':max(ends)})
            active=[]
            for e in events:
                start=int(e['start_ns']);active=[x for x in active if int(x['end_ns'])>start]
                for a in active:
                    key=(name,cid,j['mode'],a['event_kind']+'-'+e['event_kind'],a['stream']==e['stream'],a['iteration_scope']==e['iteration_scope'])
                    overlap_stats[key]+=1
                    if len(overlap_samples)<25:overlap_samples.append({'source':name,'config_id':cid,'mode':j['mode'],'repeat':j['repeat'],
                        'overlap_ns':min(int(a['end_ns']),int(e['end_ns']))-start,'first':a,'second':e})
                active.append(e)
            if j['mode']=='R':
                refs=read(audit/'reference_operations.json');per=collections.defaultdict(list)
                for op in refs:per[(op['stage'],op['iteration'])].append(op)
                for op in refs:runtime_checks.append({'source':name,'config_id':cid,'repeat':j['repeat'],'stage':op['stage'],'iteration':op['iteration'],**check_runtime(configs[cid],op)})
                for (stage,i),ops in per.items():
                    es=[e for e in events if e['iteration_scope']==f'o4.iter/{stage}/{i:06d}']
                    own=ownership([op['event_indices'] for op in ops],len(es))
                    ownerships.append({'source':name,'config_id':cid,'repeat':j['repeat'],'stage':stage,'iteration':i,
                        'recorded_events':len(es),'ATen_operations':len(ops),'unassigned':len(own['unassigned']),'multiple_owners':len(own['multiple_owners']),
                        'noncontiguous':sum(not op['contiguous'] for op in ops),'zero_recorded_ops':sum(op['zero_recorded_GPU_work'] for op in ops),
                        'max_events_per_op':max((op['recorded_GPU_event_count'] for op in ops),default=0)})
                    sequence=[semantic(op) for op in ops]
                    sequence_records.append({'source':name,'config_id':cid,'repeat':j['repeat'],'stage':stage,'iteration':i,
                        'sequence_sha256':digest(sequence),'ATen_names':[op['aten'] for op in ops],
                        'all_arguments_present':all(bool(op.get('bound_arguments')) for op in ops),
                        'all_outputs_present':all(bool(op.get('output_metadata')) for op in ops),'operations':len(ops)})
                    if stage=='measure':
                        for op in ops:
                            key=digest(semantic(op));component_index[key].append({'source':name,'config_id':cid,'repeat':j['repeat'],
                                'iteration':i,'op_index':op['op_index'],'ATen':op['aten'],'events':op['recorded_GPU_event_count'],
                                'mapped':classify(op['aten'])['mapped'],'subtype':classify(op['aten'])['subtype']})
    metrics=[]
    for cid,c in sorted(configs.items()):
        chosen=[g for g in modes if g['config_id']==cid and g['source'].endswith('_formal')]
        by={g['mode']:g for g in chosen}
        refseq=[s for s in sequence_records if s['config_id']==cid and s['source'].endswith('_formal') and s['stage']=='measure']
        record={'config_id':cid,'family':c['family'],'anchor':c['anchor'],'split_B':c['split_B'],'leakage_group':c['leakage_group'],
            'GPU_attempted':True,'GI_us':by.get('GI',{}).get('median_of_process_medians_us'),
            'GI_spread_fraction':by.get('GI',{}).get('spread_fraction'),'GI_status':by.get('GI',{}).get('stability_status','not_collected'),
            'P_event_us':by.get('P',{}).get('median_of_process_medians_us'),'P_status':by.get('P',{}).get('stability_status','not_collected'),
            'R_event_us':by.get('R',{}).get('median_of_process_medians_us'),'R_status':by.get('R',{}).get('stability_status','not_collected'),
            'formal_R_measured_iterations':len(refseq),'runtime_sequence_variants':len({s['sequence_sha256'] for s in refseq}),
            'ATen_sequences':sorted({tuple(s['ATen_names']) for s in refseq}),
            'strict_reference_verified':False,'formal_training_eligible':False}
        metrics.append(record)
    ctx_samples=collections.defaultdict(list)
    for source_name,folder in indexes:
        inv=read(folder/'inventory.json')
        if inv['protocol']['kind']!='formal':continue
        for item in inv.get('P_R_pairs',[]):
            if item.get('error'):continue
            cid=item['config_id'];rep=item['repeat']
            alignment=read(folder/f'{cid}_r{rep}_pair/P_R_ALIGNMENT.json')
            exact={a['scope'] for a in alignment if a['ordered_signature_exact'] and '/measure/' in a['scope']}
            per=collections.defaultdict(list)
            for e in csv.DictReader((folder/f'{cid}_r{rep}_P/audit/events_diagnostic.csv').open()):per[e['iteration_scope']].append(e)
            refs=read(folder/f'{cid}_r{rep}_R/audit/reference_operations.json')
            for op in refs:
                scope=f'o4.iter/{op["stage"]}/{op["iteration"]:06d}'
                if scope not in exact or not op['event_indices']:continue
                events=[per[scope][i] for i in op['event_indices']]
                span=(max(int(e['end_ns']) for e in events)-min(int(e['start_ns']) for e in events))/1000
                ctx_samples[(cid,op['op_index'],rep)].append(span)
    ctx_rows=[];ctx_by_config={}
    for record in metrics:
        cid=record['config_id'];med=[statistics.median(ctx_samples[(cid,0,r)]) for r in [1,2,3] if ctx_samples[(cid,0,r)]]
        complete=len(med)==3 and all(len(ctx_samples[(cid,0,r)])==40 for r in [1,2,3])
        # S1/S3 isolated work currently exposes one ATen operation per forward.
        one_op=all(s['operations']==1 for s in sequence_records if s['config_id']==cid and s['source'].endswith('_formal'))
        spread=(max(med)-min(med))/statistics.median(med) if complete else None
        row={'config_id':cid,'target_kind':'contextual_trace_span_us','op_index':0,'process_medians_us':med,
            'median_of_process_medians_us':statistics.median(med) if med else None,'spread_fraction':spread,
            'complete_40x3':complete,'one_operation_per_forward':one_op,
            'stability_status':'stable' if complete and one_op and spread<=.05 else 'review_required',
            'strict_reference_verified':False}
        ctx_rows.append(row);ctx_by_config[cid]=row
    usage=[]
    for record in metrics:
        cid=record['config_id'];trace=[q for q in quality if q['config_id']==cid and q['source'].endswith('_formal')]
        unmatch=sum(q['unmatched_work_APIs'] for q in trace)
        pairinfo=[p for p in pairs if p['config_id']==cid and p['source'].endswith('_formal')]
        match=sum(p.get('measured_exact',0) for p in pairinfo)
        complete=bool(trace) and len(pairinfo)==3 and match==120
        block=any(q['engineering_blockers'] for q in trace)
        own=[o for o in ownerships if o['config_id']==cid and o['source'].endswith('_formal')]
        ownership_ok=bool(own) and not any(o['unassigned'] or o['multiple_owners'] or o['noncontiguous'] or o['zero_recorded_ops'] for o in own)
        config_cells=[v for v in cells if v['config_id']==cid and v['source'].endswith('_formal')]
        monitor_ok=bool(config_cells) and all(v.get('monitor_review') is False for v in config_cells)
        parameter_rows=[v for v in runtime_checks if v['config_id']==cid and v['source'].endswith('_formal')]
        runtime_ok=bool(parameter_rows) and all(v['matched'] for v in parameter_rows)
        complete=complete and len(trace)==6 and len(config_cells)==9 and all(v['status']=='completed' and v.get('correctness_passed') and v.get('warmup_passed') for v in config_cells)
        block=block or not monitor_ok or not runtime_ok
        usage.append({'config_id':cid,'family':record['family'],'anchor':record['anchor'],
            'GI_development_use':'development_usable' if record['GI_status']=='stable' else 'limited_use',
            'monitor_all_formal_cells_ok':monitor_ok,'runtime_all_formal_rows_matched':runtime_ok,
            'recorded_boundary_class_use':'development_usable' if complete and ownership_ok and not block and not unmatch else 'limited_use',
            'P_CTX_development_use':'development_usable' if complete and ownership_ok and not block and not unmatch and ctx_by_config[cid]['stability_status']=='stable' else 'limited_use',
            'GI_stability_status':record['GI_status'],'P_stability_status':record['P_status'],'R_stability_status':record['R_status'],
            'P_CTX_stability_status':ctx_by_config[cid]['stability_status'],'P_CTX_spread_fraction':ctx_by_config[cid]['spread_fraction'],
            'formal_PR_measured_exact':match,'expected_formal_PR_measured':120,'unmatched_work_API_count':unmatch,
            'submitted_work_coverage':'unverified' if unmatch else 'recorded_reverse_audit_passed_not_strict',
            'clock_start_before_API_entry':sum(q['clock_start_before_API_entry'] for q in trace),
            'clock_end_before_API_entry':sum(q['clock_end_before_API_entry'] for q in trace),
            'max_events_per_op':max((o['max_events_per_op'] for o in own),default=0),
            'known_limitations':'NVTX warning; CPU-GPU clock mapping unverified; no strict certification'+('; unmatched API byte-count unknown, do not infer 0B' if unmatch else ''),
            'strict_reference_verified':False,'formal_training_eligible':False})
    collisions=[];unmapped=[]
    for key,uses in component_index.items():
        unique=sorted({u['config_id'] for u in uses})
        if len(unique)>1:collisions.append({'component_semantic_sha256':key,'configs':unique,'ATen':uses[0]['ATen'],
            'split_values':sorted({configs[c]['split_B'] for c in unique}),
            'group_ids':sorted({configs[c]['leakage_group'] for c in unique}),'reason':'same full ATen argument/output metadata; ignores device, bytes, repeat and op position'})
        if not uses[0]['mapped']:unmapped.append({'semantic_sha256':key,'ATen':uses[0]['ATen'],'configs':unique})
    s2cells=[c for c in cells if c['source'].startswith('S3_')];s2quality=[q for q in quality if q['source'].startswith('S3_')]
    s2formal=[c for c in s2cells if c['kind']=='formal'];s2smoke=[c for c in s2cells if c['kind']=='smoke']
    s2modes=[m for m in modes if m['source'].startswith('S3_') and m['source'].endswith('_formal')]
    s2pairs=[p for p in pairs if p['source'].startswith('S3_') and p['source'].endswith('_formal')]
    aggregate={'S3_configs_preselected':36,'S3_config_formal_collected':len({c['config_id'] for c in s2formal if c['status']=='completed'}),
        'S3_smoke_cells_planned':108,'S3_smoke_cells_in_inventory':len(s2smoke),'S3_smoke_cells_completed':sum(c['status']=='completed' for c in s2smoke),
        'S3_formal_cells_planned':324,'S3_formal_cells_in_inventory':len(s2formal),'S3_formal_cells_completed':sum(c['status']=='completed' for c in s2formal),
        'S3_formal_measured_saved':sum(c.get('measured',0) for c in s2formal),'S3_all_measured_saved':sum(c.get('measured',0) for c in s2cells),
        'S3_formal_stable_mode_groups':sum(m['stability_status']=='stable' for m in s2modes),
        'S3_formal_review_mode_groups':sum(m['stability_status']=='review_required' for m in s2modes),
        'S3_formal_PR_pairs':len(s2pairs),'S3_formal_PR_measured_exact':sum(p.get('measured_exact',0) for p in s2pairs),
        'S3_raw_traces_audited':len(s2quality),'S3_NVTX_warning_traces':sum(bool(q['warnings']) for q in s2quality),
        'S3_unmatched_work_APIs':sum(q['unmatched_work_APIs'] for q in s2quality),
        'S3_clock_start_before_API_entry':sum(q['clock_start_before_API_entry'] for q in s2quality),
        'S3_clock_end_before_API_entry':sum(q['clock_end_before_API_entry'] for q in s2quality),
        'S3_overlap_pairs':sum(q['overlap_pairs'] for q in s2quality),
        'S3_capture_bytes':capture_sizes,'S3_sqlite_bytes':sqlite_sizes,
        'combined_core_configurations':len(metrics),'combined_formal_measured_saved':sum(c.get('measured',0) for c in cells if c['kind']=='formal'),
        'combined_formal_cells_completed':sum(c['status']=='completed' for c in cells if c['kind']=='formal'),
        'combined_formal_mode_stable':sum(g['stability_status']=='stable' for g in modes if g['source'].endswith('_formal')),
        'combined_formal_mode_review':sum(g['stability_status']=='review_required' for g in modes if g['source'].endswith('_formal')),
        'combined_unassigned_events':sum(o['unassigned'] for o in ownerships),'combined_multiple_owners':sum(o['multiple_owners'] for o in ownerships),
        'combined_noncontiguous_ops':sum(o['noncontiguous'] for o in ownerships),'combined_zero_recorded_ops':sum(o['zero_recorded_ops'] for o in ownerships),
        'combined_runtime_signature_collisions':len(collisions),'combined_predictor_event_records_checked':allow_records,
        'runtime_parameter_rows_checked':len(runtime_checks),'runtime_parameter_rows_matched':sum(r['matched'] for r in runtime_checks),
        'current_train_config_groups':len({c['leakage_group'] for c in configs.values() if c['split_B']=='train'}),
        'current_all_config_groups':len({c['leakage_group'] for c in configs.values()}),
        'validation_or_test_config_groups':len({c['leakage_group'] for c in configs.values() if c['split_B']!='train'}),
        'core_configs_P_CTX_development_usable':sum(r['P_CTX_development_use']=='development_usable' for r in usage),
        'core_configs_P_CTX_limited_use':sum(r['P_CTX_development_use']=='limited_use' for r in usage),
        'core_configs_P_CTX_stable':sum(r['stability_status']=='stable' for r in ctx_rows),
        'core_configs_P_CTX_review':sum(r['stability_status']=='review_required' for r in ctx_rows),
        'strict_reference_verified':False,'formal_training_eligible':False,'models_trained':0,'GPU_other_nodes_executed':False}
    for n,rs in [('CELLS',cells),('MODE_STABILITY',modes),('P_R_PAIRS',pairs),('FAILURES',failures),('TRACE_QUALITY',quality),
                 ('OWNERSHIP',ownerships),('R_SEQUENCE_SIGNATURES',sequence_records),('R_PARAMETERS_CHECKED',runtime_checks),('CORE72_METRICS',metrics),('P_CTX_STABILITY',ctx_rows),('DEVELOPMENT_USE',usage),('CLOCK_DISTRIBUTIONS',clockdists),('ITERATIONS',all_iterations)]:
        csvwrite(out/(n+'.csv'),rs)
    write(out/'SUMMARY.json',aggregate);write(out/'SOURCE_SHA256.json',sources)
    write(out/'OVERLAP.json',{'groups':[{'source':k[0],'config_id':k[1],'mode':k[2],'event_types':k[3],'same_stream':k[4],'same_iteration':k[5],'pairs':v} for k,v in overlap_stats.items()],'examples':overlap_samples})
    write(out/'STREAMS.json',streams);write(out/'R_COMPONENT_COLLISIONS.json',collisions);write(out/'UNMAPPED_ATEN.json',unmapped)
    write(out/'PID_AUDIT.json',{'workers':len(pids),'distinct_PIDs':len({p['pid'] for p in pids}),'rows':pids,
        'note':'OS may reuse PID across long sessions; each worker command is a new subprocess; use start/end command times with PID'})
    write(out/'SPLIT_FEASIBILITY.json',{'status':'exploratory_configuration_split_runtime_audited_not_confirmatory','core_configs':len(metrics),'families':len({c['family'] for c in configs.values()}),
        'base_anchors_per_family':5,'expected_train_validation_test_groups':[36,12,12],'all_train':all(c['split_B']=='train' for c in configs.values()),
        'runtime_ATen_component_collisions':len(collisions),'collisions_cross_split':sum(len(c['split_values'])>1 for c in collisions),
        'train_validation_test_frozen':True,'OOF_ready':False,'reason':'S3 01/02 precollection manifest; one validation and one exploratory test anchor per family only; no model scores; OOF folds not yet constructed; context not collected'})
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()
