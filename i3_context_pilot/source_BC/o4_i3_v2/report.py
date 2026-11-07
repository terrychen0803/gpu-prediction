"""Read-only evidence extraction and optional independent raw replay, new output only."""
import argparse,collections,csv,json,statistics
from pathlib import Path
from o4_research.common import read,write,csvwrite,sha
from o4_i3.inventory import ownership,union
from o4_i3.taxonomy import classify
from .protocol import distribution

def compare(before,after):
    a={str(f.relative_to(before)):sha(f) for f in before.rglob('*') if f.is_file()}
    b={str(f.relative_to(after)):sha(f) for f in after.rglob('*') if f.is_file()}
    if a!=b:raise ValueError('Raw replay artifacts differ')
    return len(a)

def analyze(run,out,replay=False):
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    cells=[];traces=[];owners=[];params=[];ctx=[];examples=[];replay_rows=[];sources=[];pairs=[];groups=[]
    for collection in sorted((run/'collections').iterdir()):
        if not (collection/'inventory.json').exists():continue
        inv=read(collection/'inventory.json');p=inv['protocol'];sources.append({'path':str(collection/'inventory.json'),'sha256':sha(collection/'inventory.json')})
        for g in inv.get('mode_groups',[]):groups.append({'collection':collection.name,**g})
        for item in inv.get('P_R_pairs',[]):pairs.append({'collection':collection.name,**item})
        for j in inv['jobs']:
            folder=collection/j['folder'];record={'collection':collection.name,'config_id':j['config_id'],'repeat':j['repeat'],'mode':j['mode'],'status':j['status']}
            if not (folder/'quality.json').exists():cells.append(record);continue
            q=read(folder/'quality.json');record.update(correctness_passed=q.get('correctness',{}).get('passed'),
                relative_l2=q.get('correctness',{}).get('relative_l2'),max_abs_over_scale=q.get('correctness',{}).get('max_abs_over_scale'),
                warmup_iterations=q.get('warmup',{}).get('iterations'),warmup_passed=q.get('warmup',{}).get('passed'),
                warmup_R_active=q.get('warmup',{}).get('R_active'),instrumented_exact=q.get('instrumented_output_exact'),
                measured_n=q.get('distribution',{}).get('n'),median_us=q.get('distribution',{}).get('median_us'),
                tail_fraction=q.get('distribution',{}).get('tail_fraction'),tail_review=q.get('tail_review'),monitor_review=j.get('monitor_review'))
            for stem in ['collect','export']:
                path=folder/(stem+'_command.json')
                if path.exists():
                    cmd=read(path);record[stem+'_seconds']=(cmd['end_monotonic_ns']-cmd['start_monotonic_ns'])/1e9
            record['capture_bytes']=(folder/'trace.nsys-rep').stat().st_size if (folder/'trace.nsys-rep').exists() else 0
            record['sqlite_bytes']=(folder/'trace.sqlite').stat().st_size if (folder/'trace.sqlite').exists() else 0
            cells.append(record)
            if not (folder/'audit/SUMMARY.json').exists():continue
            audit=folder/'audit';summary=read(audit/'SUMMARY.json');engineering=read(audit/'engineering_acceptance.json')
            raw=folder/'trace.sqlite';before=sha(raw);sources.append({'path':str(raw),'sha256':before})
            tr={**{k:record[k] for k in ['collection','config_id','repeat','mode']},**summary,
                'engineering_status':engineering['status'],'actual_backend':engineering['actual_backend'],
                'strict_reference_verified':False,'formal_training_eligible':False}
            traces.append(tr)
            if replay:
                from o4_research.audit_v5 import audit as raw_audit
                dest=out/'replay'/collection.name/j['folder'];dest.parent.mkdir(parents=True,exist_ok=True)
                raw_audit(folder,dest);n=compare(audit,dest)
                if sha(raw)!=before:raise ValueError('Raw trace changed during replay')
                replay_rows.append({'cell':str(folder),'artifacts_identical':n,'raw_sha256':before})
            ev=list(csv.DictReader((audit/'events_diagnostic.csv').open()))
            for e in ev:
                if int(e['start_ns'])<int(e['api_start_ns']) or int(e['end_ns'])<int(e['api_start_ns']):
                    if sum(x.get('kind')=='timestamp' for x in examples)<8:examples.append({'kind':'timestamp','cell':str(folder),'event':e})
            # Keep concrete overlap examples, without repairing intervals or inferring PDL.
            active=[]
            for e in ev:
                a=int(e['start_ns']);b=int(e['end_ns']);active=[x for x in active if int(x['end_ns'])>a]
                for prev in active:
                    if sum(x.get('kind')=='overlap' for x in examples)<8:
                        examples.append({'kind':'overlap','cell':str(folder),'overlap_ns':min(int(prev['end_ns']),b)-a,
                            'same_stream':prev['stream']==e['stream'],'first':prev,'second':e})
                active.append(e)
            if j['mode']=='R':
                refs=read(audit/'reference_operations.json');by=collections.defaultdict(list)
                for op in refs:by[(op['stage'],op['iteration'])].append(op)
                for (stage,i),ops in by.items():
                    es=[e for e in ev if e['iteration_scope']==f'o4.iter/{stage}/{i:06d}']
                    own=ownership([op['event_indices'] for op in ops],len(es))
                    owners.append({'collection':collection.name,'config_id':j['config_id'],'repeat':j['repeat'],'stage':stage,'iteration':i,
                        'recorded_events':len(es),'ATen_ops':len(ops),'unassigned':len(own['unassigned']),'multiple_owners':len(own['multiple_owners']),
                        'noncontiguous':sum(not op['contiguous'] for op in ops),'zero_recorded_ops':sum(op['zero_recorded_GPU_work'] for op in ops)})
                for op in refs:
                    params.append({'collection':collection.name,'config_id':j['config_id'],'repeat':j['repeat'],'stage':op['stage'],'iteration':op['iteration'],
                        'aten':op['aten'],'mapped_subtype':classify(op['aten'])['subtype'],
                        'complete_bound_arguments_present':bool(op.get('bound_arguments')),'bound_arguments':op.get('bound_arguments'),
                        'output_metadata':op.get('output_metadata'),'recorded_events':op['recorded_GPU_event_count'],'runtime_reference_status':'development_warning_accepted_not_strict'})
        for item in inv.get('P_R_pairs',[]):
            if item.get('error'):continue
            cid=item['config_id'];rep=item['repeat'];dest=collection/f'{cid}_r{rep}_pair'
            pair_jobs=[job for job in inv['jobs'] if job['config_id']==cid and job['repeat']==rep and job['mode'] in ('P','R')]
            monitor_review=any(job.get('monitor_review',False) for job in pair_jobs)
            pf=collection/f'{cid}_r{rep}_P';rf=collection/f'{cid}_r{rep}_R'
            align=read(dest/'P_R_ALIGNMENT.json');refs=read(rf/'audit/reference_operations.json')
            pe=list(csv.DictReader((pf/'audit/events_diagnostic.csv').open()))
            ps=read(pf/'audit/SUMMARY.json');rs=read(rf/'audit/SUMMARY.json')
            if replay:
                from o4_research.audit import pair
                new=out/'replay'/collection.name/f'{cid}_r{rep}_pair'
                pair(out/'replay'/collection.name/pf.name,out/'replay'/collection.name/rf.name,new)
                replay_rows.append({'pair':str(dest),'artifacts_identical':compare(dest,new)})
            exact={a['scope'] for a in align if a['ordered_signature_exact']}
            for op in refs:
                scope=f'o4.iter/{op["stage"]}/{op["iteration"]:06d}'
                if scope not in exact or not op['event_indices']:continue
                es=[e for e in pe if e['iteration_scope']==scope];ints=[(int(es[i]['start_ns']),int(es[i]['end_ns'])) for i in op['event_indices']]
                usable=ps['unmatched_work_APIs']==rs['unmatched_work_APIs']==0 and not monitor_review
                ctx.append({'collection':collection.name,'config_id':cid,'repeat':rep,'stage':op['stage'],'iteration':op['iteration'],'op_index':op['op_index'],
                    'aten':op['aten'],'P_CTX_span_us':(max(b for a,b in ints)-min(a for a,b in ints))/1000,
                    'P_duration_sum_us':sum(b-a for a,b in ints)/1000,'P_interval_union_us':union(ints)/1000,
                    'event_count':len(ints),'development_use':'development_usable' if usable else 'limited_use',
                    'reason':'same-device exact P/R, full bound arguments, no unmatched submitted work; warnings preserved' if usable else 'submitted work or concurrent activity needs review',
                    'strict_reference_verified':False})
    for name,rows in [('CELLS',cells),('TRACE_QUALITY',traces),('OWNERSHIP',owners),('R_PARAMETERS',params),('P_CTX_LABELS',ctx),('MODE_STABILITY',groups),('P_R_PAIRS',pairs)]:
        if rows:csvwrite(out/(name+'.csv'),rows)
    write(out/'ANOMALY_EXAMPLES.json',examples);write(out/'SOURCE_SHA256.json',sources)
    write(out/'REPLAY.json',{'traces':sum('cell' in r for r in replay_rows),'pairs':sum('pair' in r for r in replay_rows),
        'artifacts_identical':sum(r['artifacts_identical'] for r in replay_rows),'checks':replay_rows})
    result={'attempted_cells':len(cells),'completed_cells':sum(r['status']=='completed' for r in cells),
       'measured_iterations':sum(r.get('measured_n') or 0 for r in cells),'raw_traces':len(traces),
       'P_R_pairs':len(pairs),'P_R_exact_measured':sum(r.get('measured_exact',0) for r in pairs),
       'R_argument_rows':len(params),'R_unknown_ATen':sorted({r['aten'] for r in params if r['mapped_subtype']=='unmapped'}),
       'unassigned_recorded_events':sum(r['unassigned'] for r in owners),'multiple_owner_events':sum(r['multiple_owners'] for r in owners),
       'noncontiguous_ops':sum(r['noncontiguous'] for r in owners),'zero_recorded_ops':sum(r['zero_recorded_ops'] for r in owners),
       'unmatched_work_APIs':sum(r['unmatched_work_APIs'] for r in traces),
       'clock_start_before_API_entry':sum(r['clock_start_before_API_entry'] for r in traces),
       'clock_end_before_API_entry':sum(r['clock_end_before_API_entry'] for r in traces),
       'overlap_pairs':sum(r['overlap_pairs'] for r in traces),'NVTX_warning_traces':sum(bool(r['warnings']) for r in traces),
       'formal_mode_groups_stable':sum(r['stability_status']=='stable' for r in groups),
       'formal_mode_groups_review':sum(r['stability_status']=='review_required' for r in groups),
       'strict_reference_verified':False,'formal_training_eligible':False,'models_trained':0}
    write(out/'SUMMARY.json',result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--replay',action='store_true');a=p.parse_args()
    print(json.dumps(analyze(a.run,a.out,a.replay),indent=2))
