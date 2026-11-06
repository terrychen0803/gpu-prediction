"""All measured P/R records audit, appended to a new output directory only."""
import argparse, collections, csv, hashlib, json
from pathlib import Path
from .taxonomy import classify, registry

def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p, data): Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvwrite(p, rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def union(intervals):
    if not intervals:return 0
    a,b=sorted(intervals)[0];s=0
    for u,v in sorted(intervals)[1:]:
        if u>b:s+=b-a;a,b=u,v
        else:b=max(b,v)
    return s+b-a

def ownership(indices, n):
    counts=collections.Counter(i for group in indices for i in group)
    bad=[i for i in counts if type(i) is not int or i<0 or i>=n]
    if bad:raise ValueError('Out-of-range event ownership')
    return {'unassigned':[i for i in range(n) if counts[i]==0],
            'multiple_owners':[i for i in range(n) if counts[i]>1]}

def run(index, out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    sources=[{'path':str(index),'sha256':sha(index)}];families=[];owners=[];lengths=[];signatures={};deltas=[];attrs=[]
    known=set();total_ops=total_events=0
    for r in read(index):
        root=Path(r['root']);device,work=r['device'],r['workload']
        for rep in (1,2,3):
            p=root/f'{work}_P_r{rep}'/'audit';q=root/f'{work}_R_r{rep}'/'audit'
            pf=p/'events_diagnostic.csv';rf=q/'reference_operations.json';pairf=root/f'{work}_r{rep}_pair'/'P_R_ALIGNMENT.json'
            for f in (pf,rf,pairf):sources.append({'path':str(f),'sha256':sha(f)})
            matches={x['scope'] for x in read(pairf) if x['ordered_signature_exact'] and '/measure/' in x['scope']}
            ev=collections.defaultdict(list)
            with pf.open() as f:
                for e in csv.DictReader(f):
                    if e['iteration_scope'] in matches:ev[e['iteration_scope']].append(e)
            refs=collections.defaultdict(list)
            for op in read(rf):
                scope=f"o4.iter/{op['stage']}/{op['iteration']:06d}"
                if scope in matches:refs[scope].append(op)
            fullbusy=sum(union([(int(e['start_ns']),int(e['end_ns'])) for e in es]) for es in ev.values())
            n_events=sum(map(len,ev.values()));groups=collections.defaultdict(lambda:{'calls':0,'events':0,'spans':0,'intervals':collections.defaultdict(list)})
            hist=collections.Counter();unassigned=multiple=noncontig=zeros=0;missing_by_name={}
            for scope in sorted(matches):
                es=ev[scope];ops=refs[scope]
                checked=ownership([op['event_indices'] for op in ops],len(es));unassigned+=len(checked['unassigned']);multiple+=len(checked['multiple_owners'])
                for op in ops:
                    c=classify(op['aten']);sub=c['subtype'];known.add(op['aten']);ids=op['event_indices'];hist[len(ids)]+=1
                    zeros+=not ids;noncontig+=bool(ids and ids!=list(range(ids[0],ids[-1]+1)))
                    ints=[(int(es[i]['start_ns']),int(es[i]['end_ns'])) for i in ids]
                    g=groups[sub];g['calls']+=1;g['events']+=len(ids);g['spans']+=(max(b for a,b in ints)-min(a for a,b in ints)) if ints else 0;g['intervals'][scope].extend(ints)
                    missing_by_name[op['aten']]=c
                    key=json.dumps([op['aten'],op.get('input_shapes'),op.get('input_dtypes')],sort_keys=True)
                    sign=hashlib.sha256(key.encode()).hexdigest()
                    signatures.setdefault((device,work,sign),{'device':device,'workload':work,'partial_signature_sha256':sign,'aten':op['aten'],'input_shapes':json.dumps(op.get('input_shapes')),'input_dtypes':json.dumps(op.get('input_dtypes')),'exact_subtype':sub,'configuration_identity_status':'partial_only_scalar_and_stride_missing','reference_status':'development_unverified'})
                for i in checked['unassigned']:
                    e=es[i];g=groups['unassigned_reference_work'];g['events']+=1;g['intervals'][scope].append((int(e['start_ns']),int(e['end_ns'])))
            for name,c in missing_by_name.items():
                attrs.append({'device':device,'workload':work,'repeat':rep,'aten':name,'subtype':c['subtype'],'exact_overload_mapped':c['mapped'],
                    'missing_tensor_attributes':'stride;device;layout;alias_identity',
                    'semantic_attributes_not_in_saved_schema':';'.join(c['required_semantic_attributes']),
                    'configuration_reconstruction':'pending_adapter_or_controlled_manifest_evidence',
                    'complete_configuration_verified':False,'label_status':'development_unverified'})
            owners.append({'device':device,'workload':work,'repeat':rep,'matched_measured_iterations':len(matches),
                'recorded_P_events':n_events,'reference_ops':sum(hist.values()),'zero_recorded_work_ops':zeros,
                'unassigned_events':unassigned,'multiple_owner_events':multiple,'noncontiguous_ops':noncontig,
                'recorded_event_assignment_complete':not (unassigned or multiple or noncontig),
                'strict_reference_verified':False,'formal_training_eligible':False})
            for k,n in sorted(hist.items()):lengths.append({'device':device,'workload':work,'repeat':rep,'events_per_op':k,'op_count':n,'strict_reference_verified':False})
            for sub,g in groups.items():
                families.append({'device':device,'workload':work,'repeat':rep,'subtype':sub,'matched_iterations':len(matches),
                    'calls_per_iteration':g['calls']/len(matches),'event_count':g['events'],
                    'event_coverage_pct':100*g['events']/n_events if n_events else None,
                    'busy_union_coverage_pct':100*sum(union(v) for v in g['intervals'].values())/fullbusy if fullbusy else None,
                    'mean_CTX_span_sum_us':g['spans']/1000/len(matches),'role':'evaluator_only_exact_interface_name',
                    'reference_status':'development_unverified'})
            total_ops+=sum(hist.values());total_events+=n_events
            print(device,work,rep,'ops',sum(hist.values()),'unassigned',unassigned,flush=True)
    for name,rows in [('COVERAGE_V2',families),('OWNERSHIP',owners),('LENGTH_DISTRIBUTION',lengths),('PARTIAL_SIGNATURES',list(signatures.values())),('ATTRIBUTE_GAPS',attrs)]:csvwrite(out/(name+'.csv'),rows)
    old=Path(index).parent/'analysis/FAMILY_COVERAGE.csv'
    if old.exists():
        for row in csv.DictReader(old.open()):
            if row['workload']=='UN-A' and row['family']=='linear_or_gemm':deltas.append({'device':row['device'],'workload':row['workload'],'repeat':row['repeat'],'old_family':row['family'],'old_busy_union_coverage_pct':row['busy_union_coverage_pct'],'new_subtype':'resize_bilinear','reason':'exact overload corrects substring linear collision; old outputs retained'})
    csvwrite(out/'RESIZE_CORRECTION.csv',deltas)
    write(out/'taxonomy_v1.json',registry());write(out/'SOURCE_SHA256.json',sources)
    write(out/'SUMMARY.json',{'cases':len(owners),'matched_iterations':sum(x['matched_measured_iterations'] for x in owners),'reference_ops':total_ops,'recorded_P_events':total_events,
        'observed_ATen_names':len(known),'unmapped_names':sorted(n for n in known if not classify(n)['mapped']),
        'unassigned_events':sum(x['unassigned_events'] for x in owners),'multiple_owner_events':sum(x['multiple_owner_events'] for x in owners),
        'noncontiguous_ops':sum(x['noncontiguous_ops'] for x in owners),'partial_signatures':len(signatures),
        'formal_training_eligible':False,'strict_verified':0,'all_previous_quality_gates_preserved':True,
        'time_coverage_status':'recorded_only_not_submitted_coverage','semantic_coverage_status':'ATen_interface_conditional_not_configuration_verified',
        'family_unions_may_overlap':True})

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--index',required=True);a.add_argument('--out',required=True);q=a.parse_args();run(q.index,q.out)
