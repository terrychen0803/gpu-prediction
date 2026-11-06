"""Evaluator-only operation taxonomy and P-span coverage, not a predictor."""
import argparse,collections,csv,statistics
from pathlib import Path
from .common import *
from .audit import ITER
from .quality_v4 import union_ns
def family(name):
 s=name.lower()
 for needles,label in [(['convolution','conv2d'],'conv'),(['batch_norm'],'batchnorm_eval'),(['bmm'],'bmm'),(['linear','addmm','aten.mm.'],'linear_or_gemm'),(['hardswish','hardsigmoid'],'hard_activation'),(['relu'],'relu'),(['gelu'],'gelu'),(['silu'],'silu'),(['layer_norm','rms_norm'],'normalization'),(['softmax'],'softmax'),(['pool'],'pooling'),(['upsample','interpolate'],'resize'),(['cat.','stack.'],'concat'),(['copy','clone','_to_copy','contiguous'],'layout_or_copy'),(['view','reshape','transpose','permute','flatten','detach','as_strided','slice','select','split','unbind','expand'],'view_or_index'),(['sum','mean','amax','amin','rsqrt'],'reduction_or_rsqrt'),(['add','sub','mul','div','pow','neg','sigmoid','cos','sin'],'elementwise'),(['embedding','gather','index'],'embedding_or_gather')]:
  if any(x in s for x in needles):return label
 return 'other_uncovered'
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);overall=[];families=[];operators=[]
 for root in map(Path,a.runs):
  inv=read(root/'inventory.json')
  for name in dict.fromkeys(j['workload'] for j in inv['jobs']):
   jobs={j['mode']:j for j in inv['jobs'] if j['workload']==name and j['status']=='completed'}
   if not all(x in jobs for x in ['GI','P','R']):continue
   pf=root/jobs['P']['folder']/'audit';rf=root/jobs['R']['folder']/'audit';refs=read(rf/'reference_operations.json');events=list(csv.DictReader((pf/'events_diagnostic.csv').open()));alignment=read(root/(name+'_pair')/'P_R_ALIGNMENT.json');matches={x['scope'] for x in alignment if x['ordered_signature_exact']};local=[];byiter={}
   for scope in sorted(matches):
    if ITER.fullmatch(scope)[1]!='measure':continue
    ev=[e for e in events if e['iteration_scope']==scope];byiter[scope]=ev
    for ref in [r for r in refs if r['stage']=='measure' and r['iteration']==int(ITER.fullmatch(scope)[2])]:
     selected=[ev[i] for i in ref['event_indices']];intervals=[(int(e['start_ns']),int(e['end_ns'])) for e in selected];span=max(b for a,b in intervals)-min(a for a,b in intervals) if intervals else None
     local.append({'workload':name,'scope':scope,'op_index':ref['op_index'],'aten':ref['aten'],'family':family(ref['aten']),'P_event_count':len(selected),'P_contextual_span_ns':span,'P_busy_union_ns':union_ns(intervals) if intervals else 0,'P_intervals':intervals,'input_shapes':ref.get('input_shapes'),'label_status':'warning_accepted_conditional','strict_verified':0})
   operators.extend(local);n=len(byiter);den=sum(len(e) for e in byiter.values());busy=sum(union_ns([(int(e['start_ns']),int(e['end_ns'])) for e in ev]) for ev in byiter.values());gir=root/jobs['GI']['folder'];g=[float(x['elapsed_us']) for x in csv.DictReader((gir/'iterations.csv').open()) if x['stage']=='measure'];rsummary=read(rf/'SUMMARY.json');psummary=read(pf/'SUMMARY.json')
   overall.append({'workload':name,'GPU_smoke':'completed','GI_measured_samples':len(g),'GI_median_us_development_only':statistics.median(g),'repeat_stability':'not_evaluated_1process_3samples','P_R_measured_signature_matches':n,'reference_ops_per_matched_iteration':len(local)/n if n else None,'zero_recorded_work_ops_per_iteration':sum(x['P_event_count']==0 for x in local)/n if n else None,'max_events_per_op':max((x['P_event_count'] for x in local),default=None),'ops_over_L4_per_iteration':sum(x['P_event_count']>4 for x in local)/n if n else None,'P_measured_events':den,'P_preparation_GPU_records_all4windows':psummary['preparation_GPU_records'],'R_preparation_GPU_records_all4windows':rsummary['preparation_GPU_records'],'P_unmatched_APIs_all4windows':psummary['unmatched_work_APIs'],'R_unmatched_APIs_all4windows':rsummary['unmatched_work_APIs'],'P_NVTX_warning':bool(psummary['warnings']),'R_NVTX_warning':bool(rsummary['warnings']),'strict_verified':0})
   for label in sorted({x['family'] for x in local}):
    chosen=[x for x in local if x['family']==label];by=collections.defaultdict(list)
    for x in chosen:by[x['scope']].extend(x['P_intervals'])
    families.append({'workload':name,'family':label,'operator_calls_per_iteration':len(chosen)/n if n else None,'P_event_coverage_pct':100*sum(x['P_event_count'] for x in chosen)/den if den else None,'P_busy_union_coverage_pct':100*sum(union_ns(v) for v in by.values())/busy if busy else None,'P_contextual_span_sum_us':sum(x['P_contextual_span_ns'] or 0 for x in chosen)/1000,'reference_status':'conditional warning-accepted; not predictor labels verified'})
 csvwrite(out/'A_SUPPORT_AND_SMOKE.csv',overall);csvwrite(out/'FAMILY_COVERAGE.csv',families);write(out/'evaluator_only_operation_inventory.json',operators);write(out/'SUMMARY.json',{'workloads_completed':len(overall),'workloads':[x['workload'] for x in overall],'strict_verified':0,'taxonomy_is_diagnostic_heuristic_not_a_classifier':True,'coverage_denominator':'matched measured P recorded events/busy union; excludes explicit preparation; family unions can overlap','not_a_model_training_dataset':True})
 print(overall)
if __name__=='__main__':main()
