from __future__ import annotations
import collections,csv,json,statistics
from pathlib import Path
from .common import read,write,csvwrite,jsonl,sha,now
from .trace import union_ns,closure_diagnostic

def loadlines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]
def signature(e):return [e.get(k) for k in ('event_kind','kernel_name','grid_xyz','block_xyz','dynamic_shared_memory_bytes','bytes','copy_kind','memory_kind','memset_value')]
def category(name):
 if any(s in name for s in ('convolution','conv2d')):return 'conv'
 if 'bmm' in name:return 'bmm'
 if any(s in name for s in ('addmm','linear')):return 'linear_bias'
 if any(s in name for s in ('aten.mm.','matmul')):return 'gemm'
 if 'batch_norm' in name:return 'batchnorm_eval'
 if 'relu' in name:return 'relu'
 if any(s in name for s in ('aten.add.','aten.add_.')):return 'residual_add'
 if 'pool' in name:return 'pooling'
 if 'mean' in name:return 'reduction'
 if any(s in name for s in ('view','reshape','flatten','as_strided','transpose','aten.t.','detach')):return 'layout_or_view'
 if any(s in name for s in ('copy','clone','contiguous','empty','resize')):return 'allocation_or_copy'
 return 'other_uncovered'

def composition_supported(quality,events):
 if 'composition_supported' in quality:return quality['composition_supported']
 # Older saved audits lack this field; derive it from preserved observable events.
 return not any(a['end_ns']>b['start_ns'] for a,b in zip(events,events[1:]))

def summarize(run,out):
 run=Path(run).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False);inv=read(run/'inventory.json');cells=[];groups=collections.defaultdict(list);traces=[]
 for j in inv['jobs']:
  row={k:j[k] for k in ('workload','mode','repeat','status')};row['median_us']=None;row['trace_status']=j.get('trace_status');row['sample_count']=0;row['error']=j.get('error','')
  if j.get('folder'):
   f=run/j['folder'];q=read(f/'quality.json') if (f/'quality.json').exists() else {};row['correctness_passed']=q.get('correctness',{}).get('passed',False);row['warmup_iterations']=q.get('warmup',{}).get('iterations')
   if (f/'iterations.csv').exists():
    data=[r for r in csv.DictReader((f/'iterations.csv').open()) if r['stage']=='measure'];row['sample_count']=len(data)
    if j['status']=='completed' and data:
     times=[float(r['elapsed_us']) for r in data];row.update(median_us=statistics.median(times),p10_us=sorted(times)[int(.1*(len(times)-1))],p90_us=sorted(times)[int(.9*(len(times)-1))],target_kind=data[0]['target_kind']);groups[(j['workload'],j['mode'])].append(row['median_us'])
   if j['mode']!='GI' and (run/'quality'/f.name/'quality.json').exists():traces.append({'workload':j['workload'],'repeat':j['repeat'],**read(run/'quality'/f.name/'quality.json')})
  cells.append(row)
 fields=sorted(set().union(*(r.keys() for r in cells)));csvwrite(out/'cells.csv',cells,fields)
 aggregate=[]
 for (w,m),values in sorted(groups.items()):
  med=statistics.median(values);spread=(max(values)-min(values))/med
  aggregate.append({'workload':w,'mode':m,'valid_processes':len(values),'expected_processes':inv['repeats'],'median_of_process_medians_us':med,'process_spread_pct':100*spread,'repeat_stability_passed':len(values)==inv['repeats'] and inv['repeats']>=3 and spread<=inv['protocol']['repeat_spread_threshold'],'target_kind':'iteration_device_event_us' if m=='GI' else 'profiled_iteration_event_us'})
 if aggregate:csvwrite(out/'aggregate.csv',aggregate)
 paired=[];mapped_ops=[];pair_details=[]
 for w in inv['workloads']:
  for rep in range(1,inv['repeats']+1):
   jobs={j['mode']:j for j in inv['jobs'] if j['workload']==w and j['repeat']==rep and j['status']=='completed'}
   if not all(m in jobs for m in ('P','R')):continue
   pdir=run/'quality'/Path(jobs['P']['folder']).name;rdir=run/'quality'/Path(jobs['R']['folder']).name
   pe=loadlines(pdir/'events_diagnostic.jsonl');re=loadlines(rdir/'events_diagnostic.jsonl');refs=loadlines(rdir/'reference_operations.jsonl');pq=read(pdir/'quality.json');rq=read(rdir/'quality.json');ok=True;matched=0
   for idx in range(inv['protocol']['measured_iterations']):
    scope=f'o4.iter/measure/{idx:06d}';p=[x for x in pe if x['iteration_scope']==scope];r=[x for x in re if x['iteration_scope']==scope]
    same=bool(p) and [signature(e) for e in p]==[signature(e) for e in r];ok &=same
    pair_details.append({'workload':w,'repeat':rep,'iteration':idx,'exact_ordered_signature':same,'P_events':len(p),'R_events':len(r)})
    if not same:continue
    matched+=1;parts=[]
    for op in [x for x in refs if x['stage']=='measure' and x['iteration']==idx]:
     ev=[p[k] for k in op['event_indices']];cat=category(op['aten']);related=cat in ('conv','bmm','gemm')
     row={**op,'workload':w,'repeat':rep,'category':cat,'old_three_family_related':related,'P_contextual_trace_span_us':(max(e['end_ns'] for e in ev)-min(e['start_ns'] for e in ev))/1000 if ev else None,'P_gpu_busy_union_us':union_ns([(e['start_ns'],e['end_ns']) for e in ev])/1000 if ev else 0,'P_event_count':len(ev),'P_gpu_intervals_ns':[(e['start_ns'],e['end_ns']) for e in ev],'label_status':'warning_accepted' if pq['warnings'] or rq['warnings'] else 'structural_only_pending_review'}
     mapped_ops.append(row)
     if ev:parts.append(ev)
    x=closure_diagnostic(parts)
    if x['composition_supported'] and x['closure_error_ns']!=0:raise ValueError('Transferred reference closure failed')
   paired.append({'workload':w,'repeat':rep,'matched_iterations':matched,'expected_iterations':inv['protocol']['measured_iterations'],'exact_all_measured':ok,'warning_count':len(pq['warnings'])+len(rq['warnings']),'strict_verified_operations':0,'P_composition_supported':composition_supported(pq,pe),'R_composition_supported':composition_supported(rq,re),'alignment_assumption':'known iteration + equal complete ordered event signature; no duration matching; not proof against profiler loss'})
 write(out/'pair_summary.json',paired);jsonl(out/'evaluator_only/reference_transferred_to_P.jsonl',mapped_ops)
 if pair_details:csvwrite(out/'evaluator_only/pair_iterations.csv',pair_details)
 def busy_total(records):
  groups=collections.defaultdict(list)
  for row in records:groups[(row['repeat'],row['iteration'])].extend(row['P_gpu_intervals_ns'])
  return sum(union_ns(values)/1000 for values in groups.values())
 categories=[];coverage=[]
 for w in inv['workloads']:
  ops=[r for r in mapped_ops if r['workload']==w];total_events=sum(r['P_event_count'] for r in ops);total_busy=busy_total(ops);gpu_ops=[r for r in ops if r['P_event_count']];base=[r for r in gpu_ops if r['old_three_family_related']]
  coverage.append({'workload':w,'mapped_gpu_operations':len(gpu_ops),'zero_gpu_operations':len(ops)-len(gpu_ops),'old_three_family_event_pct':100*sum(r['P_event_count'] for r in base)/total_events if total_events else None,'old_three_family_busy_pct':100*busy_total(base)/total_busy if total_busy else None,'max_events_per_operation':max((r['P_event_count'] for r in ops),default=None),'operations_over_L4':sum(r['P_event_count']>4 for r in gpu_ops),'event_pct_in_operations_over_L4':100*sum(r['P_event_count'] for r in gpu_ops if r['P_event_count']>4)/total_events if total_events else None,'note':'family relation only; does not establish shape/model support; warning-accepted'})
  for cat in sorted({r['category'] for r in ops}):
   subset=[r for r in ops if r['category']==cat];categories.append({'workload':w,'category':cat,'aten_names':';'.join(sorted({r['aten'] for r in subset})),'operations':len(subset),'zero_gpu_operations':sum(not r['P_event_count'] for r in subset),'P_events':sum(r['P_event_count'] for r in subset),'P_event_pct':100*sum(r['P_event_count'] for r in subset)/total_events if total_events else None,'P_busy_pct':100*busy_total(subset)/total_busy if total_busy else None,'P_span_sum_us':sum(r['P_contextual_trace_span_us'] or 0 for r in subset),'max_events_per_operation':max(r['P_event_count'] for r in subset)})
 if categories:csvwrite(out/'operation_inventory.csv',categories)
 if coverage:csvwrite(out/'family_coverage.csv',coverage)
 warn=sorted({s for t in traces for s in t['warnings']});summary={'utc':now(),'run':str(run),'attempted_cells':sum(j['status']!='not_started' for j in inv['jobs']),'planned_cells':len(inv['jobs']),'completed_cells':sum(j['status']=='completed' for j in inv['jobs']),'failed_cells':sum(j['status']=='failed' for j in inv['jobs']),'measured_valid_iterations':sum(r['sample_count'] for r in cells if r['status']=='completed'),'traces':len(traces),'trace_structural_passes':sum(t['status']!='failed' for t in traces),'composition_supported_traces':sum(t.get('composition_supported',True) for t in traces),'overlapping_recorded_work_events':sum(t.get('overlapping_adjacent_work_events',0) for t in traces),'cpu_gpu_clock_inversions':sum(t.get('gpu_end_before_cpu_api_start_count',0) for t in traces),'reference_gate':'not_verified','strict_verified_operations':0,'warning_types':warn,'matched_P_R_pairs':sum(x['exact_all_measured'] for x in paired),'pairs_attempted':len(paired),'coverage':coverage,'inventory_sha256':sha(run/'inventory.json'),'collector_code_manifest_sha256':__import__('hashlib').sha256(json.dumps(inv['code_manifest'],sort_keys=True).encode()).hexdigest(),'no_model_training':True,'new_confirmation_data':False}
 write(out/'SUMMARY.json',summary)
 text=['# O4 GB10 GI/P/R 與運算盤點','',f'日期：{now()}。來源 `{run.name}`。本輪為工程 pilot，未訓練新辨識器／N0／ITER，未取得4090／3090新資料。','',f"完成 {summary['completed_cells']}/{summary['planned_cells']} cells；失敗 {summary['failed_cells']}；有效 measured iterations {summary['measured_valid_iterations']}。Reference strict verified=0。",'','| 工作 | 模式 | process中位數的中位數 µs | process spread | 穩定門檻 |','|---|---|---:|---:|---|']
 for a in aggregate:text.append(f"| {a['workload']} | {a['mode']} | {a['median_of_process_medians_us']:.3f} | {a['process_spread_pct']:.2f}% | {'通過' if a['repeat_stability_passed'] else '未通過／重複不足'} |")
 text+=['','GI 為無 profiler event interval；P/R 表中為 profiled event interval，不互換為 GI。Repeat先取各程序40筆中位數，再聚合程序；未穩定樣本不刪除。','',f"P/R 完整有序signature相同：{summary['matched_P_R_pairs']}/{summary['pairs_attempted']} 對。比對含kernel診斷名稱／geometry及copy、memset；主P匯出不含名稱。此為條件結構對應，不解除warning，也不證明無事件遺失。",'','警告：'+('; '.join(warn) or '未從DIAGNOSTIC_EVENT取得warning；仍須人工核對log/report'),'', '| 工作 | 舊三類相關 event% | 舊三類相關 busy% | 最大events/op | 超L4的GPU ops |','|---|---:|---:|---:|---:|']
 for c in coverage:
  if c['old_three_family_event_pct'] is not None:text.append(f"| {c['workload']} | {c['old_three_family_event_pct']:.2f} | {c['old_three_family_busy_pct']:.2f} | {c['max_events_per_operation']} | {c['operations_over_L4']} |")
 text+=['','觀察重疊時不支援直接SUM／舊連續分段模型；不修改或裁切原timestamps。busy按每iteration的GPU interval union計算，跨類別重疊可使各類比例加總超過100%。上述為family關聯，不能當舊模型已支援比例；實際shape、Conv參數、bias與執行環境仍需檢查。busy是已記錄工作interval union，不是全GPU利用率。零GPU工作保留null span。','', '完整逐類結果見 [operation_inventory.csv](operation_inventory.csv)，逐筆 reference 位於 evaluator_only；P-only數值輸入位於來源 run 的 quality/cell_*/predictor。','', '下一步：4090／3090同版本收集；先檢查reference與閉合，再依RN-A缺類及時間占比補受控資料。新確認配置、DNN完整預測、正式reference驗收尚未完成。']
 (out/'REPORT.md').write_text('\n'.join(text)+'\n');print(json.dumps(summary,ensure_ascii=False,indent=2));return 0
