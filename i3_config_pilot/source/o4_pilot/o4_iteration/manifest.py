from .common import ROOT,read,sha
PROTOCOL='o4_gpu_iteration_smoke_v2_20261106'
IDS=('gemm_16_64_64','bmm_2_16_64_64','conv_1_16_32_32_k3_s1_p1','RN-A')
POLICY={'protocol_id':PROTOCOL,'status':'frozen_engineering_pilot','upstream_spec_commit':'891d04aeb74043e23e5309bca2856b61e01057a5','modes':['GI','P','R'],'target_kind':'iteration_device_event_us','measured_iterations':40,'transition_iterations':5,'warmup_min':50,'warmup_max':500,'warmup_window':20,'stable_windows':3,'stability_threshold':0.05,'repeat_spread_threshold':0.05,'schedule_seed':61106,'dtype':'float16','execution':'eager','autocast':False,'compile':False,'cuda_graph':False,'single_stream_required':True,'completion':'end_event.synchronize per iteration','correctness_rtol':0.01,'correctness_atol':0.01,'resnet_relative_l2_max':0.01,'resnet_max_abs_over_scale_max':0.01,'finalization':'release events/tensors; synchronize; sleep1s; worker-own cudaDeviceReset; cudaProfilerStop','reference_level':'ATen dispatch leaf operations; no module double counting','P_markers':'iteration scope only; no op markers or per-op sync','source_view':'raw individual iterations only; no repeated-event aggregation chosen','taxonomy_version':'aten_leaf_inventory_v1','candidate_limit':'not chosen; no predictor training in this pilot'}
def stable(values,policy=POLICY):
 import statistics
 w=policy['warmup_window'];n=policy['stable_windows']
 if len(values)<max(policy['warmup_min'],w*n):return False
 med=[statistics.median(values[len(values)-w*(n-i):len(values)-w*(n-i-1)] if i<n-1 else values[-w:]) for i in range(n)]
 return all(a>0 and abs(b/a-1)<=policy['stability_threshold'] for a,b in zip(med,med[1:]))
def validate_request(req):
 if req['protocol']!=POLICY: raise ValueError('Protocol differs from frozen engineering pilot')
 if req['mode'] not in POLICY['modes'] or req['workload'] not in IDS:raise ValueError('Unsupported workload/mode')
 if req.get('planned_only',False):raise ValueError('planned_only cannot execute')
 for item in req['assets'].values():
  p=ROOT/item['path']
  if not p.is_file() or sha(p)!=item['sha256']:raise ValueError('Asset hash mismatch: '+str(p))
 return req
