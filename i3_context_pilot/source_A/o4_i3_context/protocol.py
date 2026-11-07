import hashlib,json,math,statistics

VERSION='o4_i3_context_DIS_40x3_v1_20261107'
CRITERIA={'relative_l2_max':0.02,'max_abs_over_scale_max':0.02,
          'reference':'CPU FP32 operation on exact frozen FP16 operand values'}

def protocol(kind):
    if kind not in ('smoke','formal'):raise ValueError('Unknown collection kind')
    return {'protocol_id':VERSION if kind=='formal' else 'o4_i3_context_DIS_smoke_v1_20261107',
       'kind':kind,'status':'engineering_development_not_strict_reference',
       'modes':['GI','P','R'],'independent_process_repeats':3 if kind=='formal' else 1,
       'measured_iterations':40 if kind=='formal' else 3,'transition_iterations':5 if kind=='formal' else 1,
       'warmup_min':50 if kind=='formal' else 8,'warmup_max':500 if kind=='formal' else 8,
       'warmup_window':20,'stable_windows':3,'window_change_threshold':.05,
       'repeat_spread_threshold':.05,'within_process_tail_review_threshold':.20,
       'schedule_seed':61107,'trace_backend':'cuda-sw','dtype':'float16',
       'eager':True,'autocast':False,'compile':False,'cuda_graph':False,
       'single_stream_required':True,'reference_warmup_active':True,
       'GPU_monitor_interval_seconds':5,'source_snapshot_required':True,
       'finalization':'release own resources; synchronize; own-context reset; profiler stop; no drain/sleep',
       'scope':'frozen scale0 CBR/residual D/I/S; all node outputs retained until next prepare outside timing',
       'independent_control':'precomputed local D node-input bytes outside capture; data edges removed, single-stream order retained',
       'input_asset_policy':'shared external operands; local derived intermediate bytes and hashes frozen before cells; no per-worker regeneration',
       'criteria':dict(CRITERIA),'strict_reference_verified':False,'formal_training_eligible':False}

def validate_protocol(p):
    if p!=protocol(p.get('kind')):raise ValueError('Frozen protocol mismatch')
    return p

def warmup_stable(values,p):
    if len(values)<max(p['warmup_min'],p['warmup_window']*p['stable_windows']):return False
    if any(not math.isfinite(x) or x<=0 for x in values):raise ValueError('Invalid warmup duration')
    w=p['warmup_window'];last=values[-3*w:]
    med=[statistics.median(last[i*w:(i+1)*w]) for i in range(3)]
    return all(abs(b-a)/a<=p['window_change_threshold'] for a,b in zip(med,med[1:]))

def distribution(values):
    if not values or any(not math.isfinite(v) or v<=0 for v in values):raise ValueError('Invalid duration sample')
    a=sorted(values);mid=statistics.median(a)
    def q(f):
        pos=(len(a)-1)*f;i=int(pos);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(pos-i)
    return {'n':len(a),'median_us':mid,'min_us':a[0],'max_us':a[-1],'p10_us':q(.1),'p90_us':q(.9),
       'MAD_us':statistics.median(abs(v-mid) for v in a),'tail_fraction':(q(.9)-q(.1))/mid}

def measured_values(rows,p):
    selected=[r for r in rows if r['stage']=='measure']
    if len(selected)!=p['measured_iterations'] or sorted(int(r['iteration']) for r in selected)!=list(range(p['measured_iterations'])):
        raise ValueError('Measured iteration set incomplete/duplicate')
    if any(r['status']!='completed' for r in selected):raise ValueError('Incomplete measured iteration')
    values=[float(r['elapsed_us']) for r in selected];distribution(values);return values

def schedule(configs,p):
    if len(set(configs))!=len(configs) or not 0<len(configs)<=12:raise ValueError('Invalid/oversize batch')
    jobs=[]
    for rep in range(1,p['independent_process_repeats']+1):
        ordered=sorted(configs,key=lambda c:hashlib.sha256(f'61107|S1|{rep}|{c}'.encode()).hexdigest())
        modes=[['GI','P','R'],['P','R','GI'],['R','GI','P']][(rep-1)%3]
        for c in ordered:
            for mode in modes:jobs.append({'config_id':c,'repeat':rep,'mode':mode,'folder':f'{c}_r{rep}_{mode}'})
    return jobs
