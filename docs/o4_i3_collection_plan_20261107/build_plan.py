"""CPU-only design generator. No torch, subprocess, GPU execution or asset generation."""
import csv, hashlib, json, math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
def dump(name, obj):
    (ROOT/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
def sha_obj(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def table(name, rows):
    with (ROOT/name).open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

TRAIN={0,3,4,7,8,11}; VAL={1,6,9}; TEST={2,5,10}
def split(i):return 'train' if i in TRAIN else 'validation' if i in VAL else 'test_exploratory'
# Twelve distinct spatial/channel or matrix-size anchors per family. Batch/layout
# variants never create extra independent groups.
C=[16,24,32,48,64,80,96,128,160,192,256,320]
H=[112,96,80,64,56,48,40,32,28,24,16,14]
B=[1,1,2,1,2,4,1,2,4,1,2,4]
MNK=[(32,48,64),(48,64,96),(64,96,128),(96,128,160),
     (128,160,192),(160,192,256),(192,256,320),(256,320,384),
     (320,384,512),(384,512,640),(512,640,768),(640,768,1024)]
families=['conv','depthwise','gemm','bmm','linear','batchnorm_eval','relu',
          'add','maxpool','adaptive_average_pool','bilinear_resize','concat']
rows=[]
def strides(shape, layout):
    if layout=='channels_last':
        n,c,h,w=shape;return [h*w*c,1,w*c,c]
    if layout=='transpose_last2_view':
        # Backing shape swaps the last axes; the returned logical view keeps shape.
        base=list(shape);base[-2:]=[shape[-1],shape[-2]]
        s=strides(base,'contiguous');s[-2:]=[s[-1],s[-2]];return s
    s=[];v=1
    for d in reversed(shape):s.insert(0,v);v*=d
    return s
def config(f,i,variant='base', **override):
    b,c,h=B[i],C[i],H[i];p={};x={};out=None
    if f in ('conv','depthwise'):
        g=1 if f=='conv' else c;co=2*c if f=='conv' else c
        p=dict(stride=[1,1],padding=[1,1],dilation=[1,1],groups=g,bias=False)
        x={'x':[b,c,h,h],'weight':[co,c//g,3,3]};out=[b,co,h,h]
    elif f in ('gemm','bmm'):
        m,n,k=MNK[i]
        x={'x':[m,k],'weight':[k,n]};out=[m,n]
        if f=='bmm':x={k0:[B[i]*4]+v for k0,v in x.items()};out=[B[i]*4]+out
    elif f=='linear':
        m,n,k=MNK[i];x={'x':[b,m,k],'weight':[n,k],'bias':[n]};out=[b,m,n];p={'bias':True}
    else:
        x={'x':[b,c,h,h]};out=[b,c,h,h]
        if f=='batchnorm_eval':
            x.update({n:[c] for n in ['weight','bias','running_mean','running_var']});p=dict(training=False,momentum=.1,eps=1e-5)
        elif f=='relu':p={'inplace':False}
        elif f=='add':x['other']=[b,c,h,h];p=dict(alpha=1,broadcast=False,inplace=False)
        elif f=='maxpool':p=dict(kernel_size=[3,3],stride=[2,2],padding=[1,1],dilation=[1,1],ceil_mode=False);out=[b,c,(h+1)//2,(h+1)//2]
        elif f=='adaptive_average_pool':p={'output_size':[1,1]};out=[b,c,1,1]
        elif f=='bilinear_resize':p=dict(size=[2*h,2*h],mode='bilinear',align_corners=False,antialias=False);out=[b,c,2*h,2*h]
        elif f=='concat':x['other']=[b,c//2,h,h];p={'dim':1};out=[b,c+c//2,h,h]
    layout='contiguous';operand_layouts={k:'contiguous' for k in x}
    if variant=='control1':
        if f=='conv':p['stride']=[2,2];out[-2:]=[(h+1)//2]*2
        elif f=='depthwise':p['groups']=c//2;x['weight']=[c,2,3,3] # matched group vs depthwise
        elif f in ('gemm','bmm'):operand_layouts['weight']='transpose_last2_view'
        elif f=='linear':x.pop('bias');p['bias']=False
        elif f=='batchnorm_eval':p['eps']=1e-3
        elif f=='relu':
            x['x'][0]*=2;out[0]*=2
        elif f=='add':x['other']=[1,c,1,1];p['broadcast']=True
        elif f=='maxpool':p['stride']=[1,1];out[-2:]=[h,h]
        elif f=='adaptive_average_pool':p['output_size']=[2,2];out[-2:]=[2,2]
        elif f=='bilinear_resize':p['align_corners']=True
        elif f=='concat':p['dim']=2;x['other']=[b,c,h//2,h];out=[b,c,h+h//2,h]
    if variant=='control2':
        if f in ('gemm','bmm','linear'):
            operand_layouts['x']='transpose_last2_view'
        else:
            layout='channels_last'
            for key,shape in x.items():
                if len(shape)==4 and key!='weight':operand_layouts[key]='channels_last'
    if f=='conv':
        if variant=='kernel1':x['weight'][-2:]=[1,1];p['padding']=[0,0]
        elif variant=='dilation2':p['dilation']=[2,2];p['padding']=[2,2]
        elif variant=='groups4':p['groups']=4;x['weight'][1]//=4
        elif variant=='batch4':x['x'][0]=4;out[0]=4
        elif variant=='padding0':p['padding']=[0,0];out[-2:]=[h-2,h-2]
        elif variant=='bias_true':p['bias']=True;x['bias']=[2*c];operand_layouts['bias']='contiguous'
    assert not override
    tensors={k:{'shape':v,'stride':strides(v,operand_layouts[k]),'layout':operand_layouts[k],
                'storage_offset':0,'dtype':'float16'} for k,v in x.items()}
    semantic={'family':f,'tensors':tensors,'parameters':p,'output_shape':out,
              'execution':'eager_unfused','inplace':False}
    return {'id':f'I3_{f}_{i:02d}_{variant}','family':f,'anchor':i,'variant':variant,
            'split_B':split(i),'leakage_group':f'{f}_anchor_{i:02d}',
            'semantic_sha256':sha_obj(semantic),'definition':semantic,
            'phase':'core_first' if i in [0,3,4] and variant=='base' else 'core_expand',
            'asset_status':'not_generated','adapter_status':'requires_new_version',
            'gpu_status':'not_run'}
for f in families:
    for i in range(12):rows.append(config(f,i))
    for v in ['control1','control2']:rows.append(config(f,0,v))
for v in ['kernel1','dilation2','groups4','batch4','padding0','bias_true']:
    rows.append(config('conv',0,v))
dump('OPERATOR_CONFIGS.json',{'status':'planned_only','configurations':rows})
table('OPERATOR_MATRIX.csv',[{'id':r['id'],'family':r['family'],'anchor':r['anchor'],
    'variant':r['variant'],'split_B':r['split_B'],'leakage_group':r['leakage_group'],
    'inputs':json.dumps(r['definition']['tensors'],separators=(',',':')),
    'parameters':json.dumps(r['definition']['parameters'],separators=(',',':')),
    'output_shape':str(r['definition']['output_shape']),'phase':r['phase'],'gpu_status':'not_run'} for r in rows])

# Explicit dependency graphs. External inputs are tensor names; node IDs are edges.
graphs=[]
for scale,(b,c,h,sp) in enumerate([(1,20,36,'train'),(2,40,24,'validation'),(1,80,12,'test_exploratory')]):
    shape=[b,c,h,h]
    def graph(kind,external,specs):
        known=dict(external);nodes=[]
        for name,op,inputs,params,output in specs:
            assert all(s in known for s in inputs),(kind,name)
            ns={'id':name,'op':op,'input_edges':inputs,'input_shapes':[known[s] for s in inputs],
                'parameters':params,'output_shape':output}
            nodes.append(ns);known[name]=output
        graphs.append({'id':f'CTX_{kind}_{scale}','family':kind,'scale':scale,'split_B':sp,
          'leakage_group':f'context_scale_{scale}','external_inputs':external,'nodes':nodes,
          'output_nodes':[nodes[-1]['id']],'dtype':'float16','layout':'contiguous',
          'timing_scope':'all listed nodes; no per-op synchronization; all external assets preloaded',
          'independent_control_id':f'IND_{kind}_{scale}',
          'isolated_control_ids':[f'ISO_{kind}_{scale}_{n["id"]}' for n in nodes],
          'asset_status':'not_generated','adapter_status':'not_implemented','gpu_status':'not_run'})
    conv=lambda ci,co,k=3,s=1,g=1:dict(weight_shape=[co,ci//g,k,k],stride=[s,s],padding=[k//2]*2,dilation=[1,1],groups=g,bias=False)
    bn=dict(training=False,eps=1e-5,momentum=.1,affine=True,running_stats=True)
    graph('conv_bn_relu',{'x':shape},[
      ('c','conv',['x'],conv(c,c),shape),('n','batchnorm_eval',['c'],bn,shape),('r','relu',['n'],{'inplace':False},shape)])
    graph('residual',{'x':shape},[
      ('c1','conv',['x'],conv(c,c),shape),('n1','batchnorm_eval',['c1'],bn,shape),('r1','relu',['n1'],{'inplace':False},shape),
      ('c2','conv',['r1'],conv(c,c),shape),('n2','batchnorm_eval',['c2'],bn,shape),
      ('a','add',['n2','x'],{'alpha':1,'broadcast':False},shape),('r2','relu',['a'],{'inplace':False},shape)])
    graph('depthwise_pointwise',{'x':shape},[
      ('dw','depthwise',['x'],conv(c,c,g=c),shape),('n','batchnorm_eval',['dw'],bn,shape),
      ('hs','hardswish',['n'],{'inplace':False},shape),('pw','conv',['hs'],conv(c,2*c,k=1),[b,2*c,h,h])])
    small=[b,c,1,1];hidden=[b,c//4,1,1]
    graph('se_gate',{'x':shape},[
      ('pool','adaptive_average_pool',['x'],{'output_size':[1,1]},small),
      ('reduce','conv',['pool'],{**conv(c,c//4,k=1),'bias':True},hidden),
      ('r','relu',['reduce'],{'inplace':False},hidden),
      ('expand','conv',['r'],{**conv(c//4,c,k=1),'bias':True},small),
      ('gate','hardsigmoid',['expand'],{'inplace':False},small),
      ('scale','mul',['x','gate'],{'broadcast':True},shape)])
    high=[b,c,2*h,2*h];cat=[b,2*c,2*h,2*h]
    graph('resize_concat_conv',{'x':shape,'skip':high},[
      ('up','bilinear_resize',['x'],dict(size=[2*h]*2,align_corners=False,antialias=False,mode='bilinear'),high),
      ('cat','concat',['up','skip'],{'dim':1},cat),
      ('c','conv',['cat'],conv(2*c,c),high),('r','relu',['c'],{'inplace':False},high)])
    heads=4;length=[32,64,128][scale];d=32;batch=b*heads
    q=[batch,length,d];kt=[batch,d,length];scores=[batch,length,length]
    graph('attention_eager',{'q':q,'kt':kt,'v':q},[
      ('qk','bmm',['q','kt'],{},scores),('scale','mul_scalar',['qk'],{'value':d**-.5},scores),
      ('softmax','softmax',['scale'],{'dim':-1,'dtype':'float16'},scores),
      ('av','bmm',['softmax','v'],{},q)])
dump('CONTEXT_PLAN.json',{'status':'planned_only','graphs':graphs,
    'independent_control':'replace every intermediate edge by a precomputed same-node tensor outside capture; issue all nodes in the same order and stream, retaining all outputs until end; this removes data edges, not CUDA stream ordering',
    'isolated_control':'one separate workload per node position, same precomputed inputs and params; no events inside the dependent graph',
    'scope_limit':'six controlled templates, not full RN/MV/UN/YO/SW/Qwen forward or production fused attention'})

core_first=[r['id'] for r in rows if r['phase']=='core_first']
sentinels=[f'I3_{f}_00_base' for f in ['batchnorm_eval','gemm','bilinear_resize']]
assert set(sentinels)<=set(core_first)
work=[]
for r in rows:work.append({'work_id':r['id'],'kind':'isolated_base_or_control','phase':r['phase'],'split_B':r['split_B'],'leakage_group':r['leakage_group']})
for g in graphs:
    for name,kind in [(g['id'],'dependent_graph'),(g['independent_control_id'],'independent_sequence')]+[(x,'isolated_graph_node') for x in g['isolated_control_ids']]:
        work.append({'work_id':name,'kind':kind,'phase':'contexts_'+g['split_B'],'split_B':g['split_B'],'leakage_group':g['leakage_group']})
table('WORK_ITEMS.csv',work)
cells=[]
for w in work:
    for rep in range(3):
        order=[['GI','P','R'],['P','R','GI'],['R','GI','P']][rep]
        for pos,mode in enumerate(order):cells.append({**w,'node':'GB10','repeat':rep+1,'mode':mode,
          'within_work_order':pos,'measured':40,'transition':5,'status':'planned_only'})
table('GB10_CELL_PLAN.csv',cells)
summary={'status':'planning_only_not_GPU_results','base_configs':144,'contrast_configs':30,
    'base_independent_groups':144,'base_groups_per_family':12,
    'graphs':len(graphs),'independent_sequences':len(graphs),
    'isolated_node_positions':sum(len(g['nodes']) for g in graphs),'total_work_items':len(work),
    'formal_cells_upper_bound_per_node':len(cells),'formal_measured_upper_bound_per_node':40*len(cells),
    'formal_P_R_captures_upper_bound_per_node':6*len(work),'minimum_warmup_iterations_per_cell':60,
    'sentinels':sentinels,'core_first_work_items':len(core_first),
    'core_first_cells_including_sentinels':9*len(core_first),
    'core_first_measured':360*len(core_first),
    'remaining_core_work_items':len(rows)-len(core_first),
    'phase_work_items':dict(Counter(w['phase'] for w in work)),
    'count_scope':'upper bound if all planned supported work is collected GI/P/R x3; excludes smoke, retries, optional extensions and prior A0 data; not an execution quota',
    'schedule':'repeat rounds; stable hash order within each phase/round; mode Latin rotation from table; emit actual sequence before launch'}
dump('PLAN_COUNTS.json',summary)

# Algebraic audit only; no claim that tensors/adapters/backends have run.
checks=[]
def check(name,condition):
    assert condition,name;checks.append(name)
check('unique_work_ids',len({w['work_id'] for w in work})==len(work))
check('unique_base_semantic_hashes',len({r['semantic_sha256'] for r in rows})==len(rows))
for f in families:
    selected=[r for r in rows if r['family']==f and r['variant']=='base']
    check(f+'_6_3_3_groups',Counter(r['split_B'] for r in selected)=={'train':6,'validation':3,'test_exploratory':3})
for r in rows:
    d=r['definition'];p=d['parameters'];ts=d['tensors'];out=d['output_shape']
    check(r['id']+'_positive_shapes',all(v>0 for t in ts.values() for v in t['shape']) and all(v>0 for v in out))
    if d['family'] in ('conv','depthwise'):
        x=ts['x']['shape'];w=ts['weight']['shape'];g=p['groups']
        check(r['id']+'_conv_groups',x[1]%g==0 and w[0]%g==0 and w[1]*g==x[1])
        expected=[x[0],w[0]]+[(x[2+j]+2*p['padding'][j]-p['dilation'][j]*(w[2+j]-1)-1)//p['stride'][j]+1 for j in range(2)]
        check(r['id']+'_conv_output',out==expected)
    check(r['id']+'_group_fixed',r['split_B']==split(r['anchor']))
check('all_context_edges_topological',all(all(s in g['external_inputs'] or s in [x['id'] for x in g['nodes'][:j]] for s in n['input_edges']) for g in graphs for j,n in enumerate(g['nodes'])))
check('all_context_positions_have_isolated_control',all(len(g['nodes'])==len(g['isolated_control_ids']) for g in graphs))
check('contexts_no_cross_scale_group_split',all(len({w['split_B'] for w in work if w['leakage_group']==group})==1 for group in {w['leakage_group'] for w in work}))
check('nine_cells_each',len(cells)==9*len(work))
dump('VALIDATION.json',{'status':'passed_design_consistency_only','checks':len(checks),
    'check_names':checks,'GPU_executed':False,'adapters_implemented':False,
    'runtime_stride_or_correctness_verified':False,'full_ATen_signature_collision_audit':'pending actual R schema; split must not be frozen until audited',
    'no_statistical_power_claim':True})
print(json.dumps(summary,ensure_ascii=False,indent=2))
