import hashlib,json
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

