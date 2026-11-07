"""A0 deterministic controlled assets and operation adapters, independent of DNN weights."""
import argparse, hashlib, json, sys
from pathlib import Path

VERSION='o4_i3_A0_controlled_v1_20261107'
CRITERIA={'relative_l2_max':0.02,'max_abs_over_scale_max':0.02,
          'reference':'CPU FP32 operation on exact fp16 operand values; parity only, not task accuracy'}
def configs():
    rows=[]
    def add(kind,variant,operands,params,output):
        rows.append({'id':f'A0_{kind}_{variant}','subtype':kind,'operands':operands,'parameters':params,
            'output_shape':output,'dtype':'float16','layout':'contiguous','weights_kind':'initialized_performance_only',
            'scope':'one specified controlled operation; no surrounding activation or synchronization',
            'origin':'new controlled configuration; selected to bridge observed interfaces, not same DNN weight bytes',
            'inplace':False})
    for v,b in [('a',1),('b',2)]:
        add('conv',v,{'x':[b,3,224,224],'weight':[64,3,7,7]},
            {'stride':[2,2],'padding':[3,3],'dilation':[1,1],'groups':1,'bias':False},[b,64,112,112])
        add('gemm',v,{'x':[64*b,64],'weight':[64,96]}, {},[64*b,96])
        add('bmm',v,{'x':[4*b,32,64],'weight':[4*b,64,48]}, {},[4*b,32,48])
        add('linear',v,{'x':[b,2048],'weight':[1000,2048],'bias':[1000]}, {'bias':True},[b,1000])
        add('batchnorm_eval',v,{'x':[b,64,112,112],'weight':[64],'bias':[64],'running_mean':[64],'running_var':[64]},
            {'training':False,'momentum':0.1,'eps':1e-5},[b,64,112,112])
        add('relu',v,{'x':[b,64,56,56]}, {'inplace':False},[b,64,56,56])
        add('add',v,{'x':[b,64,56,56],'other':[b,64,56,56]}, {'alpha':1,'broadcast':False,'inplace':False},[b,64,56,56])
        add('maxpool',v,{'x':[b,64,112,112]},
            {'kernel_size':[3,3],'stride':[2,2],'padding':[1,1],'dilation':[1,1],'ceil_mode':False},[b,64,56,56])
        add('adaptive_average_pool',v,{'x':[b,2048,7,7]}, {'output_size':[1,1]},[b,2048,1,1])
    return rows

def validate_config(c):
    if c not in configs():raise ValueError('Configuration not in frozen A0 v1; create a new version')
    if c['inplace'] or c['dtype']!='float16':raise ValueError('Unsupported A0 state or dtype')

def write_json(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def generate(out):
    import torch
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if sys.byteorder!='little':raise ValueError('Frozen raw fp16 assets require little-endian serialization')
    generator=torch.Generator(device='cpu').manual_seed(20261107);rows=[]
    for c in configs():
        validate_config(c);folder=out/c['id'];folder.mkdir();operands={}
        for name,shape in c['operands'].items():
            value=torch.randn(shape,generator=generator,dtype=torch.float32)*0.1
            if name=='running_var':value=value.abs()+0.8
            if name=='weight' and c['subtype']=='batchnorm_eval':value=value+1
            value=value.half().contiguous();path=folder/(name+'.fp16.bin');path.write_bytes(value.numpy().tobytes())
            operands[name]={'path':str(path.relative_to(out)),'shape':shape,'dtype':'float16','sha256':sha(path),'bytes':path.stat().st_size}
        rows.append({**c,'asset_operands':operands,'configuration_sha256':hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest(),
            'correctness_criteria':dict(CRITERIA)})
    manifest={'version':VERSION,'generator_seed':20261107,'generation_device':'CPU','byteorder':'little','configurations':rows,
        'asset_policy':'share these exact bytes between nodes; do not regenerate per-node','reference_status':'development_unverified'}
    write_json(out/'ASSETS.json',manifest);return manifest

def verify_assets(path):
    path=Path(path);m=json.loads(path.read_text());root=path.parent.resolve();results=[]
    if m['version']!=VERSION or len(m['configurations'])!=18:raise ValueError('Unknown or incomplete A0 manifest')
    if len({c['id'] for c in m['configurations']})!=18:raise ValueError('Duplicate config IDs')
    for c in m['configurations']:
        pure={k:c[k] for k in configs()[0]};validate_config(pure)
        if c['correctness_criteria']!=CRITERIA:raise ValueError('Frozen correctness criteria changed')
        if c['configuration_sha256']!=hashlib.sha256(json.dumps(pure,sort_keys=True).encode()).hexdigest():raise ValueError('Config hash mismatch')
        for name,a in c['asset_operands'].items():
            f=(root/a['path']).resolve()
            if root not in f.parents or a['shape']!=c['operands'][name]:raise ValueError('Asset path or shape invalid')
            import math
            if f.stat().st_size!=math.prod(a['shape'])*2 or sha(f)!=a['sha256']:raise ValueError('Asset bytes mismatch: '+str(f))
            results.append({'path':a['path'],'sha256':a['sha256']})
        if set(c['operands'])!=set(c['asset_operands']):raise ValueError('Missing/extra operands')
    return {'manifest_sha256':sha(path),'verified_files':len(results),'files':results}

def operation(torch,c,tensors):
    import torch.nn.functional as F
    p=c['parameters'];x=tensors['x'];k=c['subtype']
    if k=='conv':return F.conv2d(x,tensors['weight'],None,p['stride'],p['padding'],p['dilation'],p['groups'])
    if k=='gemm':return torch.mm(x,tensors['weight'])
    if k=='bmm':return torch.bmm(x,tensors['weight'])
    if k=='linear':return F.linear(x,tensors['weight'],tensors['bias'])
    if k=='batchnorm_eval':return F.batch_norm(x,tensors['running_mean'],tensors['running_var'],tensors['weight'],tensors['bias'],p['training'],p['momentum'],p['eps'])
    if k=='relu':return F.relu(x,inplace=False)
    if k=='add':return torch.add(x,tensors['other'],alpha=p['alpha'])
    if k=='maxpool':return F.max_pool2d(x,p['kernel_size'],p['stride'],p['padding'],p['dilation'],p['ceil_mode'])
    if k=='adaptive_average_pool':return F.adaptive_avg_pool2d(x,p['output_size'])
    raise ValueError('Unsupported subtype: '+k)

class Adapter:
    def __init__(self,torch,manifest,config_id):
        self.torch=torch;self.manifest=Path(manifest);m=json.loads(self.manifest.read_text())
        self.config=next(c for c in m['configurations'] if c['id']==config_id);self.cpu={}
        for name,a in self.config['asset_operands'].items():
            f=self.manifest.parent/a['path']
            if sha(f)!=a['sha256']:raise ValueError('Operand checksum mismatch')
            self.cpu[name]=torch.frombuffer(bytearray(f.read_bytes()),dtype=torch.float16).reshape(a['shape']).clone()
        self.reference=operation(torch,self.config,{k:v.float() for k,v in self.cpu.items()}).detach()
        self.tensors={k:v.cuda() for k,v in self.cpu.items()}
        self.metadata={'configuration':self.config,'assets_manifest_sha256':sha(self.manifest),'reference_role':'evaluator_only'}
    def prepare(self):pass  # All A0 v1 operations are stateless/out-of-place.
    def forward(self):return operation(self.torch,self.config,self.tensors)
    def correctness(self):
        t=self.torch;result=self.forward();t.cuda.synchronize();actual=result.float().cpu();expected=self.reference
        if list(actual.shape)!=self.config['output_shape'] or not bool(t.isfinite(actual).all()):raise RuntimeError('Shape/finite failure')
        delta=actual-expected;relative=float(t.linalg.vector_norm(delta)/t.linalg.vector_norm(expected).clamp_min(1e-12));scaled=float(delta.abs().max()/expected.abs().max().clamp_min(1))
        criteria=self.config['correctness_criteria'];passed=relative<=criteria['relative_l2_max'] and scaled<=criteria['max_abs_over_scale_max']
        evidence={'passed':passed,'relative_l2':relative,'max_abs_over_scale':scaled,'criteria':criteria,'output_shape':list(actual.shape)}
        if not passed:raise RuntimeError('CPU FP32 parity failed: '+str(evidence))
        return evidence

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','verify','plan']);p.add_argument('--path',required=True);a=p.parse_args()
    if a.action=='generate':print('Generated',len(generate(a.path)['configurations']),'configurations')
    elif a.action=='verify':print(json.dumps(verify_assets(a.path),indent=2))
    else:write_json(a.path,{'status':'planned_only','configurations':configs(),'model_training':False})
