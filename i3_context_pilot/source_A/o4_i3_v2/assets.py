"""Frozen config catalog and logical FP16 bytes; no per-node regeneration."""
import argparse,hashlib,json,math,sys
from pathlib import Path
from .protocol import CRITERIA
from o4_research.common import read,write,sha

VERSION='o4_i3_controlled_assets_v2_20261107'
CATALOG=Path(__file__).with_name('CONFIGURATIONS.json')
def configurations():return read(CATALOG)['configurations']
def lookup(cid):
    found=[c for c in configurations() if c['id']==cid]
    if len(found)!=1:raise ValueError('Unknown configuration '+cid)
    return found[0]

def operation(torch,c,t):
    import torch.nn.functional as F
    d=c['definition'];p=d['parameters'];k=d['family'];x=t['x']
    if k in ('conv','depthwise'):return F.conv2d(x,t['weight'],t.get('bias'),p['stride'],p['padding'],p['dilation'],p['groups'])
    if k=='gemm':return torch.mm(x,t['weight'])
    if k=='bmm':return torch.bmm(x,t['weight'])
    if k=='linear':return F.linear(x,t['weight'],t.get('bias'))
    if k=='batchnorm_eval':return F.batch_norm(x,t['running_mean'],t['running_var'],t['weight'],t['bias'],p['training'],p['momentum'],p['eps'])
    if k=='relu':return F.relu(x,inplace=False)
    if k=='add':return torch.add(x,t['other'],alpha=p['alpha'])
    if k=='maxpool':return F.max_pool2d(x,p['kernel_size'],p['stride'],p['padding'],p['dilation'],p['ceil_mode'])
    if k=='adaptive_average_pool':return F.adaptive_avg_pool2d(x,p['output_size'])
    if k=='bilinear_resize':return F.interpolate(x,size=p['size'],mode=p['mode'],align_corners=p['align_corners'],antialias=p['antialias'])
    if k=='concat':return torch.cat([x,t['other']],dim=p['dim'])
    raise ValueError('Unsupported family '+k)

def logical_tensor(torch,c,name):
    spec=c['definition']['tensors'][name]
    # Same anchor/name/shape gets the same values across layout/parameter controls.
    key=f'61107|{c["family"]}|{c["anchor"]}|{name}'
    seed=int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'little')%(2**63-1)
    g=torch.Generator(device='cpu').manual_seed(seed)
    t=torch.randn(spec['shape'],generator=g,dtype=torch.float32)*.1
    if name=='running_var':t=t.abs()+.8
    if name=='weight' and c['family']=='batchnorm_eval':t=t+1
    return t.half().contiguous()

def generate(out,ids):
    import torch
    if sys.byteorder!='little':raise ValueError('Assets require little endian')
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if len(set(ids))!=len(ids) or not ids:raise ValueError('Duplicate/empty selection')
    entries=[]
    for cid in ids:
        c=lookup(cid);folder=out/cid;folder.mkdir();operands={}
        for name,spec in c['definition']['tensors'].items():
            t=logical_tensor(torch,c,name);f=folder/(name+'.fp16.bin');f.write_bytes(t.numpy().tobytes())
            operands[name]={'path':str(f.relative_to(out)),'shape':spec['shape'],'bytes':f.stat().st_size,'sha256':sha(f),'serialization':'logical_contiguous_FP16_little_endian'}
        entries.append({'configuration':c,'operands':operands,'criteria':dict(CRITERIA)})
    m={'version':VERSION,'catalog_sha256':sha(CATALOG),'configurations':entries,
       'weights_kind':'initialized_performance_only','policy':'generate once on CPU; distribute exact bytes to other nodes'}
    write(out/'ASSETS.json',m);return m

def verify(manifest):
    path=Path(manifest).resolve();m=read(path);root=path.parent;seen=set();count=0
    if m['version']!=VERSION or m['catalog_sha256']!=sha(CATALOG):raise ValueError('Asset schema/catalog mismatch')
    if not m['configurations']:raise ValueError('Empty assets')
    for entry in m['configurations']:
        c=entry['configuration'];cid=c['id']
        if cid in seen or c!=lookup(cid) or entry['criteria']!=CRITERIA:raise ValueError('Changed config/criteria or duplicate ID')
        seen.add(cid)
        if set(entry['operands'])!=set(c['definition']['tensors']):raise ValueError('Missing/extra operands')
        for name,a in entry['operands'].items():
            spec=c['definition']['tensors'][name];f=(root/a['path']).resolve()
            if root not in f.parents or a['shape']!=spec['shape'] or a['bytes']!=math.prod(spec['shape'])*2:raise ValueError('Invalid operand metadata/path')
            if f.stat().st_size!=a['bytes'] or sha(f)!=a['sha256']:raise ValueError('Operand size/hash mismatch '+str(f))
            count+=1
    return {'manifest_sha256':sha(path),'configurations':len(seen),'operand_files':count}

def load_tensor(torch,logical,spec,device):
    # Reconstruct view on CPU outside timing; preserve format in device transfer.
    view=torch.empty_strided(spec['shape'],spec['stride'],dtype=logical.dtype,device='cpu');view.copy_(logical)
    value=view.to(device=device,memory_format=torch.preserve_format)
    if list(value.stride())!=spec['stride'] or value.storage_offset()!=spec['storage_offset']:raise ValueError('Actual tensor layout/offset differs from frozen definition')
    return value

class Adapter:
    def __init__(self,torch,manifest,cid):
        verify(manifest);self.torch=torch;path=Path(manifest);entry=next(e for e in read(path)['configurations'] if e['configuration']['id']==cid)
        self.config=entry['configuration'];self.cpu={}
        for name,a in entry['operands'].items():
            self.cpu[name]=torch.frombuffer(bytearray((path.parent/a['path']).read_bytes()),dtype=torch.float16).reshape(a['shape']).clone()
        self.reference=operation(torch,self.config,{n:t.float() for n,t in self.cpu.items()}).detach()
        self.tensors={n:load_tensor(torch,t,self.config['definition']['tensors'][n],'cuda') for n,t in self.cpu.items()}
        self.metadata={'configuration':self.config,'assets_manifest_sha256':sha(path),
          'actual_input_strides':{n:list(t.stride()) for n,t in self.tensors.items()},'metadata_role':'evaluator_only'}
    def prepare(self):pass
    def forward(self):return operation(self.torch,self.config,self.tensors)
    def correctness(self):
        t=self.torch;states={k:v.clone() for k,v in self.tensors.items() if k in ('running_mean','running_var')}
        output=self.forward();t.cuda.synchronize();actual=output.float().cpu();expected=self.reference
        if list(actual.shape)!=self.config['definition']['output_shape'] or not bool(t.isfinite(actual).all()):raise RuntimeError('Shape/nonfinite correctness failure')
        delta=actual-expected;rel=float(t.linalg.vector_norm(delta)/t.linalg.vector_norm(expected).clamp_min(1e-12));scaled=float(delta.abs().max()/expected.abs().max().clamp_min(1))
        state_ok=all(t.equal(self.tensors[k],v) for k,v in states.items())
        result={'passed':rel<=CRITERIA['relative_l2_max'] and scaled<=CRITERIA['max_abs_over_scale_max'] and state_ok,
           'relative_l2':rel,'max_abs_over_scale':scaled,'criteria':CRITERIA,'state_unchanged':state_ok,
           'output_shape':list(output.shape),'output_fp16_sha256':hashlib.sha256(output.cpu().contiguous().numpy().tobytes()).hexdigest()}
        if not result['passed']:raise RuntimeError('CPU FP32 parity/state failed: '+str(result))
        return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['generate','verify']);p.add_argument('--path',required=True);p.add_argument('--config',nargs='+');a=p.parse_args()
    print(json.dumps(generate(a.path,a.config) if a.action=='generate' else verify(a.path),indent=2))
