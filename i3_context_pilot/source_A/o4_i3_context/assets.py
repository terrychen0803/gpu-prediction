"""Shared external FP16 assets; per-node exact D inputs for I/S, never regenerated in capture."""
import argparse,hashlib,json,math,shutil,sys
from pathlib import Path
from o4_research.common import read,write,sha
from o4_i3_v2.protocol import CRITERIA
PLAN=Path(__file__).with_name('PLAN.json')
BASE_VERSION='o4_context_external_fp16_v1_20261107'
VERSION='o4_context_local_DIS_fp16_v1_20261107'
def plan():return read(PLAN)
def work(cid):
 found=[x for x in plan()['work_items'] if x['id']==cid]
 if len(found)!=1:raise ValueError('Unknown context work '+cid)
 return found[0]
def record_tensor(torch,t,path,root):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);v=t.detach().cpu().to(torch.float16).contiguous()
 path.write_bytes(v.numpy().tobytes());return {'path':str(path.relative_to(root)),'shape':list(v.shape),'dtype':'float16','bytes':path.stat().st_size,'sha256':sha(path),'serialization':'logical_contiguous_FP16_little_endian'}
def check_record(root,a):
 f=(root/a['path']).resolve()
 if root not in f.parents or a['dtype']!='float16' or a['bytes']!=math.prod(a['shape'])*2:raise ValueError('Bad tensor metadata/path')
 if f.stat().st_size!=a['bytes'] or sha(f)!=a['sha256']:raise ValueError('Tensor bytes changed '+str(f))
def load(torch,root,a,device='cpu',fp32=False):
 v=torch.frombuffer(bytearray((root/a['path']).read_bytes()),dtype=torch.float16).reshape(a['shape']).clone()
 return v.to(device=device,dtype=torch.float32 if fp32 else torch.float16)
def operand_shapes(g):
 result={f'external/{n}':s for n,s in g['external_inputs'].items()}
 for n in g['nodes']:
  p=n['parameters'];prefix='node/'+n['id']+'/'
  if n['op']=='conv':
   result[prefix+'weight']=p['weight_shape']
   if p['bias']:result[prefix+'bias']=[p['weight_shape'][0]]
  if n['op']=='batchnorm_eval':
   for key in ['weight','bias','running_mean','running_var']:result[prefix+key]=[n['input_shapes'][0][1]]
 return result
def generate(out):
 import torch
 if sys.byteorder!='little':raise ValueError('Little-endian required')
 root=Path(out);root.mkdir(parents=True,exist_ok=False);graphs=[]
 for g in plan()['graphs']:
  operands={}
  for name,shape in operand_shapes(g).items():
   seed=int.from_bytes(hashlib.sha256(f'61107|S4|{g["id"]}|{name}'.encode()).digest()[:8],'little')%(2**63-1)
   v=torch.randn(shape,generator=torch.Generator().manual_seed(seed))*0.1
   if name.endswith('/running_var'):v=v.abs()+.8
   if name.endswith('/weight') and len(shape)==1:v=v+1
   operands[name]=record_tensor(torch,v,root/g['id']/(name+'.bin'),root)
  graphs.append({'graph':g,'operands':operands})
 m={'version':BASE_VERSION,'plan_sha256':sha(PLAN),'graphs':graphs,'weights_kind':'initialized_performance_only','role':'shared_external_input_weights_and_state','criteria':CRITERIA}
 write(root/'BASE.json',m);return m

def verify_base(path):
 path=Path(path).resolve();root=path.parent;m=read(path)
 if m['version']!=BASE_VERSION or m['plan_sha256']!=sha(PLAN) or m['criteria']!=CRITERIA:raise ValueError('Base contract changed')
 if [e['graph'] for e in m['graphs']]!=plan()['graphs']:raise ValueError('Graph definitions changed')
 for e in m['graphs']:
  specs=operand_shapes(e['graph'])
  if set(e['operands'])!=set(specs):raise ValueError('Missing/extra base operands')
  for key,a in e['operands'].items():
   if a['shape']!=specs[key]:raise ValueError('Wrong operand shape')
   check_record(root,a)
 return m

def op(torch,n,inputs,b):
 import torch.nn.functional as F
 p=n['parameters'];prefix='node/'+n['id']+'/'
 if n['op']=='conv':return F.conv2d(inputs[0],b[prefix+'weight'],b.get(prefix+'bias'),p['stride'],p['padding'],p['dilation'],p['groups'])
 if n['op']=='batchnorm_eval':return F.batch_norm(inputs[0],b[prefix+'running_mean'],b[prefix+'running_var'],b[prefix+'weight'],b[prefix+'bias'],False,p['momentum'],p['eps'])
 if n['op']=='relu':return F.relu(inputs[0],inplace=False)
 if n['op']=='add':return torch.add(inputs[0],inputs[1],alpha=p['alpha'])
 raise ValueError('Unsupported context node '+n['op'])

def execute(torch,g,b,independent=None,selected=None):
 wires={k.split('/',1)[1]:v for k,v in b.items() if k.startswith('external/')};outs=[];used={}
 for n in g['nodes']:
  if selected is not None and n['id']!=selected:continue
  inputs=independent[n['id']] if independent is not None else [wires[x] for x in n['input_edges']]
  used[n['id']]=inputs;value=op(torch,n,inputs,b);wires[n['id']]=value;outs.append(value)
 return outs,used

def derive(base,out,expect_gpu,device='cuda'):
 import torch
 from o4_iteration import common as bootstrap
 from inference_pilot.benchmark import precision_policy
 torch.set_num_threads(1);policy=precision_policy(torch)
 base=Path(base).resolve();bm=verify_base(base);root=Path(out);root.mkdir(parents=True,exist_ok=False)
 shutil.copytree(base.parent,root/'base')
 if device=='cuda':
  if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('Exactly one CUDA GPU required')
  name=torch.cuda.get_device_properties(0).name
  if expect_gpu.lower() not in name.lower():raise ValueError('GPU mismatch')
 else:name='CPU_FIXTURE_NOT_GPU_MEASUREMENT'
 entries=[]
 with torch.inference_mode():
  for e in bm['graphs']:
   g=e['graph'];b={n:load(torch,base.parent,a,device) for n,a in e['operands'].items()};outputs,inputs=execute(torch,g,b)
   nodes={}
   for n,value in zip(g['nodes'],outputs):
    prefix=root/'derived'/g['id']/n['id'];nodes[n['id']]={'inputs':[record_tensor(torch,v,prefix/f'input_{i}.bin',root) for i,v in enumerate(inputs[n['id']])],'output':record_tensor(torch,value,prefix/'output.bin',root),'input_edges':n['input_edges']}
   entries.append({'graph_id':g['id'],'nodes':nodes})
 m={'version':VERSION,'plan_sha256':sha(PLAN),'base_manifest':'base/BASE.json','base_manifest_sha256':sha(base),
    'derivation':{'device':device,'gpu':name,'torch':str(torch.__version__),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'precision_policy':policy,'timing_role':'outside capture and GI','same_node_inputs':'exact native D FP16 tensors; local-device derived, not assumed equal across GPUs'},
    'graphs':entries,'configurations':[{'configuration':w} for w in plan()['work_items']],'criteria':CRITERIA,'strict_reference_verified':False}
 write(root/'ASSETS.json',m);verify(root/'ASSETS.json');return m

def verify(path):
 path=Path(path).resolve();root=path.parent;m=read(path)
 if m['version']!=VERSION or m['plan_sha256']!=sha(PLAN) or m['criteria']!=CRITERIA:raise ValueError('Local context asset contract changed')
 base=(root/m['base_manifest']).resolve()
 if root not in base.parents or sha(base)!=m['base_manifest_sha256']:raise ValueError('Base manifest changed/path escape')
 bm=verify_base(base)
 if [e['configuration'] for e in m['configurations']]!=plan()['work_items']:raise ValueError('Work list changed')
 if [e['graph_id'] for e in m['graphs']]!=[g['id'] for g in plan()['graphs']]:raise ValueError('Missing/extra derived graph')
 count=0
 for g,e in zip(plan()['graphs'],m['graphs']):
  if set(e['nodes'])!={n['id'] for n in g['nodes']}:raise ValueError('Wrong position set')
  for n in g['nodes']:
   d=e['nodes'][n['id']]
   if d['input_edges']!=n['input_edges'] or [a['shape'] for a in d['inputs']]!=n['input_shapes'] or d['output']['shape']!=n['output_shape']:raise ValueError('Derived position contract differs')
   for a in d['inputs']+[d['output']]:check_record(root,a);count+=1
 return {'configurations':len(m['configurations']),'derived_tensor_files':count,'base_operand_files':sum(len(e['operands']) for e in bm['graphs']),'assets_manifest_sha256':sha(path),'base_manifest_sha256':sha(base)}

class Adapter:
 def __init__(self,torch,manifest,cid,device='cuda'):
  verify(manifest);self.torch=torch;self.root=Path(manifest).resolve().parent;self.m=read(manifest);self.work=work(cid);self.device=device
  if device=='cuda':
   env=self.m['derivation']
   if env['device']!='cuda' or env['gpu']!=torch.cuda.get_device_properties(0).name or env['torch']!=str(torch.__version__) or env['cuda']!=torch.version.cuda or env['cudnn']!=torch.backends.cudnn.version():raise ValueError('Local derived inputs require the same device/toolchain; derive a new version on this node')
  bm=verify_base(self.root/self.m['base_manifest']);e=next(e for e in bm['graphs'] if e['graph']['id']==self.work['graph_id']);self.graph=e['graph'];base=(self.root/self.m['base_manifest']).parent
  self.b={n:load(torch,base,a,device) for n,a in e['operands'].items()};self.cpu={n:load(torch,base,a,'cpu',True) for n,a in e['operands'].items()}
  local=next(e for e in self.m['graphs'] if e['graph_id']==self.graph['id']);self.inputs={n:[load(torch,self.root,a,device) for a in d['inputs']] for n,d in local['nodes'].items()};self.gold={n:load(torch,self.root,d['output'],device) for n,d in local['nodes'].items()}
  self.retained=[];self.used={};self.metadata={'configuration':self.work,'graph':self.graph,'assets_manifest_sha256':sha(Path(manifest)),'input_provenance':local,'output_lifetime':'all node outputs retained until next prepare outside timing','metadata_role':'evaluator_only_never_predictor'}
 def prepare(self):self.retained=[];self.used={}
 def forward(self):
  kind=self.work['kind'];ind=None if kind=='dependent_graph' else self.inputs;selected=self.work.get('node_id') if kind=='isolated_position' else None
  self.retained,self.used=execute(self.torch,self.graph,self.b,ind,selected);return self.retained[-1]
 def correctness(self):
  t=self.torch;state={n:v.clone() for n,v in self.b.items() if n.endswith('/running_mean') or n.endswith('/running_var')}
  self.prepare();self.forward()
  if self.device=='cuda':t.cuda.synchronize()
  ind=None if self.work['kind']=='dependent_graph' else {n:[v.float().cpu() for v in values] for n,values in self.inputs.items()};selected=self.work.get('node_id') if self.work['kind']=='isolated_position' else None
  expected,_=execute(t,self.graph,self.cpu,ind,selected);nodes=[n for n in self.graph['nodes'] if selected is None or n['id']==selected];checks=[]
  for n,a,ref in zip(nodes,self.retained,expected):
   v=a.float().cpu();delta=v-ref;rel=float(t.linalg.vector_norm(delta)/t.linalg.vector_norm(ref).clamp_min(1e-12));scaled=float(delta.abs().max()/ref.abs().max().clamp_min(1));native=t.equal(a,self.gold[n['id']]);inputs_exact=all(t.equal(a,b) for a,b in zip(self.used[n['id']],self.inputs[n['id']]))
   passed=list(a.shape)==n['output_shape'] and bool(t.isfinite(v).all()) and rel<=CRITERIA['relative_l2_max'] and scaled<=CRITERIA['max_abs_over_scale_max'] and native and inputs_exact
   checks.append({'node_id':n['id'],'passed':passed,'relative_l2':rel,'max_abs_over_scale':scaled,'native_D_output_exact':native,'native_D_inputs_exact':inputs_exact})
  state_ok=all(t.equal(v,self.b[n]) for n,v in state.items());result={'passed':state_ok and all(c['passed'] for c in checks),'relative_l2':max(c['relative_l2'] for c in checks),'max_abs_over_scale':max(c['max_abs_over_scale'] for c in checks),'state_unchanged':state_ok,'positions':checks,'criteria':CRITERIA}
  if not result['passed']:raise RuntimeError('Context numerical/edge parity failed: '+str(result))
  self.prepare();return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['generate-base','verify-base','derive','verify']);p.add_argument('--path',required=True);p.add_argument('--base');p.add_argument('--expect-gpu');a=p.parse_args()
 value=generate(a.path) if a.action=='generate-base' else verify_base(a.path) if a.action=='verify-base' else derive(a.base,a.path,a.expect_gpu) if a.action=='derive' else verify(a.path)
 print(json.dumps(value,indent=2))
