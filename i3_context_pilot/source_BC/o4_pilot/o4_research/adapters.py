"""Exact seven A workloads. Reference/shape metadata is evaluator-only."""
import copy,inspect,hashlib
from pathlib import Path
from .common import read,sha
SPECS={
 'RN-A':{'model':'torchvision.resnet50','weights':'IMAGENET1K_V2','shape':[1,3,224,224],'output':'1000 logits'},
 'MV-A':{'model':'torchvision.mobilenet_v3_large','weights':'IMAGENET1K_V2','shape':[1,3,224,224],'output':'1000 logits'},
 'SW-A':{'model':'torchvision.swin_t','weights':'IMAGENET1K_V1','shape':[1,3,224,224],'output':'1000 logits'},
 'UN-A':{'model':'unet2d_v1','weights':'initialized_performance_only','shape':[1,1,256,256],'output':'2 class spatial logits'},
 'YO-A':{'model':'ultralytics.yolo26n.detect','weights':'yolo26n.pt','shape':[1,3,640,640],'output':'native one2one decoded/topk tensor, no external NMS'},
 'QP-A':{'model':'Qwen/Qwen3-0.6B','weights':'pretrained','shape':[1,128],'output':'last position logits + fresh KV length128'},
 'QD-A':{'model':'Qwen/Qwen3-0.6B','weights':'pretrained','shape':[1,1],'cache_before':128,'output':'last position logits + KV length129'},
}
def require_a(name):
 if name not in SPECS:raise ValueError('Only seven A configurations supported; never substitute a model/shape')
 return SPECS[name]
def make_unet(torch):
 nn=torch.nn
 class Pair(nn.Module):
  def __init__(self,a,b):super().__init__();self.net=nn.Sequential(nn.Conv2d(a,b,3,padding=1,bias=True),nn.ReLU(),nn.Conv2d(b,b,3,padding=1,bias=True),nn.ReLU())
  def forward(self,x):return self.net(x)
 class UNet(nn.Module):
  def __init__(self):
   super().__init__();width=[32,64,128,256,512];self.enc=nn.ModuleList([Pair(1,32)]+[Pair(width[i-1],width[i]) for i in range(1,5)]);self.pool=nn.MaxPool2d(2);self.dec=nn.ModuleList([Pair(512+256,256),Pair(256+128,128),Pair(128+64,64),Pair(64+32,32)]);self.head=nn.Conv2d(32,2,1,bias=True)
  def forward(self,x):
   skip=[]
   for i,block in enumerate(self.enc):
    if i:x=self.pool(x)
    x=block(x);skip.append(x)
   for block,s in zip(self.dec,reversed(skip[:-1])):x=block(torch.cat([torch.nn.functional.interpolate(x,scale_factor=2,mode='bilinear',align_corners=False),s],dim=1))
   return self.head(x)
 return UNet()
class Adapter:
 def __init__(self,torch,name,manifest_path):
  self.torch=torch;self.name=name;spec=require_a(name);manifest=read(manifest_path);entry=manifest['workloads'][name];self.entry=entry;self.cache=None;self.last_cache_before=None;self.last_cache_after=None;self.cpu_reference=None
  if entry['status']!='ready':raise RuntimeError('Pending assets/dependency: '+str(entry))
  root=Path(manifest_path).resolve().parent
  def check(key):
   item=entry[key];p=root/item['path'];assert sha(p)==item['sha256'],f'Asset changed: {key}';return p
  weight=check('weights');input_path=check('input');self.input_path=input_path
  if name in ('RN-A','MV-A','SW-A'):
   import torchvision.models as tv
   constructors={'RN-A':tv.resnet50,'MV-A':tv.mobilenet_v3_large,'SW-A':tv.swin_t};self.model=constructors[name](weights=None);self.model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
  elif name=='UN-A':self.model=make_unet(torch);self.model.load_state_dict(torch.load(weight,map_location='cpu',weights_only=True),strict=True)
  elif name=='YO-A':
   from ultralytics import YOLO
   self.model=YOLO(str(weight),task='detect').model;head=self.model.model[-1]
   if not getattr(head,'end2end',False) or not hasattr(head,'one2one'):raise RuntimeError('Installed YOLO does not expose required native one2one head')
   head.export=True;head.format=None
   # No predict(), preprocessing, NMS, graph/compile, backbone fusion, or branch surgery.
  else:
   from transformers import AutoModelForCausalLM
   from transformers.cache_utils import DynamicCache
   self.DynamicCache=DynamicCache;config=check('config')
   self.model=AutoModelForCausalLM.from_pretrained(str(weight.parent),local_files_only=True,trust_remote_code=False,torch_dtype=torch.float32,attn_implementation='eager');assert self.model.config.model_type=='qwen3' and self.model.config.num_hidden_layers==28
   tokens=read(input_path);self.prefix=torch.tensor(tokens['prefix_ids'],dtype=torch.long).reshape(1,128);self.decode=torch.tensor([[tokens['decode_id']]],dtype=torch.long)
  self.model.eval();self.device='cpu'
  if name not in ('QP-A','QD-A'):
   raw=input_path.read_bytes();assert len(raw)==2*__import__('math').prod(spec['shape']);self.x=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(spec['shape']).float().clone()
  self.metadata={'workload':name,**spec,'weights_kind':entry['weights_kind'],'weights_sha256':entry['weights']['sha256'],'input_sha256':entry['input']['sha256'],'scope_version':'o4_A_adapter_v1_20261106','dtype':'float16','eager':True,'autocast':False,'compile':False,'cuda_graph':False,'predictor_eligible_metadata':False,'adapter_source_sha256':sha(__file__),'native_model_source_sha256':sha(inspect.getfile(type(self.model))),'forward_scope':spec['output'],'reference_status':'development_unverified'}
  if name=='YO-A':self.metadata.update(native_auxiliary_one2many_computed=True,head_export_tensor=True,external_NMS=False,backbone_fusion=False,correctness_rule='exact parity to native same-precision forward + fixed tensor schema; FP32 topk differences diagnostic only')
  if name in ('QP-A','QD-A'):self.metadata.update(attention_backend='eager',cache='DynamicCache(config=model.config)',logits_to_keep=1,cache_rebuild='fresh prefill outside every decode interval',prefix_token_count=128,full_generation=False)
 def prepare(self):
  t=self.torch
  if self.name not in ('QP-A','QD-A'):return
  self.cache=self.DynamicCache(config=self.model.config)
  if self.name=='QD-A':
   ids=self.prefix.to(self.device);self.model(input_ids=ids,attention_mask=t.ones_like(ids),position_ids=t.arange(128,device=self.device).reshape(1,128),cache_position=t.arange(128,device=self.device),past_key_values=self.cache,use_cache=True,logits_to_keep=1)
   if self.cache.get_seq_length()!=128:raise RuntimeError('Decode cache preparation wrong length')
  before=0 if self.name=='QP-A' else 128;self.ready_ids=(self.prefix if self.name=='QP-A' else self.decode).to(self.device);length=self.ready_ids.shape[1];self.ready_position=t.arange(before,before+length,device=self.device);self.ready_mask=t.ones((1,before+length),dtype=t.long,device=self.device)
 def forward(self):
  if self.name not in ('QP-A','QD-A'):return self.model(self.x)
  t=self.torch;before=self.cache.get_seq_length();expected=0 if self.name=='QP-A' else 128
  if before!=expected:raise RuntimeError('Cache reuse/growth across iterations is forbidden')
  ids=self.ready_ids;length=ids.shape[1];position=self.ready_position
  output=self.model(input_ids=ids,attention_mask=self.ready_mask,position_ids=position.reshape(1,length),cache_position=position,past_key_values=self.cache,use_cache=True,logits_to_keep=1)
  self.last_cache_before=before;self.last_cache_after=self.cache.get_seq_length();return output.logits
 def to_cuda(self):
  self.cache=None;self.model=self.model.cuda().half().eval();self.device='cuda'
  if hasattr(self,'x'):self.x=self.x.cuda().half()
  if hasattr(self,'prefix'):self.prefix=self.prefix.cuda();self.decode=self.decode.cuda()
 def correctness(self):
  t=self.torch
  self.prepare();cpu=self.forward().detach().float();self.cache=None
  self.to_cuda();self.prepare();actual=self.forward();t.cuda.synchronize();host=actual.float().cpu();self.prepare();native=self.forward();t.cuda.synchronize()
  # Separate native-route parity and numerical precision diagnostics.
  exact=bool(t.equal(actual,native));finite=bool(t.isfinite(host).all());shape=list(host.shape);delta=host-cpu
  l2=float(t.linalg.vector_norm(delta)/t.linalg.vector_norm(cpu).clamp_min(1e-12));scale=float(delta.abs().max()/cpu.abs().max().clamp_min(1));expected=[1,1000] if self.name in ('RN-A','MV-A','SW-A') else ([1,2,256,256] if self.name=='UN-A' else ([1,300,6] if self.name=='YO-A' else [1,1,self.model.config.vocab_size]))
  passed=finite and exact and shape==expected and (self.name=='YO-A' or (l2<=.02 and scale<=.02))
  result={'passed':passed,'finite':finite,'native_route_same_precision_exact':exact,'CPU_FP32_relative_l2':l2,'CPU_FP32_max_abs_over_scale':scale,'precision_gate': 'diagnostic_only_discrete_topk' if self.name=='YO-A' else 'relative_l2<=0.02 and scaled_max<=0.02','output_shape':shape,'expected_shape':expected,'cache_before':self.last_cache_before,'cache_after':self.last_cache_after,'task_accuracy_validated':False}
  if not passed:raise RuntimeError('Adapter correctness failed: '+str(result))
  return result
