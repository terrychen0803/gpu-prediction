from __future__ import annotations
from .common import ROOT,read,sha
class Adapter:
 def __init__(self,torch,req):
  self.torch=torch;self.req=req;self.model=None
  if req['workload']=='RN-A':
   import torchvision
   from torchvision.models import resnet50
   assets=req['assets'];w=ROOT/assets['weights']['path'];side=read(ROOT/assets['sidecar']['path'])
   if side['sha256']!=sha(w) or side['weights']!='IMAGENET1K_V2':raise ValueError('ResNet weight identity mismatch')
   self.model=resnet50(weights=None).eval()
   self.model.load_state_dict(torch.load(w,map_location='cpu',weights_only=True),strict=True)
   raw=(ROOT/assets['input']['path']).read_bytes()[:1*3*224*224*2]
   x=torch.frombuffer(bytearray(raw),dtype=torch.float16).reshape(1,3,224,224).clone()
   self.reference=self.model(x.float()).detach().float()
   self.model=self.model.cuda().half().eval();self.x=x.cuda();self.output_shape=[1,1000]
   self.metadata={'model_signature':'torchvision.resnet50.IMAGENET1K_V2.B1.224.fp16.eager.native_forward.v1','weights_kind':'pretrained','input_prefix_sha256':__import__('hashlib').sha256(raw).hexdigest(),'torchvision':torchvision.__version__,'forward_scope':'native device tensor -> 1000 logits; no preprocessing/postprocessing'}
  else:
   from o1_operator.worker import load_operation
   from o1_operator.full_matrix import PROTOCOL
   from o1_operator.full_matrix import BY_ID
   files={k:{**v,'path':str(ROOT/v['path'])} for k,v in req['assets'].items()}
   self.op,self.reference,self.operands=load_operation(torch,{'protocol':PROTOCOL,'config_id':req['workload'],'inputs':files})
   self.output_shape=list(self.reference.shape)
   self.metadata={'model_signature':BY_ID[req['workload']],'reference':'CPU FP32 op then FP16 cast','weights_kind':'not_applicable'}
 def forward(self):return self.model(self.x) if self.model is not None else self.op()
 def correctness(self):
  t=self.torch;out=self.forward();t.cuda.synchronize();actual=out.float().cpu();expected=self.reference.float()
  if list(actual.shape)!=self.output_shape or not bool(t.isfinite(actual).all()) or not bool(t.isfinite(expected).all()):raise RuntimeError('Output shape/finite correctness failed')
  delta=actual-expected;rel=float(t.linalg.vector_norm(delta)/t.linalg.vector_norm(expected).clamp_min(1e-12));scaled=float(delta.abs().max()/expected.abs().max().clamp_min(1))
  p=self.req['protocol']
  passed=(rel<=p['resnet_relative_l2_max'] and scaled<=p['resnet_max_abs_over_scale_max']) if self.model is not None else bool(t.allclose(actual,expected,rtol=p['correctness_rtol'],atol=p['correctness_atol']))
  result={'passed':passed,'relative_l2':rel,'max_abs_over_scale':scaled,'max_absolute_error':float(delta.abs().max()),'reference':'CPU FP32 native forward' if self.model is not None else 'CPU FP32 op cast FP16','output_shape':list(out.shape)}
  if not passed:raise RuntimeError('FP32 reference correctness failed: '+str(result))
  return result,out.detach()
def dispatch_inventory(torch):
 from torch.utils._python_dispatch import TorchDispatchMode
 from torch.utils._pytree import tree_flatten
 class Inventory(TorchDispatchMode):
  def __init__(self):super().__init__();self.rows=[];self.active=False;self.stage='';self.iteration=-1;self.index=0
  def begin(self,stage,index):self.active=True;self.stage=stage;self.iteration=index;self.index=0
  def end(self):self.active=False
  def __torch_dispatch__(self,func,types,args=(),kwargs=None):
   if not self.active:return func(*args,**(kwargs or {}))
   idx=self.index;self.index+=1;label=f'o4.op/{self.stage}/{self.iteration:06d}/{idx:04d}'
   tensors=[x for x in tree_flatten((args,kwargs or {}))[0] if isinstance(x,torch.Tensor)]
   metadata={'scope':label,'stage':self.stage,'iteration':self.iteration,'op_index':idx,'aten':str(func),'input_shapes':[list(x.shape) for x in tensors],'input_dtypes':[str(x.dtype) for x in tensors]}
   torch.cuda.nvtx.range_push(label)
   try:out=func(*args,**(kwargs or {}))
   finally:torch.cuda.nvtx.range_pop()
   self.rows.append(metadata);return out
 return Inventory()
