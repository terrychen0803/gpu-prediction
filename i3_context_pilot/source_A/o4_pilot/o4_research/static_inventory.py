"""CPU-only module inventory: architecture evidence, not runtime op labels."""
import argparse,collections,gc
from pathlib import Path
from .common import *
from .adapters import Adapter,SPECS
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--assets-manifest',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);import torch
 torch.set_num_threads(1);rows=[];summaries=[]
 for name in SPECS:
  if name=='QD-A':continue
  adapter=Adapter(torch,name,a.assets_manifest);types=collections.Counter();groups=[]
  for path,module in adapter.model.named_modules():
   kind=type(module).__module__+'.'+type(module).__name__;types[kind]+=1;row={'workload':name,'module_path':path,'class':kind,'parameter_shapes':{k:list(v.shape) for k,v in module.named_parameters(recurse=False)},'evaluator_only':True}
   for key in ('in_channels','out_channels','kernel_size','stride','padding','dilation','groups','eps'):
    if hasattr(module,key):
     value=getattr(module,key);row[key]=value.detach().cpu().tolist() if isinstance(value,torch.Tensor) else value
   if isinstance(module,torch.nn.Conv2d):row['bias']=module.bias is not None;row['depthwise']=module.groups==module.in_channels and module.groups>1;groups.append(row)
   rows.append(row)
  summaries.append({'workload':name,'module_types':dict(types),'grouped_conv_modules':sum(x['groups']>1 for x in groups),'depthwise_conv_modules':sum(x['depthwise'] for x in groups),'runtime_ATen_mapping_proven':False,'weights_kind':adapter.metadata['weights_kind'],'strict_verified':0});del adapter;gc.collect()
 q=next(x for x in summaries if x['workload']=='QP-A');summaries.append({**q,'workload':'QD-A','architecture_shared_with':'QP-A','forward_and_cache_policy_separate':True});write(out/'STATIC_MODULES_EVALUATOR_ONLY.json',rows);write(out/'SUMMARY.json',summaries);print([(x['workload'],x['depthwise_conv_modules']) for x in summaries])
if __name__=='__main__':main()
