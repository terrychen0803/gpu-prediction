"""Freeze shared assets once; never regenerate them independently on nodes."""
import argparse,hashlib,importlib.util,shutil
from pathlib import Path
from .common import *
from .adapters import SPECS,make_unet
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--downloads',required=True);p.add_argument('--original-root',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);downloads=Path(a.downloads);old=Path(a.original_root);import torch
 def record(p):return {'path':str(p.relative_to(out)),'sha256':sha(p),'bytes':p.stat().st_size}
 def copy(src,name):dest=out/name;shutil.copy2(src,dest);return record(dest)
 image=(old/'assets/e1_shared_fp16_b32_seed42_20261020.bin').read_bytes()[:1*3*224*224*2];image_path=out/'image224_shared_fp16.bin';image_path.write_bytes(image)
 items={};input224=record(image_path)
 for name,weight in [('RN-A',old/'assets/resnet50_imagenet1k_v2.pth'),('MV-A',downloads/'mobilenet_v3_large-5c1a4163.pth'),('SW-A',downloads/'swin_t-704ceda3.pth'),('YO-A',downloads/'yolo26n.pt')]:
  if not weight.exists():items[name]={'status':'pending_assets','required_file':weight.name};continue
  w=copy(weight,weight.name);inp=input224
  if name=='YO-A':
   x=((torch.arange(1*3*640*640,dtype=torch.int32)%256).float()/255).half();path=out/'image640_shared_fp16.bin';path.write_bytes(x.numpy().tobytes());inp=record(path)
  items[name]={'status':'ready','weights':w,'weights_kind':'pretrained','input':inp,'spec':SPECS[name]}
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(61106);model=make_unet(torch);path=out/'unet2d_v1_initialized_seed61106.pth';torch.save(model.state_dict(),path)
 x=((torch.arange(256*256,dtype=torch.int32)%251).float()/125-1).half();input_path=out/'unet256_shared_fp16.bin';input_path.write_bytes(x.numpy().tobytes());items['UN-A']={'status':'ready','weights':record(path),'weights_kind':'initialized_performance_only','input':record(input_path),'spec':SPECS['UN-A'],'task_segmentation_quality_validated':False}
 qsource=downloads/'qwen3_0_6b';qout=out/'qwen3_0_6b';qout.mkdir()
 if all((qsource/n).exists() for n in ('model.safetensors','config.json')):
  for n in ('model.safetensors','config.json','generation_config.json','SOURCE_REVISION.json'):
   if (qsource/n).exists():shutil.copy2(qsource/n,qout/n)
  path=out/'qwen128_tokens.json';write(path,{'prefix_ids':[42+i%100 for i in range(128)],'decode_id':17,'tokenizer_used':False,'prompt_kind':'fixed synthetic token IDs, no chat/generation','cache_before_decode':128});tokens=record(path)
  for name in ('QP-A','QD-A'):items[name]={'status':'ready','weights':record(qout/'model.safetensors'),'weights_kind':'pretrained','config':record(qout/'config.json'),'input':tokens,'spec':SPECS[name],'source_revision':read(qout/'SOURCE_REVISION.json')['revision'],'transformers_required':'4.57.1'}
 else:
  for name in ('QP-A','QD-A'):items[name]={'status':'pending_assets','reason':'Qwen3-0.6B exact checkpoint missing'}
 write(out/'ASSETS.json',{'version':'o4_A_assets_v1_20261106','workloads':items,'shared_bytes_required':True,'copy_not_regenerate_on_other_nodes':True,'weights_and_inputs_not_in_Git':True});print({k:v['status'] for k,v in items.items()})
if __name__=='__main__':main()
