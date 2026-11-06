"""Download public official bytes into a new directory, freeze hashes, no GPU."""
import argparse,hashlib,json,urllib.request
from pathlib import Path
from .common import *
def fetch(url,p):
 p=Path(p)
 if p.exists():raise FileExistsError(p)
 tmp=p.with_suffix(p.suffix+'.partial')
 with urllib.request.urlopen(url,timeout=60) as response,tmp.open('xb') as f:
  while b:=response.read(8<<20):f.write(b)
 tmp.rename(p);return {'file':p.name,'sha256':sha(p),'bytes':p.stat().st_size,'source_url':url}
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);entries={};failures={}
 sources={'mobilenet_v3_large-5c1a4163.pth':'https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth','swin_t-704ceda3.pth':'https://download.pytorch.org/models/swin_t-704ceda3.pth','yolo26n.pt':'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt'}
 for name,url in sources.items():
  try:entries[name]=fetch(url,out/name);print('downloaded',name,entries[name]['bytes'],flush=True)
  except Exception as e:failures[name]=type(e).__name__+': '+str(e)
  write(out/'DOWNLOAD_MANIFEST.json',{'entries':entries,'failures':failures})
 try:
  with urllib.request.urlopen('https://huggingface.co/api/models/Qwen/Qwen3-0.6B',timeout=30) as f:metadata=json.load(f)
  ref=metadata['sha'];q=out/'qwen3_0_6b';q.mkdir();write(q/'SOURCE_REVISION.json',{'repo':'Qwen/Qwen3-0.6B','revision':ref})
  for name in ['config.json','generation_config.json','model.safetensors']:
   entries['qwen/'+name]=fetch('https://huggingface.co/Qwen/Qwen3-0.6B/resolve/'+ref+'/'+name,q/name);print('downloaded qwen',name,entries['qwen/'+name]['bytes'],flush=True)
 except Exception as e:failures['qwen']=type(e).__name__+': '+str(e)
 write(out/'DOWNLOAD_MANIFEST.json',{'entries':entries,'failures':failures});print('Failures:',failures)
if __name__=='__main__':main()
