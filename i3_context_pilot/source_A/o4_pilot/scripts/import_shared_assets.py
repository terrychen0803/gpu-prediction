"""Import exact shared assets without torch, downloads or GPU work."""
from pathlib import Path
import argparse,shutil,sys,tempfile,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from o4_iteration.common import ROOT,read,sha

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--from',dest='source',required=True);args=parser.parse_args();source=Path(args.source).resolve()
 manifest=read(ROOT/'manifests/assets.json');items=list(manifest['resnet'].values())
 for files in manifest['operators'].values():items.extend(files.values())
 pairs=[]
 for item in items:
  relative=Path(item['path']);assert relative.parts[0]=='assets' and '..' not in relative.parts
  src=source/Path(*relative.parts[1:]);dst=ROOT/relative
  if not src.is_file() or sha(src)!=item['sha256']:raise ValueError('Missing/corrupted shared asset: '+str(src))
  if dst.exists() and sha(dst)!=item['sha256']:raise ValueError('Refusing to overwrite different existing asset: '+str(dst))
  pairs.append((src,dst,item['sha256']))
 for src,dst,digest in pairs:
  if dst.exists():continue
  dst.parent.mkdir(parents=True,exist_ok=True)
  with tempfile.NamedTemporaryFile(dir=dst.parent,delete=False) as f:temp=Path(f.name)
  try:
   shutil.copyfile(src,temp)
   if sha(temp)!=digest:raise ValueError('Copy hash mismatch')
   if dst.exists():raise FileExistsError(dst)
   os.link(temp,dst)
  finally:temp.unlink(missing_ok=True)
 print('PASS:',len(pairs),'shared assets copied/verified; no CUDA process started')
if __name__=='__main__':main()
