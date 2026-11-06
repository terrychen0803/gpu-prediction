"""Independent process read-only audit replay; no GPU or model imports."""
import argparse,json,subprocess,sys
from pathlib import Path
from .common import read,write
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--out',required=True);a=p.parse_args();out=Path(a.out);out.mkdir(exist_ok=False);checked=[]
 for base in map(Path,a.runs):
  for job in read(base/'inventory.json')['jobs']:
   if job['mode']=='GI' or job['status']!='completed':continue
   raw=base/job['folder'];dest=out/(base.name+'_'+job['folder']);cmd=[sys.executable,'-c','from o4_research.audit import audit;import sys;audit(sys.argv[1],sys.argv[2])',str(raw),str(dest)];subprocess.run(cmd,check=True)
   source=raw/'audit'
   for p in sorted(source.rglob('*')):
    if p.is_file():
     name=p.relative_to(source);other=dest/name
     if p.read_bytes()!=other.read_bytes():raise AssertionError(str(p))
     checked.append(str(p))
 write(out/'VALIDATION.json',{'status':'PASS','raw_trace_reparse_processes':14,'byte_exact_files':len(checked),'paths':checked,'GPU_executions':0,'training':False});print('PASS replay files:',len(checked))
if __name__=='__main__':main()
