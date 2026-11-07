from __future__ import annotations
import csv, hashlib, json, subprocess, sys
from pathlib import Path
from datetime import datetime, timezone
ROOT=Path(__file__).resolve().parent.parent
LEGACY=ROOT/'vendor/legacy'
if str(LEGACY) not in sys.path: sys.path.insert(0,str(LEGACY))
def now(): return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''): h.update(block)
 return h.hexdigest()
def read(path): return json.loads(Path(path).read_text())
def write(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvwrite(path,rows,fields=None):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def jsonl(path,rows):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w') as f:
  for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
def code_manifest():
 paths=list((ROOT/'o4_iteration').glob('*.py'))+list((LEGACY/'o1_operator').glob('*.py'))+list((LEGACY/'inference_pilot').glob('*.py'))
 return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}
def snapshot():
 result={}
 for key,cmd in {'gpu':['nvidia-smi','--query-gpu=name,driver_version,utilization.gpu,pstate,temperature.gpu,power.draw,clocks.sm,memory.used','--format=csv,noheader'],'processes':['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader']}.items():
  p=subprocess.run(cmd,capture_output=True,text=True);result[key]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
 return result
