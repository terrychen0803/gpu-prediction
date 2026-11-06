from pathlib import Path
import csv,datetime,hashlib,json,os,signal,subprocess,time
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def csvwrite(p,rows):
 rows=list(rows)
 if not rows:return
 with Path(p).open('w',newline='') as f:
  keys=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in rows:w.writerow({k:json.dumps(r.get(k)) if isinstance(r.get(k),(dict,list,tuple)) else r.get(k) for k in keys})
def command(argv,folder,stem,timeout=300,env=None,cwd=None):
 folder=Path(folder);start=now();tick=time.perf_counter_ns();record={'argv':[str(x) for x in argv],'start_utc':start,'start_monotonic_ns':tick}
 with (folder/(stem+'.log')).open('w') as f:
  proc=subprocess.Popen([str(x) for x in argv],cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  try:rc=proc.wait(timeout=timeout)
  except BaseException:
   try:os.killpg(proc.pid,signal.SIGTERM)
   except ProcessLookupError:pass
   try:proc.wait(timeout=10)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   record.update(status='interrupted_or_timeout',returncode=proc.returncode,end_utc=now(),end_monotonic_ns=time.perf_counter_ns());write(folder/(stem+'_command.json'),record);raise
 record.update(status='completed' if rc==0 else 'failed',returncode=rc,end_utc=now(),end_monotonic_ns=time.perf_counter_ns());write(folder/(stem+'_command.json'),record)
 if rc:raise RuntimeError(f'{stem} exited{rc}; see {folder}/{stem}.log')
 return record
def gpu_guard(folder,expect='GB10'):
 """Allow only the same-user RustDesk context; never terminate any context."""
 folder=Path(folder);q=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True);p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,capture_output=True);owners=[];blocked=[]
 if q.returncode or p.returncode:blocked.append('GPU discovery failed')
 lines=[x for x in q.stdout.splitlines() if x.strip()]
 if len(lines)!=1 or expect.lower() not in q.stdout.lower():blocked.append('Expected exactly one matching GPU')
 for row in csv.reader(p.stdout.splitlines()):
  if not row:continue
  try:
   pid=int(row[0]);uid=Path(f'/proc/{pid}').stat().st_uid;comm=Path(f'/proc/{pid}/comm').read_text().strip();allowed=uid==os.getuid() and comm.lower()=='rustdesk';owners.append({'pid':pid,'uid':uid,'comm':comm,'allowed_background':allowed})
   if not allowed:blocked.append('Other GPU workload present')
  except (ValueError,OSError):blocked.append('Cannot establish GPU process ownership')
 try:
  utilization=int(lines[0].split(',')[-1].strip().replace('%',''))
  if utilization>5:blocked.append('GPU utilization exceeds5%; defer diagnostic')
 except (IndexError,ValueError):blocked.append('GPU utilization unavailable')
 write(folder/'gpu_guard.json',{'utc':now(),'gpu_query':q.stdout,'process_query':p.stdout,'owners':owners,'blocked':blocked,'services_unchanged':True,'performance_benchmark':False})
 if blocked:raise RuntimeError('; '.join(blocked))
