"""Read-only low-frequency host telemetry; never kills external workloads."""
import csv,json,os,subprocess,threading
from pathlib import Path
from o4_research.common import read,now

class Monitor:
    def __init__(self,folder,period=5):
        self.folder=Path(folder);self.period=period;self.stop_event=threading.Event();self.thread=None
        self.rows=[];self.foreign=False;self.errors=[]
    def __enter__(self):
        self.thread=threading.Thread(target=self._run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop_event.set();self.thread.join(timeout=15)
        if self.thread.is_alive():self.errors.append('monitor failed to join within bound')
        return False
    def _run(self):
        with (self.folder/'host_monitor.jsonl').open('w') as log:
            while True:
                row={'utc':now(),'period_seconds':self.period,'processes':[],'errors':[]}
                try:
                    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,temperature.gpu,pstate,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True,timeout=8)
                    proc=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True,timeout=8)
                    row.update(gpu_returncode=gpu.returncode,gpu_stdout=gpu.stdout,gpu_stderr=gpu.stderr,process_returncode=proc.returncode,process_stdout=proc.stdout,process_stderr=proc.stderr)
                    expected=None
                    try:expected=read(self.folder/'environment.json')['pid']
                    except (FileNotFoundError,KeyError,json.JSONDecodeError):pass
                    if gpu.returncode or proc.returncode:row['errors'].append('nvidia-smi discovery failed')
                    for fields in csv.reader(proc.stdout.splitlines()):
                        if not fields:continue
                        try:
                            pid=int(fields[0]);uid=Path(f'/proc/{pid}').stat().st_uid;comm=Path(f'/proc/{pid}/comm').read_text().strip()
                            own=pid==expected;allowed=uid==os.getuid() and comm.lower()=='rustdesk'
                            foreign=not own and not allowed
                            row['processes'].append({'pid':pid,'uid':uid,'comm':comm,'own_worker':own,'allowed_background':allowed,'foreign':foreign})
                            self.foreign|=foreign
                        except (OSError,ValueError):row['errors'].append('process owner unavailable during sample')
                except Exception as e:row['errors'].append(f'{type(e).__name__}: {e}')
                self.errors.extend(row['errors']);self.rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if self.stop_event.wait(self.period):break
