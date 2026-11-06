"""Serial S3-lite batches. Preserve failures; only healthy smoke configs enter 40x3."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PYTHON='/home/good_egg/RTchen/O4/analysis_runs/o4_i1_reference_i2_adapters_20261106T091134Z/env/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def eligible(inv):
    if inv.get('status','').startswith('paused_'):return []
    result=[]
    for cid in inv['configs']:
        jobs=[j for j in inv['jobs'] if j['config_id']==cid]
        pairs=[p for p in inv.get('P_R_pairs',[]) if p['config_id']==cid]
        if len(jobs)!=3 or {(j.get('mode'),j.get('repeat')) for j in jobs}!={('GI',1),('P',1),('R',1)}:continue
        if any(j['status']!='completed' or j.get('monitor_review',True) for j in jobs):continue
        if len(pairs)!=1 or pairs[0].get('measured_exact')!=3 or not pairs[0].get('all_windows_exact'):continue
        result.append(cid)
    return result

def main():
    if (ROOT/'EXECUTION_STATUS.json').exists():raise RuntimeError('Existing execution state; preserve it. Resume needs reviewed new orchestration, not overwrite.')
    env=dict(os.environ,PATH='/home/good_egg/.local/bin:'+os.environ.get('PATH',''),PYTHONPATH=str(ROOT/'source')+os.pathsep+str(ROOT/'source/o4_pilot'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    scope=read(ROOT/'SCOPE.json');state={'status':'running','utc_start':now(),'batches':[],'GPU_execution':True,'models_trained':0,'strict_reference_verified':False,'formal_training_eligible':False,'test_model_scores_evaluated':False}
    write(ROOT/'EXECUTION_STATUS.json',state);partial=False
    try:
        for b in scope['batches']:
            entry={'batch_id':b['batch_id'],'split':b['split'],'configs':b['configs'],'stages':[]};state['batches'].append(entry)
            for kind in ['smoke','formal']:
                ids=b['configs'] if kind=='smoke' else eligible(read(ROOT/'collections'/(b['batch_id']+'_smoke')/'inventory.json'))
                excluded=sorted(set(b['configs'])-set(ids));entry[kind+'_excluded_configs']=excluded;partial|=bool(excluded)
                if not ids:entry['stages'].append({'kind':kind,'status':'not_run_no_smoke_eligible_configs'});write(ROOT/'EXECUTION_STATUS.json',state);continue
                folder=ROOT/'collections'/(b['batch_id']+'_'+kind)
                argv=[PYTHON,'-m','o4_i3_v2.collect','--kind',kind,'--expect-gpu','GB10','--assets-manifest',str(ROOT/'assets_S3/ASSETS.json'),'--config',*ids,'--out',str(folder)]
                record={'kind':kind,'status':'running','argv':argv,'utc_start':now(),'monotonic_start_ns':time.perf_counter_ns()};entry['stages'].append(record);write(ROOT/'EXECUTION_STATUS.json',state)
                print('START',b['batch_id'],kind,len(ids),'configs',now(),flush=True)
                with (ROOT/'logs'/(b['batch_id']+'_'+kind+'.log')).open('w') as log:
                    proc=subprocess.run(argv,cwd=ROOT/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
                record.update(returncode=proc.returncode,utc_end=now(),monotonic_end_ns=time.perf_counter_ns(),status='completed' if proc.returncode==0 else 'failed_or_partial');partial|=proc.returncode!=0
                write(ROOT/'EXECUTION_STATUS.json',state);inv=read(folder/'inventory.json')
                print('END',b['batch_id'],kind,proc.returncode,now(),flush=True)
                if inv['status'].startswith('paused_'):
                    state.update(status='paused_GPU_guard_or_foreign_work',utc_end=now());write(ROOT/'EXECUTION_STATUS.json',state);return 2
        state.update(status='finished_with_preserved_failures_or_exclusions' if partial else 'completed_all_preselected_batches',utc_end=now());write(ROOT/'EXECUTION_STATUS.json',state)
        return 1 if partial else 0
    except BaseException as e:
        state.update(status='orchestration_interrupted',error=f'{type(e).__name__}: {e}',utc_end=now());write(ROOT/'EXECUTION_STATUS.json',state);raise
if __name__=='__main__':raise SystemExit(main())
