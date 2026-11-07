"""Bounded, resumable config collector. Existing raw and completed cells never overwritten."""
import argparse,csv,hashlib,os,shutil,sys
from pathlib import Path
from o4_research.common import read,write,sha,now,command,gpu_guard
from .assets import verify
from .protocol import protocol,schedule,distribution,measured_values
from o4_i3_v2.monitor import Monitor

def code_index(root):
    return [{'path':str(f.relative_to(root)),'sha256':sha(f)} for f in sorted(Path(root).rglob('*'))
       if f.is_file() and '__pycache__' not in f.parts and f.suffix!='.pyc']

def verify_code(root,expected):
    if code_index(root)!=expected:raise ValueError('Source snapshot changed; do not continue')

def summarize(out,inv):
    from o4_research.audit import pair
    out=Path(out);p=inv['protocol'];groups=[];pairs=[]
    for cid in inv['configs']:
        for mode in p['modes']:
            selected=[j for j in inv['jobs'] if j['config_id']==cid and j['mode']==mode]
            med=[]
            for j in selected:
                if j['status']!='completed':continue
                folder=out/j['folder'];q=read(folder/'quality.json')
                if not q.get('correctness',{}).get('passed') or not q.get('warmup',{}).get('passed'):continue
                rows=list(csv.DictReader((folder/'iterations.csv').open()));med.append(distribution(measured_values(rows,p))['median_us'])
            complete=len(med)==p['independent_process_repeats'];spread=(max(med)-min(med))/__import__('statistics').median(med) if complete and len(med)>1 else None
            groups.append({'config_id':cid,'mode':mode,'completed_processes':len(med),'process_medians_us':med,
                'median_of_process_medians_us':__import__('statistics').median(med) if med else None,
                'spread_fraction':spread,'stability_status':'not_evaluated_smoke' if p['kind']=='smoke' else 'stable' if complete and spread<=p['repeat_spread_threshold'] else 'review_required',
                'target_kind':'iteration_device_event_us' if mode=='GI' else 'profiled_iteration_event_us'})
        for rep in range(1,p['independent_process_repeats']+1):
            by={j['mode']:j for j in inv['jobs'] if j['config_id']==cid and j['repeat']==rep and j['status']=='completed'}
            if not {'P','R'}<=set(by):continue
            dest=out/f'{cid}_r{rep}_pair'
            try:
                result=read(dest/'P_R_ALIGNMENT.json') if dest.exists() else pair(out/by['P']['folder']/'audit',out/by['R']['folder']/'audit',dest)
                measured=[r for r in result if r['scope'].startswith('o4.iter/measure/')]
                pairs.append({'config_id':cid,'repeat':rep,'windows':len(result),'measured_expected':p['measured_iterations'],
                    'measured_exact':sum(r['ordered_signature_exact'] for r in measured),'all_windows_exact':all(r['ordered_signature_exact'] for r in result)})
            except Exception as e:pairs.append({'config_id':cid,'repeat':rep,'error':str(e)})
    inv['mode_groups']=groups;inv['P_R_pairs']=pairs

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['smoke','formal'],required=True)
    ap.add_argument('--assets-manifest',required=True);ap.add_argument('--config',nargs='+',required=True)
    ap.add_argument('--expect-gpu',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--plan-only',action='store_true');ap.add_argument('--resume',action='store_true');a=ap.parse_args()
    p=protocol(a.kind);manifest=Path(a.assets_manifest).resolve();asset_check=verify(manifest)
    available={e['configuration']['id'] for e in read(manifest)['configurations']}
    if not set(a.config)<=available:raise ValueError('Requested configurations missing from assets')
    planned=schedule(a.config,p);out=Path(a.out).resolve()
    if a.resume:
        if a.plan_only:raise ValueError('Cannot resume as plan-only')
        inv=read(out/'inventory.json')
        if inv['protocol']!=p or inv['configs']!=a.config or inv['assets_manifest_sha256']!=sha(manifest) or inv['expect_gpu']!=a.expect_gpu or inv['status']=='planned_only':raise ValueError('Resume arguments differ from frozen collection')
        source=out/'source_snapshot';index=read(out/'CODE_SHA256.json');verify_code(source,index)
        # Retrying a failed cell requires a separate explicit diagnostic batch.
        if any(j['status']=='failed' for j in inv['jobs']):raise ValueError('Failed cells require separate reviewed retry batch; do not overwrite')
    else:
        out.mkdir(parents=True,exist_ok=False);source=out/'source_snapshot'
        original=Path(__file__).resolve().parents[1];shutil.copytree(original,source,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        index=code_index(source);write(out/'CODE_SHA256.json',index);write(out/'ASSET_VERIFICATION.json',asset_check)
        jobs=[{**j,'status':'planned_only' if a.plan_only else 'not_started'} for j in planned]
        inv={'protocol':p,'configs':a.config,'assets_manifest':str(manifest),'assets_manifest_sha256':sha(manifest),
             'expect_gpu':a.expect_gpu,'jobs':jobs,'utc_start':now(),'models_trained':0,
             'strict_reference_verified':False,'formal_training_eligible':False,'status':'planned_only' if a.plan_only else 'running'}
        write(out/'protocol.json',p);write(out/'schedule.json',planned);write(out/'inventory.json',inv)
    if a.plan_only:return
    env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join([str(source),str(source/'o4_pilot')]),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    paused=False;failures={}
    try:
        for j in inv['jobs']:
            if j['status']=='completed':continue
            if any(n>=2 for n in failures.get(j['config_id'],{}).values()):j['status']='deferred_repeated_failure';continue
            folder=out/j['folder'];folder.mkdir(exist_ok=False);j['status']='running';write(out/'inventory.json',inv)
            req={'protocol':p,'config_id':j['config_id'],'repeat':j['repeat'],'mode':j['mode'],
                 'expect_gpu':a.expect_gpu,'assets_manifest':str(manifest),'output':str(folder),'trace_backend':'cuda-sw'}
            write(folder/'request.json',req)
            try:
                verify_code(source,index)
                if sha(manifest)!=inv['assets_manifest_sha256']:raise ValueError('Asset manifest changed during collection')
                verify(manifest)
                gpu_guard(folder,a.expect_gpu,idle_wait_seconds=30,check_interval_seconds=3,idle_checks_required=2)
                argv=[sys.executable,'-m','o4_i3_context.worker','--request',str(folder/'request.json')]
                if j['mode']!='GI':argv=['nsys','profile','--trace=cuda-sw,nvtx','--sample=none','--cpuctxsw=none',
                    '--gpu-metrics-devices=none','--discard-environment=true','--capture-range=cudaProfilerApi',
                    '--capture-range-end=stop','--output='+str(folder/'trace')]+argv
                with Monitor(folder,p['GPU_monitor_interval_seconds']) as monitor:
                    command(argv,folder,'collect',timeout=600,env=env,cwd=source)
                write(folder/'monitor_summary.json',{'samples':len(monitor.rows),'foreign_GPU_work_observed':monitor.foreign,'errors':monitor.errors,
                    'availability':'sampled every 5s; does not prove absence between samples'})
                j['monitor_review']=monitor.foreign or bool(monitor.errors)
                if monitor.foreign:paused=True
                q=read(folder/'quality.json')
                if q['status']!='completed':raise RuntimeError('Worker did not complete')
                j['GPU_execution_completed']=True
                if j['mode']!='GI':
                    command(['nsys','export','--type=sqlite','--output='+str(folder/'trace.sqlite'),str(folder/'trace.nsys-rep')],folder,'export',timeout=120)
                    from o4_research.audit_v5 import audit
                    summary=audit(folder,folder/'audit');j['engineering_acceptance']=summary['engineering_acceptance']['status']
                    if not summary['engineering_acceptance']['accepted']:raise RuntimeError('Trace engineering gate blocked; preserve raw')
                verify_code(source,index);j.update(status='completed',utc_end=now())
            except Exception as e:
                j.update(status='failed',error=f'{type(e).__name__}: {e}',utc_end=now());write(folder/'failure.json',j)
                cause=read(folder/'quality.json').get('errors') if (folder/'quality.json').exists() else None
                key=str(cause) if cause else type(e).__name__+':'+str(e).split(';')[0].split(' see ')[0]
                bucket=failures.setdefault(j['config_id'],{});bucket[key]=bucket.get(key,0)+1
                if (folder/'gpu_guard.json').exists() and read(folder/'gpu_guard.json').get('blocked'):paused=True
            finally:write(out/'inventory.json',inv)
            print(j,flush=True)
            if paused:break
    finally:
        summarize(out,inv)
        inv.update(utc_end=now(),status='paused_GPU_guard_or_foreign_work' if paused else 'completed_development_collection' if all(j['status']=='completed' for j in inv['jobs']) else 'failed_or_incomplete')
        write(out/'inventory.json',inv)
    if inv['status']!='completed_development_collection':raise SystemExit(2 if paused else 1)

if __name__=='__main__':main()
