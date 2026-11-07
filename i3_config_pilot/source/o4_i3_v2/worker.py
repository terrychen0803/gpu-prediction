"""One process/context per cell; no CPU timing fallback or timestamp repair."""
import argparse,contextlib,math,os,platform,sys,time,traceback
from pathlib import Path
from o4_research.common import read,write,csvwrite,now
from .assets import Adapter
from .protocol import validate_protocol,warmup_stable,distribution,measured_values

def run(request):
    req=read(request);out=Path(req['output']);p=validate_protocol(req['protocol'])
    inv=None;capturing=False;rows=[];warm=[]
    quality={'status':'started','utc_start':now(),'strict_reference_verified':False,'formal_training_eligible':False,'errors':[]}
    write(out/'quality.json',quality)
    try:
        import torch
        from o4_iteration import common as bootstrap
        from inference_pilot.benchmark import precision_policy
        from o4_i3.dispatch import inventory
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible CUDA GPU required; no CPU fallback')
        torch.cuda.set_device(0);policy=precision_policy(torch);stream=torch.cuda.current_stream();props=torch.cuda.get_device_properties(0)
        if req['expect_gpu'].lower() not in props.name.lower():raise RuntimeError('GPU identity mismatch')
        write(out/'environment.json',{'utc':now(),'pid':os.getpid(),'python':sys.version,'machine':platform.machine(),
           'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
           'gpu':props.name,'precision_policy':policy,'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),
           'single_stream_semantics':'requested current stream; raw audit must independently check observed work'})
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            adapter=Adapter(torch,req['assets_manifest'],req['config_id']);write(out/'adapter.json',adapter.metadata)
            quality['correctness']=adapter.correctness()
            if req['mode']=='R':
                inv=inventory(torch);plain=adapter.forward().detach()
                with inv:
                    inv.begin('measure',0)
                    try:marked=adapter.forward().detach()
                    finally:inv.end()
                torch.cuda.synchronize()
                if not torch.equal(plain,marked):raise RuntimeError('Active dispatch output parity failed')
                del plain,marked;inv.rows.clear();quality['instrumented_output_exact']=True
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record(stream);end.record(stream);end.synchronize()
            if req['mode']!='GI':torch.cuda.nvtx.range_push('o4.setup_nvtx');torch.cuda.nvtx.range_pop()
            with inv if inv is not None else contextlib.nullcontext():
                stable=False
                for i in range(p['warmup_max']):
                    adapter.prepare();torch.cuda.synchronize()
                    if torch.cuda.current_stream()!=stream:raise RuntimeError('Prepare changed current stream')
                    if inv:inv.begin('warmup',i)
                    try:
                        start.record(stream);output=adapter.forward();end.record(stream);end.synchronize()
                        elapsed=float(start.elapsed_time(end))*1000
                    finally:
                        if inv:inv.end();inv.rows.clear()
                    if not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('Invalid warmup GPU event duration')
                    warm.append({'iteration':i,'utc':now(),'elapsed_us':elapsed,'reference_active':inv is not None})
                    del output
                    if p['kind']=='smoke':stable=len(warm)==p['warmup_max']
                    elif len(warm)%p['warmup_window']==0:stable=warmup_stable([r['elapsed_us'] for r in warm],p)
                    if stable:break
                quality['warmup']={'passed':stable,'iterations':len(warm),'stability_evaluated':p['kind']=='formal','R_active':inv is not None}
                if not stable:raise RuntimeError('Warmup failed three-window stability before 500-iteration cap')
                if req['mode']!='GI':torch.cuda.profiler.start();capturing=True
                for stage,n in [('transition',p['transition_iterations']),('measure',p['measured_iterations'])]:
                    for i in range(n):
                        marked=req['mode']!='GI'
                        if marked:torch.cuda.nvtx.range_push(f'o4.prepare/{stage}/{i:06d}')
                        adapter.prepare();torch.cuda.synchronize()
                        if torch.cuda.current_stream()!=stream:raise RuntimeError('Iteration preparation changed current stream')
                        if marked:torch.cuda.nvtx.range_pop();torch.cuda.nvtx.range_push(f'o4.iter/{stage}/{i:06d}')
                        if inv:inv.begin(stage,i)
                        utc=now();tick=time.perf_counter_ns()
                        try:
                            start.record(stream);output=adapter.forward();end.record(stream);end.synchronize()
                            elapsed=float(start.elapsed_time(end))*1000
                        finally:
                            if inv:inv.end()
                            if marked:torch.cuda.nvtx.range_pop()
                        finish=time.perf_counter_ns()
                        if torch.cuda.current_stream()!=stream or not math.isfinite(elapsed) or elapsed<=0:raise RuntimeError('Invalid iteration stream/duration')
                        rows.append({'stage':stage,'iteration':i,'utc_start':utc,'utc_end':now(),
                           'start_monotonic_ns':tick,'end_monotonic_ns':finish,'elapsed_us':elapsed,'status':'completed',
                           'target_kind':'iteration_device_event_us' if req['mode']=='GI' else 'profiled_iteration_event_us'})
                        del output
            quality['distribution']=distribution(measured_values(rows,p))
            quality['tail_review']=quality['distribution']['tail_fraction']>p['within_process_tail_review_threshold']
            if capturing:
                del start,end,adapter;torch.cuda.synchronize()
                from o1_operator.worker import reset_cuda_context_for_trace
                quality['own_context_reset_returncode']=reset_cuda_context_for_trace(torch.version.cuda)
                if quality['own_context_reset_returncode']!=0:raise RuntimeError('Own-context finalization failed')
                torch.cuda.profiler.stop();capturing=False
            quality.update(status='completed',utc_end=now(),measured_iterations=p['measured_iterations'])
    except Exception as e:
        quality.update(status='failed',utc_end=now());quality['errors'].append(f'{type(e).__name__}: {e}')
        (out/'worker_error.txt').write_text(traceback.format_exc());print(quality['errors'][-1],file=sys.stderr)
    finally:
        if capturing:
            try:
                import torch
                torch.cuda.profiler.stop()
            except Exception:pass
        if warm:csvwrite(out/'warmup.csv',warm)
        if rows:csvwrite(out/'iterations.csv',rows)
        if inv is not None:
            with (out/'reference_dispatch.jsonl').open('w') as f:
                for row in inv.rows:f.write(__import__('json').dumps(row,allow_nan=False)+'\n')
        write(out/'quality.json',quality)
    return 0 if quality['status']=='completed' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args();raise SystemExit(run(a.request))
