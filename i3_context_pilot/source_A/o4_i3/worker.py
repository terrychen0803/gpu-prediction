"""Bounded A0 smoke worker; all outputs development/unverified."""
import argparse,contextlib,os,platform,sys,time,traceback
from pathlib import Path
from .controlled import Adapter
from o4_research.common import read,write,csvwrite,now

PROTOCOL={'protocol_id':'o4_i3_A0_smoke_v1_20261107','warmup':8,'transition_iterations':1,
          'measured_iterations':3,'process_repeats':1,'dtype':'float16','trace_backend':'cuda-sw',
          'eager':True,'autocast':False,'compile':False,'cuda_graph':False,
          'reference_arguments':'full bound ATen schema and tensor metadata; R only',
          'repeat_stability_evaluated':False,'formal_training_eligible':False,
          'correctness_relative_l2_max':.02,'correctness_max_abs_over_scale_max':.02,
          'scope':'single controlled operation; preparation and correctness outside timing',
          'finalization':'release own tensors/events; synchronize; own-context reset; profiler stop; no drain'}

def run(reqfile):
    req=read(reqfile);out=Path(req['output']);capturing=False;inv=None;rows=[]
    q={'status':'started','utc_start':now(),'formal_reference_verified':False,'formal_training_eligible':False,'errors':[]}
    try:
        import torch
        from o4_iteration import common as bootstrap
        from inference_pilot.benchmark import precision_policy
        from .dispatch import inventory
        if req['protocol']!=PROTOCOL:raise ValueError('Protocol mismatch')
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Exactly one visible GPU required')
        torch.cuda.set_device(0);policy=precision_policy(torch);stream=torch.cuda.current_stream();props=torch.cuda.get_device_properties(0)
        if req['expect_gpu'].lower() not in props.name.lower():raise RuntimeError('GPU identity mismatch')
        write(out/'environment.json',{'utc':now(),'pid':os.getpid(),'python':sys.version,'machine':platform.machine(),
            'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
            'gpu':props.name,'precision':policy})
        with torch.inference_mode(),torch.autocast('cuda',enabled=False):
            adapter=Adapter(torch,req['assets_manifest'],req['config_id']);write(out/'adapter.json',adapter.metadata)
            q['correctness']=adapter.correctness()
            if req['mode']=='R':
                inv=inventory(torch);plain=adapter.forward().detach()
                with inv:marked=adapter.forward().detach()
                torch.cuda.synchronize()
                if not torch.equal(plain,marked):raise RuntimeError('Dispatch inactive changes output')
                # Also validate an active reference route before profiling, then reset its rows.
                with inv:
                    inv.begin('measure',0);active=adapter.forward().detach();inv.end()
                torch.cuda.synchronize()
                if not torch.equal(plain,active):raise RuntimeError('Active dispatch changes output')
                inv.rows.clear();q['instrumented_output_exact']=True;del plain,marked,active
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record(stream);end.record(stream);end.synchronize()
            if req['mode']!='GI':torch.cuda.nvtx.range_push('o4.setup_nvtx');torch.cuda.nvtx.range_pop()
            for _ in range(PROTOCOL['warmup']):adapter.forward();torch.cuda.synchronize()
            if req['mode']!='GI':torch.cuda.profiler.start();capturing=True
            with inv if inv is not None else contextlib.nullcontext():
                for stage,count in [('transition',1),('measure',3)]:
                    for i in range(count):
                        if req['mode']!='GI':torch.cuda.nvtx.range_push(f'o4.prepare/{stage}/{i:06d}')
                        adapter.prepare();torch.cuda.synchronize()
                        if req['mode']!='GI':torch.cuda.nvtx.range_pop();torch.cuda.nvtx.range_push(f'o4.iter/{stage}/{i:06d}')
                        if inv:inv.begin(stage,i)
                        utc_start=now();tick=time.perf_counter_ns();start.record(stream);result=adapter.forward();end.record(stream);end.synchronize()
                        elapsed=start.elapsed_time(end)*1000;finish=time.perf_counter_ns();utc_end=now()
                        if inv:inv.end()
                        if req['mode']!='GI':torch.cuda.nvtx.range_pop()
                        rows.append({'stage':stage,'iteration':i,'utc_start':utc_start,'utc_end':utc_end,
                            'start_monotonic_ns':tick,'end_monotonic_ns':finish,'elapsed_us':elapsed,
                            'target_kind':'iteration_device_event_us' if req['mode']=='GI' else 'profiled_iteration_event_us','status':'completed'})
                        del result
            if capturing:
                del start,end,adapter;torch.cuda.synchronize()
                from o1_operator.worker import reset_cuda_context_for_trace
                q['own_context_reset_returncode']=reset_cuda_context_for_trace(torch.version.cuda)
                if q['own_context_reset_returncode']!=0:raise RuntimeError('Own context finalization failed')
                torch.cuda.profiler.stop();capturing=False
            q.update(status='completed',utc_end=now(),measured_iterations=3,reference_status='development_unverified')
    except Exception as e:
        q.update(status='failed',utc_end=now());q['errors'].append(f'{type(e).__name__}: {e}')
        (out/'worker_error.txt').write_text(traceback.format_exc());print(q['errors'][-1],file=sys.stderr)
    finally:
        if capturing:
            try:
                import torch
                torch.cuda.profiler.stop()
            except Exception:pass
        if rows:csvwrite(out/'iterations.csv',rows)
        if inv is not None:
            with (out/'reference_dispatch.jsonl').open('w') as f:
                for row in inv.rows:f.write(__import__('json').dumps(row,allow_nan=False)+'\n')
        write(out/'quality.json',q)
    return 0 if q['status']=='completed' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);a=p.parse_args();raise SystemExit(run(a.request))
