"""CPU-only old P-feature replay and explicit compatibility limitations."""
import argparse,csv,hashlib,json,os,sys
from pathlib import Path

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(old,selected,out):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    old=Path(old);out=Path(out);out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(old/'src'))
    from p_features import requests_from_p,source_views
    from model import z_transform
    import numpy as np,torch
    torch.set_num_threads(1)
    snap=read(old/'formal_session_snapshot.json');replays=[];normalizers=[];checked=[]
    for dev in ('gb10','rtx3090','rtx4090'):
        plan=Path('/home/good_egg/RTchen/O3')/snap[dev]['paths']['o2_plan'];p=read(plan)
        for fold in (0,4):
            t=next(t for t in p['templates'] if t['fold']==fold);view='S2' if fold<4 else 'S1'
            source=plan.parent/'runs'/t['id']/'P/run02/events_allowlist.jsonl'
            events=[json.loads(x) for x in source.read_text().splitlines() if x.strip()]
            actual=[{'a':x['a'],'b':x['b'],'z':x['z'][:27]} for x in source_views(requests_from_p(events))[view]]
            sid=hashlib.sha256((dev+'|'+t['id']).encode()).hexdigest()[:24]
            expected=next(x for x in read(old/f'predictor/{view}/sequences.json') if x['id']==sid)
            assert actual==[{'a':x['a'],'b':x['b'],'z':x['z'][:27]} for x in expected['candidates']]
            normfile=old/f'detectors/fold{fold}/s17/normalizers.json';norm=read(normfile)['z']
            transformed=z_transform([x['z'] for x in actual],norm).numpy()
            independent=((np.log1p(np.array([x['z'] for x in actual]))-np.array(norm['mean']))/np.array(norm['std'])).astype(np.float32)
            assert np.array_equal(transformed,independent) and np.isfinite(independent).all()
            replays.append({'device':dev,'fold':fold,'view':view,'template':t['id'],'source':str(source),'source_sha256':sha(source),
                'candidates':len(actual),'features':27,'span_unit':'microseconds','raw_feature_exact':True,
                'normalization_float32_exact':True,'normalizer_sha256':sha(normfile),
                'normalized_bytes_sha256':hashlib.sha256(independent.tobytes()).hexdigest()})
    manifest=read(old/'model_manifest.json');frozen=read(old/'FINAL_ARTIFACT_SHA256.json')
    for rel,h in manifest.items():
        if (rel.startswith('detectors/') and rel.endswith('epoch120.pt')) or ('/N0/' in rel and rel.endswith('model.pt')):
            f=old/rel;assert sha(f)==h
            nf=f.parent/'normalizers.json';assert sha(nf)==frozen[str(nf.relative_to(old))]
            n=read(nf);nz=n.get('source',n);dims=len(nz['z']['mean'])
            assert dims==27 and len(nz['h']['mean'])==2
            checked.append({'path':str(f),'sha256':h,'normalizer':str(nf),'normalizer_sha256':sha(nf),'z_dimensions':dims,'hardware_dimensions':2})
    from .controlled import write_json
    write_json(out/'OLD_FEATURE_REPLAY.json',{'replays':replays,'scope':'6 archived P inputs, 2 folds x 3 devices; no model inference or training'})
    write_json(out/'MODEL_NORMALIZER_INDEX.json',checked)
    new=[]
    for row in read(selected):
        root=Path(row['root']);work=row['workload'];p=root/f'{work}_P_r1/audit/predictor/measure_000007.jsonl'
        events=[json.loads(x) for x in p.read_text().splitlines()]
        non=[e for e in events if e['event_kind']!='kernel'];missing=sum(e.get('grid_xyz') is None or e.get('block_xyz') is None or e.get('dynamic_shared_memory_bytes') is None for e in events)
        new.append({'device':row['device'],'workload':work,'P_iteration_8':str(p),'P_input_sha256':sha(p),'events':len(events),
                    'nonkernel_events':len(non),'events_missing_legacy_geometry':missing,
                    'old_numeric_builder_compatible':not missing and not non,
                    'full_model_transfer_ready':False,'reason':'source view and architecture-specific hardware anchors still require validation; L4 and three-class taxonomy exclude DNN completeness'})
    write_json(out/'O4_INPUT_COMPATIBILITY.json',new)
    write_json(out/'SUMMARY.json',{'old_P_replays_exact':len(replays),'weights_and_normalizers_verified':len(checked),
        'O4_cases_examined':len(new),'O4_geometry_incompatible_cases':sum(not r['old_numeric_builder_compatible'] for r in new),
        'old_hardware_calibration':'retained, not revalidated for current Torch/CUDA stacks',
        'model_inference_executed':False,'new_training':False,'formal_DNN_transfer_eligible':False})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--old',required=True);p.add_argument('--selected',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.old,a.selected,a.out)
