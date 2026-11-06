"""Audit completed train/validation R metadata before any S3 test capture."""
import collections,datetime,importlib.util,json
from pathlib import Path
r=Path(__file__).resolve().parent
sp=importlib.util.spec_from_file_location('agg',r/'summarize_s3.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
assert not (r/'collections/sealed_test_02_smoke').exists(),'Test capture already started; cannot call this a pre-test audit'
old1=r.parent/'o4_i3_s1_gb10_20261106T213722Z';old2=r.parent/'o4_i3_s2_gb10_20261106T220336Z'
configs={}
for f in [old1/'assets_S1/ASSETS.json',old2/'assets_S2/ASSETS.json',r/'assets_S3/ASSETS.json']:
 for e in m.read(f)['configurations']:configs[e['configuration']['id']]=e['configuration']
indexes=list((old1/'collections').glob('*/inventory.json'))+list((old2/'collections').glob('*/inventory.json'))+list((r/'collections').glob('*/inventory.json'))
uses=collections.defaultdict(set);checked=0;mismatches=[];seen=set()
for invpath in indexes:
 inv=m.read(invpath);folder=invpath.parent
 for j in inv['jobs']:
  if j['mode']!='R' or j['status']!='completed':continue
  cid=j['config_id'];seen.add(cid)
  for op in m.read(folder/j['folder']/'audit/reference_operations.json'):
   checked+=1;result=m.check_runtime(configs[cid],op)
   if not result['matched']:mismatches.append({'config_id':cid,**result})
   if op['stage']=='measure':uses[m.digest(m.semantic(op))].add(cid)
collisions=[]
for key,ids in uses.items():
 splits=sorted({configs[c]['split_B'] for c in ids})
 if len(splits)>1:collisions.append({'semantic_sha256':key,'configs':sorted(ids),'splits':splits})
validation={c['id'] for c in m.read(r/'SELECTED_CONFIGS.json') if c['split_B']=='validation'}
assert validation<=seen,validation-seen
out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'test_capture_started':False,'validation_configs_seen':len(validation),'R_parameter_rows_checked':checked,'runtime_mismatches':mismatches,'exact_runtime_cross_split_collisions':collisions,'model_scores_read':False,'limitations':'Full metadata equality audit only; mathematical equivalence / nearest-neighbor exposure / historical DNN pretraining not excluded; final complete formal audit still required','strict_reference_verified':False}
(r/'evidence/PRE_TEST_RUNTIME_AUDIT.json').write_text(json.dumps(out,indent=2)+'\n')
assert not mismatches and not collisions
print(json.dumps(out,indent=2))
