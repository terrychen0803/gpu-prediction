"""Configuration groups and honest split feasibility; never split repeats as configs."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
from .controlled import write_json

def signature(c):
    keys=['subtype','operands','parameters','output_shape','dtype','layout','inplace']
    return hashlib.sha256(json.dumps({k:c[k] for k in keys},sort_keys=True).encode()).hexdigest()

def feasibility(configurations):
    groups={}
    for c in configurations:
        s=signature(c)
        groups.setdefault(s,{'group_id':s,'subtype':c['subtype'],'config_ids':[]})['config_ids'].append(c['id'])
    counts=Counter(g['subtype'] for g in groups.values())
    return {'status':'insufficient_for_train_validation_test' if any(n<3 for n in counts.values()) else 'allocation_not_yet_frozen',
            'groups':list(groups.values()),'groups_per_subtype':dict(counts),
            'requirement':'at least distinct train/validation/test configuration groups per subtype; 3 is only a structural minimum, not statistical sufficiency',
            'split_assignment':None,'cross_device_repeat_view_policy':'same group_id across devices, dates, repeats, views and derived contexts',
            'context_A':'requires separately frozen dependency-context manifest; no subgraphs collected yet',
            'model_C':'existing seven A are development; Qwen prefill/decode same family for model-transfer claims',
            'training_ready':False,'confirmation_set_created':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    out=Path(a.out)
    if out.exists():raise FileExistsError(out)
    write_json(out,feasibility(json.loads(Path(a.assets).read_text())['configurations']))
