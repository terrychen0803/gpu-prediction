"""CPU static candidate evidence; never upgrades historical runtime arguments."""
import argparse,csv,hashlib,inspect,json,os
from pathlib import Path
from .controlled import write_json

def run(index,out):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch,torchvision
    from torchvision.models import resnet50
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261107);model=resnet50(weights=None).eval()
    candidates=[]
    for name,m in model.named_modules():
        if isinstance(m,torch.nn.Conv2d):
            candidates.append({'module':name,'weight_shape':list(m.weight.shape),'parameters':{
                'stride':list(m.stride),'padding':list(m.padding),'dilation':list(m.dilation),
                'groups':m.groups,'bias_presence':m.bias is not None,'padding_mode':m.padding_mode}})
    rows=[]
    for r in json.loads(Path(index).read_text()):
        if r['workload']!='RN-A':continue
        f=Path(r['root'])/'RN-A_R_r1/audit/reference_operations.json'
        for op in json.loads(f.read_text()):
            if op['stage']!='measure' or op['iteration']!=0 or op['aten']!='aten.conv2d.default':continue
            matches=[c for c in candidates if c['weight_shape']==op['input_shapes'][1]]
            possible={json.dumps(c['parameters'],sort_keys=True) for c in matches}
            rows.append({'device':r['device'],'op_index':op['op_index'],'weight_shape':op['input_shapes'][1],
                'matching_static_modules':[c['module'] for c in matches],
                'candidate_parameter_sets':[json.loads(x) for x in sorted(possible)],
                'candidate_parameters_unambiguous':len(possible)==1,
                'runtime_arguments_verified':False,'status':'static_candidate_only_not_historical_runtime_proof'})
    file=Path(inspect.getsourcefile(resnet50))
    write_json(out,{'torchvision_build':torchvision.__version__,'definition_file':str(file),
        'definition_sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'model':'resnet50(weights=None).eval() used only for structure',
        'random_initialization_used_as_measured_weights':False,'rows':rows,
        'parameter_unambiguous_rows':sum(r['candidate_parameters_unambiguous'] for r in rows),
        'parameter_ambiguous_or_absent_rows':sum(not r['candidate_parameters_unambiguous'] for r in rows),
        'warning':'cross-stack candidates require adapter/runtime argument verification; shape matching does not identify layer or weight bytes'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--index',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    if Path(a.out).exists():raise FileExistsError(a.out)
    run(a.index,a.out)
