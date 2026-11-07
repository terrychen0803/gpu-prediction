"""Independent-process reparse of the new smoke only; preserves original raw/audit."""
import argparse,json,hashlib
from pathlib import Path
from o4_research.audit_v5 import audit
from o4_research.audit import pair
from .controlled import write_json

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compare(old,new):
    files=[p for p in old.rglob('*') if p.is_file()]
    assert {str(p.relative_to(old)) for p in files}=={str(p.relative_to(new)) for p in new.rglob('*') if p.is_file()}
    for p in files:assert sha(p)==sha(new/p.relative_to(old)),p
    return len(files)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
    out=run/'analysis/smoke_replay';out.mkdir(exist_ok=False)
    checks=[];pairs=[]
    for name in ['smoke_gb10_bn','smoke_gb10_A0','smoke_gb10_frozen_tail']:
        source=run/name;inv=json.loads((source/'inventory.json').read_text());dest=out/name;dest.mkdir()
        for j in inv['jobs']:
            if j['status']!='completed' or j['mode']=='GI':continue
            cell=source/j['folder'];before=sha(cell/'trace.sqlite');summary=audit(cell,dest/j['folder'])
            count=compare(cell/'audit',dest/j['folder'])
            assert before==sha(cell/'trace.sqlite')==summary['sqlite_sha256']
            checks.append({'collection':name,'cell':j['folder'],'files_byte_exact':count,'raw_sha256':before})
        for pairinfo in inv['P_R_pairs']:
            c=pairinfo['config_id'];old=source/(c+'_pair');new=dest/(c+'_pair')
            pair(dest/(c+'_P'),dest/(c+'_R'),new);count=compare(old,new)
            pairs.append({'collection':name,'config_id':c,'files_byte_exact':count})
    write_json(out/'SUMMARY.json',{'trace_replays':len(checks),'P_R_pair_replays':len(pairs),
        'byte_identical_artifacts':sum(x['files_byte_exact'] for x in checks+pairs),
        'trace_checks':checks,'pair_checks':pairs,'strict_reference_verified':False,
        'scope':'new GB10 I3 smoke; reproduction does not prove reference completeness'})

if __name__=='__main__':main()
