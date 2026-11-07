"""Additive v5 engineering overlay; v4 numerical outputs remain unchanged."""
import csv,json,sqlite3
from pathlib import Path
from .audit import audit as historical_audit
from .common import read,write
from .trace_contract import acceptance

def audit(folder, out):
    folder, out = Path(folder), Path(out)
    summary = historical_audit(folder, out)
    env = read(folder/'environment.json'); req = read(folder/'request.json')
    with sqlite3.connect('file:'+str(folder/'trace.sqlite')+'?mode=ro',uri=True) as c:
        c.row_factory=sqlite3.Row
        tables={x[0] for x in c.execute("select name from sqlite_master where type='table'")}
        strings={x['id']:x['value'] for x in c.execute('select * from StringIds')}
        apis=[]
        for table in ('CUPTI_ACTIVITY_KIND_RUNTIME','CUPTI_ACTIVITY_KIND_DRIVER'):
            if table not in tables:continue
            for row in c.execute('select rowid as _rowid,* from '+table):
                a=dict(row)
                if ((a['globalTid']>>24)&0xffffff)!=env['pid']:continue
                a.update(key=f'{table}:{a["_rowid"]}',name=strings.get(a['nameId'],''));apis.append(a)
    events=[]
    with (out/'events_diagnostic.csv').open() as f:
        for row in csv.DictReader(f):
            e=dict(row)
            for key in ('grid_xyz','block_xyz'):
                try:e[key]=json.loads(e[key])
                except (ValueError,KeyError):e[key]=None
            for key in ('dynamic_shared_memory_bytes','bytes','copy_kind','memory_kind','memset_value'):
                try:e[key]=int(e[key])
                except (ValueError,KeyError,TypeError):e[key]=None
            events.append(e)
    result=acceptance(read(out/'diagnostics.json'),read(out/'schema.json'),events,apis,
                      summary['status']=='recorded_structure_complete',req.get('trace_backend','cuda-sw'))
    write(out/'engineering_acceptance.json',result)
    # SUMMARY.json is intentionally the unchanged v4 summary for numerical regression.
    return {**summary,'engineering_acceptance':result}
