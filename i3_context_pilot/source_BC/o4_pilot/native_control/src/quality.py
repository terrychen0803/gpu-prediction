"""Read-only quality primitives, version v4; never repairs timestamps."""
from __future__ import annotations
import hashlib,json,csv,statistics
from pathlib import Path
VERSION='o4_quality_gates_v4_20261106'
GATES=('recorded_event_integrity','submitted_work_coverage','reference_semantic_alignment','timestamp_consistency','event_index_partition_supported','nonoverlap_sum_supported','overlap_aware_closure_supported','strict_reference_verified')
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def csvwrite(p,data,fields=None):
 data=list(data);keys=fields or list(dict.fromkeys(k for r in data for k in r))
 if not keys:return
 with Path(p).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
  for r in data:w.writerow({k:json.dumps(r.get(k),ensure_ascii=False) if isinstance(r.get(k),(dict,list,tuple)) else r.get(k) for k in keys})
def quant(a):
 a=sorted(a)
 if not a:return {'n':0}
 def q(p):
  x=(len(a)-1)*p;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
 return {'n':len(a),'min':a[0],'median':q(.5),'p90':q(.9),'p99':q(.99),'max':a[-1],'mean':statistics.mean(a)}
def union_ns(intervals):
 end=None;total=0
 for a,b in sorted(intervals):
  if not isinstance(a,int) or not isinstance(b,int) or b<=a:raise ValueError('Expected positive INTEGER-ns intervals')
  total+=b-a if end is None else max(0,b-max(a,end));end=b if end is None else max(end,b)
 return total
def interval_closure(intervals):
 intervals=list(intervals)
 if not intervals:raise ValueError('No nonzero-work intervals; span must remain null')
 covered=union_ns(intervals);span=max(b for a,b in intervals)-min(a for a,b in intervals);parts=sum(b-a for a,b in intervals)
 overlap=parts-covered;gap=span-covered
 return {'interval_count':len(intervals),'span_ns':span,'span_sum_ns':parts,'interval_union_ns':covered,'positive_gap_ns':gap,'union_overlap_ns':overlap,'closure_error_ns':span-parts-gap+overlap,'nonoverlap_observed':overlap==0,'interpretation':'recorded interval arithmetic only; not GI or reference integrity proof'}
def overlapping_pairs(events):
 seq=sorted(events,key=lambda e:(e['start'],e['end'],e['uid']));active=[];out=[]
 for e in seq:
  active=[x for x in active if x['end']>e['start']]
  out.extend((x,e,min(x['end'],e['end'])-e['start']) for x in active);active.append(e)
 return out
def partition_supported(owners):
 """owners in GPU event-index order; each owner must form one contiguous run."""
 if not owners or any(x is None for x in owners):return False
 seen=set();previous=object()
 for x in owners:
  if x!=previous:
   if x in seen:return False
   seen.add(x);previous=x
 return True
def gate(status,evidence,reason):
 if status not in ('passed','warning_accepted','failed','unverified'):raise ValueError(status)
 return {'status':status,'evidence':evidence,'reason':reason,'version':VERSION}
def strict_gate(gates,evidence):
 # Recorded-span arithmetic cannot upgrade semantic/submitted/clock evidence.
 required=('recorded_event_integrity','submitted_work_coverage','reference_semantic_alignment','timestamp_consistency')
 ok=all(gates[k]['status']=='passed' for k in required)
 return gate('passed' if ok else 'unverified',evidence,'all required independent integrity gates passed' if ok else 'at least one independent integrity gate remains unresolved or failed')
