import copy,json,sqlite3,tempfile,unittest
from pathlib import Path
from o4_iteration.common import ROOT,read,code_manifest
from o4_iteration.manifest import POLICY,stable,validate_request
from o4_iteration.trace import audit,closure,safe_event,SAFE
class Core(unittest.TestCase):
 def test_stability_uses_last_three_windows_and_cap(self):
  self.assertFalse(stable([10]*50));self.assertTrue(stable([10]*60));self.assertFalse(stable([10]*40+[20]*20));self.assertTrue(stable([100]*20+[10]*60))
 def test_units_and_timer_boundaries(self):
  from o4_iteration.timer import measure
  calls=[]
  class Stream:pass
  stream=Stream()
  class Event:
   def __init__(self,name):self.name=name
   def record(self,s):self_test.assertIs(s,stream);calls.append(self.name)
   def synchronize(self):calls.append('sync')
   def elapsed_time(self,end):calls.append('elapsed');return .0125
  class Cuda:
   @staticmethod
   def current_stream():return stream
  class Torch:cuda=Cuda()
  class A:
   def forward(self):calls.append('forward');return object()
  self_test=self
  row=measure(Torch(),A(),Event('start'),Event('end'),stream,'GI','measure',0)
  self.assertEqual(calls,['start','forward','end','sync','elapsed']);self.assertEqual(row['elapsed_us'],12.5);self.assertEqual(row['target_kind'],'iteration_device_event_us')
 def test_timer_rejects_changed_stream(self):
  from o4_iteration.timer import measure
  class C:current_stream=staticmethod(lambda:object())
  class T:cuda=C()
  with self.assertRaisesRegex(RuntimeError,'stream'):measure(T(),None,None,None,object(),'GI','measure',0)
 def test_closure_no_double_internal_gap(self):
  g=[[{'start_ns':0,'end_ns':2},{'start_ns':5,'end_ns':8}],[{'start_ns':10,'end_ns':15}]]
  x=closure(g);self.assertEqual(x['operation_span_sum_us'],.013);self.assertEqual(x['inter_operation_gap_us'],.002);self.assertEqual(x['closure_error_ns'],0)
  with self.assertRaises(ValueError):closure([g[1],g[0]])
 def test_protocol_and_planned_rejected(self):
  r={'protocol':copy.deepcopy(POLICY),'mode':'GI','workload':'RN-A','assets':{},'planned_only':True}
  with self.assertRaisesRegex(ValueError,'planned'):validate_request(r)
  r['planned_only']=False;r['protocol']['measured_iterations']=1
  with self.assertRaisesRegex(ValueError,'Protocol'):validate_request(r)
 def test_missing_asset_rejected(self):
  r={'protocol':POLICY,'mode':'GI','workload':'RN-A','assets':{'bad':{'path':'nonexistent','sha256':'x'}}}
  with self.assertRaisesRegex(ValueError,'hash'):validate_request(r)
 def test_safe_export_poison_names_shapes_truth(self):
  e={'start_ns':100,'end_ns':200,'api_start_ns':90,'api_end_ns':95,'event_kind':'kernel','kernel_name':'SECRET','shape':[123],'target_time':999,'op_type':'SECRET','iteration_scope':'SECRET','api_tid':999}
  self.assertEqual(safe_event(e,90,0),safe_event({**e,'shape':[0],'target_time':0,'kernel_name':'changed','op_type':'changed'},90,0));self.assertLessEqual(set(safe_event(e,90,0)),SAFE)
 def test_request_gpu_import_free_and_reject_overwrite(self):
  import subprocess,sys
  p=subprocess.run([sys.executable,'-m','o4_iteration','--help'],capture_output=True,text=True);self.assertEqual(p.returncode,0)
  p=subprocess.run([sys.executable,'-m','o4_iteration','run','--out',str(ROOT),'--expect-gpu','GB10','--dry-run'],capture_output=True,text=True);self.assertNotEqual(p.returncode,0)
class Trace(unittest.TestCase):
 def fixture(self,root,mode='R'):
  folder=root/'raw';folder.mkdir();dest=root/'audit';pid=7;tid=pid<<24
  (folder/'request.json').write_text(json.dumps({'mode':mode,'protocol':POLICY,'opaque_id':'opaque'}));(folder/'environment.json').write_text(json.dumps({'pid':pid}));(folder/'quality.json').write_text(json.dumps({'status':'completed'}));(folder/'trace.nsys-rep').write_bytes(b'fixture_only')
  c=sqlite3.connect(folder/'trace.sqlite');c.executescript('CREATE TABLE StringIds(id,value); INSERT INTO StringIds VALUES(1,"cudaLaunchKernel"),(2,"diagnostic_name"); CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start,end,globalTid,correlationId,nameId); CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start,end,globalPid,correlationId,deviceId,contextId,streamId,demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ,dynamicSharedMemory); CREATE TABLE NVTX_EVENTS(start,end,text,textId,globalTid); CREATE TABLE DIAGNOSTIC_EVENT(severity,text); INSERT INTO DIAGNOSTIC_EVENT VALUES(2,"fixture warning");')
  meta=[];i=0
  for stage,n in [('transition',5),('measure',40)]:
   for idx in range(n):
    t=i*1000;i+=1;label=f'o4.iter/{stage}/{idx:06d}';c.execute('insert into NVTX_EVENTS values(?,?,?,?,?)',(t,t+900,label,None,tid));c.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)',(t+100,t+110,tid,i,1));c.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(t+120,t+220,tid,i,0,1,7,2,1,1,1,32,1,1,0))
    if mode=='R':
     for op_idx,a,b in [(0,90,115),(1,300,400)]:
      scope=f'o4.op/{stage}/{idx:06d}/{op_idx:04d}';c.execute('insert into NVTX_EVENTS values(?,?,?,?,?)',(t+a,t+b,scope,None,tid));meta.append({'scope':scope,'stage':stage,'iteration':idx,'op_index':op_idx,'aten':'aten.mm.default' if op_idx==0 else 'aten.view.default','input_shapes':[[1,1]]})
  c.commit();c.close();(folder/'reference_dispatch.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in meta));return folder,dest
 def test_reference_warning_zero_work_and_scope_coverage(self):
  with tempfile.TemporaryDirectory() as d:
   f,o=self.fixture(Path(d));q=audit(f,o);self.assertEqual(q['status'],'warning_accepted');self.assertEqual(q['zero_gpu_operations'],45);self.assertEqual(q['mapped_work_events'],45);self.assertFalse(q['formal_reference_verified']);self.assertTrue(q['all_closure_errors_zero'])
 def test_numeric_export_no_reference_required(self):
  with tempfile.TemporaryDirectory() as d:
   f,o=self.fixture(Path(d),'P');(f/'reference_dispatch.jsonl').unlink();q=audit(f,o);self.assertNotEqual(q['status'],'failed');paths=list((o/'predictor').glob('*/events.jsonl'));self.assertEqual(len(paths),40)
   for p in paths:
    for row in map(json.loads,p.read_text().splitlines()):self.assertLessEqual(set(row),SAFE);self.assertNotIn('kernel_name',row)
 def test_ambiguous_correlation_is_failure(self):
  with tempfile.TemporaryDirectory() as d:
   f,o=self.fixture(Path(d));c=sqlite3.connect(f/'trace.sqlite');c.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME select * from CUPTI_ACTIVITY_KIND_RUNTIME limit 1');c.commit();c.close();self.assertEqual(audit(f,o)['status'],'failed')
 def test_side_stream_and_missing_scope_fail(self):
  for sql in ['update CUPTI_ACTIVITY_KIND_KERNEL set streamId=9 where correlationId=1','delete from NVTX_EVENTS where text="o4.iter/measure/000000"']:
   with tempfile.TemporaryDirectory() as d:
    f,o=self.fixture(Path(d));c=sqlite3.connect(f/'trace.sqlite');c.execute(sql);c.commit();c.close();self.assertEqual(audit(f,o)['status'],'failed')
 def test_orphan_gpu_work_fails(self):
  with tempfile.TemporaryDirectory() as d:
   f,o=self.fixture(Path(d));c=sqlite3.connect(f/'trace.sqlite');c.execute('delete from NVTX_EVENTS where text="o4.op/measure/000000/0000"');c.commit();c.close();self.assertEqual(audit(f,o)['status'],'failed')
if __name__=='__main__':unittest.main()
