import json,sqlite3,tempfile,unittest
from pathlib import Path
import test_o4
from o4_iteration.trace import audit
class ExtraTrace(unittest.TestCase):
 def test_copy_is_exported_and_counted(self):
  with tempfile.TemporaryDirectory() as d:
   f,o=test_o4.Trace().fixture(Path(d),'P');c=sqlite3.connect(f/'trace.sqlite')
   c.executescript('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY(start,end,globalPid,correlationId,deviceId,contextId,streamId,bytes,copyKind); INSERT INTO StringIds VALUES(3,"cudaMemcpyAsync");')
   c.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,?,?)',(150,155,7<<24,10000,3));c.execute('insert into CUPTI_ACTIVITY_KIND_MEMCPY values(?,?,?,?,?,?,?,?,?)',(250,270,7<<24,10000,0,1,7,128,1));c.commit();c.close()
   q=audit(f,o);self.assertNotEqual(q['status'],'failed');self.assertEqual(q['copy_count'],1);self.assertEqual(q['work_event_count'],46)
 def test_unknown_work_schema_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   f,o=test_o4.Trace().fixture(Path(d),'P');c=sqlite3.connect(f/'trace.sqlite');c.executescript('CREATE TABLE CUPTI_ACTIVITY_KIND_MEMSET(start,end);');c.commit();c.close();self.assertEqual(audit(f,o)['status'],'failed')
 def test_zero_work_span_is_null(self):
  with tempfile.TemporaryDirectory() as d:
   f,o=test_o4.Trace().fixture(Path(d));q=audit(f,o);ref=[json.loads(x) for x in (o/'reference_operations.jsonl').read_text().splitlines()]
   zero=[x for x in ref if x['zero_gpu_work']];self.assertEqual(len(zero),45);self.assertTrue(all(x['contextual_trace_span_us'] is None for x in zero))
 def test_unique_correlation_preserved_despite_cpu_gpu_clock_inversion(self):
  with tempfile.TemporaryDirectory() as d:
   f,o=test_o4.Trace().fixture(Path(d),'P');c=sqlite3.connect(f/'trace.sqlite');c.execute('update CUPTI_ACTIVITY_KIND_KERNEL set start=90,end=99 where correlationId=1');c.commit();c.close();q=audit(f,o);self.assertNotEqual(q['status'],'failed');self.assertEqual(q['gpu_end_before_cpu_api_start_count'],1)
 def test_overlapping_gpu_work_is_preserved_but_composition_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   f,o=test_o4.Trace().fixture(Path(d),'P');c=sqlite3.connect(f/'trace.sqlite');c.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL select start+1,end+1,globalPid,9999,deviceId,contextId,streamId,demangledName,gridX,gridY,gridZ,blockX,blockY,blockZ,dynamicSharedMemory from CUPTI_ACTIVITY_KIND_KERNEL limit 1');c.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME select start+1,end+1,globalTid,9999,nameId from CUPTI_ACTIVITY_KIND_RUNTIME limit 1');c.commit();c.close();q=audit(f,o);self.assertNotEqual(q['status'],'failed');self.assertFalse(q['composition_supported']);self.assertEqual(q['work_event_count'],46)
