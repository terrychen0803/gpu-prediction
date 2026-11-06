"""Regression probes for diagnostic exceptions, required features, and return values."""
import json,sqlite3,tempfile,unittest
from pathlib import Path
from o4_research.trace_contract import diagnostic_policy,feature_contract,api_return_audit,KNOWN_NVTX
from o4_research.audit_v5 import audit
from o4_research.common import write
from o4_research.provenance import source_hashes

BACKEND={'severity':1,'text':'CUDA legacy/software instrumented trace was collected.'}

def fixture(root,warning=KNOWN_NVTX,return_value=0,geometry=True):
    write(root/'request.json',{'mode':'P','protocol':{'transition_iterations':5,'measured_iterations':40},'trace_backend':'cuda-sw'})
    write(root/'environment.json',{'pid':7})
    c=sqlite3.connect(root/'trace.sqlite')
    c.executescript('''CREATE TABLE META_DATA_EXPORT(name TEXT,value TEXT);
    CREATE TABLE StringIds(id INTEGER,value TEXT);CREATE TABLE DIAGNOSTIC_EVENT(severity INTEGER,text TEXT);
    CREATE TABLE NVTX_EVENTS(globalTid INTEGER,text TEXT,textId INTEGER,start INTEGER,end INTEGER);
    CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(globalTid INTEGER,nameId INTEGER,correlationId INTEGER,start INTEGER,end INTEGER,returnValue INTEGER);
    CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(globalPid INTEGER,correlationId INTEGER,start INTEGER,end INTEGER,deviceId INTEGER,contextId INTEGER,streamId INTEGER);''')
    fields=['demangledName','gridX','gridY','gridZ','blockX','blockY','blockZ','dynamicSharedMemory'] if geometry else []
    for field in fields:c.execute('ALTER TABLE CUPTI_ACTIVITY_KIND_KERNEL ADD COLUMN '+field+' INTEGER')
    c.execute('INSERT INTO META_DATA_EXPORT VALUES(?,?)',('EXPORT_SCHEMA_VERSION','3.31.0'))
    c.executemany('INSERT INTO StringIds VALUES(?,?)',[(1,'cudaLaunchKernel'),(2,'known_kernel')])
    c.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(?,?)',(2,warning));c.execute('INSERT INTO DIAGNOSTIC_EVENT VALUES(?,?)',(1,BACKEND['text']))
    for k in range(45):
        stage='transition' if k<5 else 'measure';index=k if k<5 else k-5;t=1000*k
        c.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?,?)',((7<<24)+1,f'o4.iter/{stage}/{index:06d}',None,t,t+900))
        c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?,?)',((7<<24)+1,1,k,t+100,t+200,return_value))
        values=[7<<24,k,t+210,t+300,0,0,1]+([2,1,1,1,32,1,1,0] if geometry else [])
        c.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES('+','.join('?' for _ in values)+')',values)
    c.commit();c.close()

class PolicyTests(unittest.TestCase):
    def test_known_nvtx_only_exception(self):
        r=diagnostic_policy([BACKEND,{'severity':2,'text':KNOWN_NVTX}])
        self.assertFalse(r['blockers']);self.assertEqual(len(r['accepted_warnings']),1)
    def test_cuda_drop_blocks_even_at_info_severity(self):
        for severity in (1,2,3):
            r=diagnostic_policy([BACKEND,{'severity':severity,'text':'Incomplete CUPTI Events dropped'}]);self.assertTrue(r['blockers'])
    def test_unknown_warning_is_not_nvtx_exception(self):
        self.assertTrue(diagnostic_policy([BACKEND,{'severity':2,'text':'Unexpected warning'}])['blockers'])
    def test_error_using_nvtx_text_is_not_exception(self):
        self.assertTrue(diagnostic_policy([BACKEND,{'severity':3,'text':KNOWN_NVTX}])['blockers'])
    def test_backend_required_and_conflicts_block(self):
        self.assertTrue(diagnostic_policy([])['blockers'])
        self.assertTrue(diagnostic_policy([BACKEND,{'severity':1,'text':'Hardware tracing used for CUDA tracing.'}])['blockers'])
    def test_return_value_checked_with_explicit_query_exception(self):
        self.assertTrue(api_return_audit([{'name':'cudaLaunchKernel','returnValue':1}]))
        self.assertTrue(api_return_audit([{'name':'cudaLaunchKernel'}]))
        self.assertFalse(api_return_audit([{'name':'cudaEventQuery','returnValue':34}]))
        self.assertTrue(api_return_audit([{'name':'cudaMemsetAsync','returnValue':34}]))
    def test_missing_geometry_not_silently_valid(self):
        self.assertTrue(feature_contract({'CUPTI_ACTIVITY_KIND_KERNEL':[]},[]))
    def test_full_source_hash_covers_shared_dependencies(self):
        keys=source_hashes();self.assertIn('o4_research/common.py',keys);self.assertIn('o4_iteration/adapters.py',keys);self.assertIn('vendor/legacy/inference_pilot/benchmark.py',keys)
    def test_source_mutation_detected_and_outputs_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);(r/'common.py').write_text('before');a=source_hashes(r);(r/'common.py').write_text('after');self.assertNotEqual(a,source_hashes(r))
            a=source_hashes(r);(r/'analysis_runs').mkdir();(r/'analysis_runs/ignored.json').write_text('{}');self.assertEqual(a,source_hashes(r))

class SQLiteOverlayTests(unittest.TestCase):
    def run_fixture(self,**kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);fixture(r,**kwargs);before=(r/'trace.sqlite').read_bytes();result=audit(r,r/'audit')
            self.assertEqual(before,(r/'trace.sqlite').read_bytes());self.assertEqual(result['iteration_count'],45)
            self.assertEqual(result['engineering_acceptance']['strict_verified'],0)
            self.assertNotIn('engineering_acceptance',json.loads((r/'audit/SUMMARY.json').read_text()))
            return result['engineering_acceptance']
    def test_45_window_known_warning_accepted(self):self.assertTrue(self.run_fixture()['accepted'])
    def test_45_window_dropped_blocked(self):self.assertFalse(self.run_fixture(warning='Incomplete CUPTI Events dropped')['accepted'])
    def test_45_window_failed_api_blocked(self):self.assertFalse(self.run_fixture(return_value=1)['accepted'])
    def test_45_window_absent_geometry_blocked(self):self.assertFalse(self.run_fixture(geometry=False)['accepted'])

if __name__=='__main__':unittest.main()

class TimingQualityTests(unittest.TestCase):
    def test_missing_warmup_rejected(self):
        from o4_research.run_formal import measured_median
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);write(r/'quality.json',{'status':'completed','correctness':{'passed':True}})
            with self.assertRaises(ValueError):measured_median(r,40)
    def test_duplicate_iteration_ids_rejected(self):
        from o4_research.run_formal import measured_median
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);write(r/'quality.json',{'status':'completed','correctness':{'passed':True},'warmup':{'passed':True}})
            (r/'iterations.csv').write_text('stage,iteration,elapsed_us\n'+''.join('measure,0,10\n' for _ in range(40)))
            with self.assertRaises(ValueError):measured_median(r,40)
