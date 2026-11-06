import importlib.util,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('runner',Path(__file__).with_name('run_s3.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
class EligibilityTests(unittest.TestCase):
    def fixture(self):return {'status':'completed_development_collection','configs':['a'],'jobs':[{'config_id':'a','status':'completed','mode':mode,'repeat':1,'monitor_review':False} for mode in ['GI','P','R']],'P_R_pairs':[{'config_id':'a','measured_exact':3,'all_windows_exact':True}]}
    def test_healthy(self):self.assertEqual(m.eligible(self.fixture()),['a'])
    def test_failed_worker(self):
        a=self.fixture();a['jobs'][0]['status']='failed';self.assertEqual(m.eligible(a),[])
    def test_duplicated_mode_rejected(self):
        a=self.fixture();a['jobs'][0]['mode']='P';self.assertEqual(m.eligible(a),[])
    def test_missing_monitor_rejected(self):
        a=self.fixture();del a['jobs'][0]['monitor_review'];self.assertEqual(m.eligible(a),[])
    def test_foreign_monitor_rejected(self):
        a=self.fixture();a['jobs'][0]['monitor_review']=True;self.assertEqual(m.eligible(a),[])
    def test_PR_mismatch_rejected(self):
        a=self.fixture();a['P_R_pairs'][0]['measured_exact']=2;self.assertEqual(m.eligible(a),[])
    def test_duplicate_pair_rejected(self):
        a=self.fixture();a['P_R_pairs']*=2;self.assertEqual(m.eligible(a),[])
    def test_paused_never_formal(self):
        a=self.fixture();a['status']='paused_GPU_guard_or_foreign_work';self.assertEqual(m.eligible(a),[])
if __name__=='__main__':unittest.main()
