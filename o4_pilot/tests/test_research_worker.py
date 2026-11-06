import subprocess,sys,unittest
class WorkerBootstrapTests(unittest.TestCase):
 def test_legacy_import_path_in_fresh_worker(self):
  p=subprocess.run([sys.executable,'-c','from o4_iteration import common; from inference_pilot.benchmark import precision_policy; from o1_operator.worker import reset_cuda_context_for_trace'],capture_output=True,text=True)
  self.assertEqual(p.returncode,0,p.stderr)
if __name__=='__main__':unittest.main()
