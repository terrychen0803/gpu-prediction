import os,subprocess,tempfile,unittest
from unittest.mock import patch
from o4_research.common import gpu_guard
class GuardTests(unittest.TestCase):
 def fake(self,name='NVIDIA GeForce RTX 4090',process=''):
  return [subprocess.CompletedProcess([],0,name+', 0 %\n',''),subprocess.CompletedProcess([],0,process,'')]
 def test_explicit_other_node_identity_supported(self):
  with tempfile.TemporaryDirectory() as d,patch('o4_research.common.subprocess.run',side_effect=self.fake()):gpu_guard(d,'4090')
 def test_wrong_node_identity_fails(self):
  with tempfile.TemporaryDirectory() as d,patch('o4_research.common.subprocess.run',side_effect=self.fake()):
   with self.assertRaises(RuntimeError):gpu_guard(d,'GB10')
 def test_non_desktop_workload_blocks_without_killing(self):
  process=f'{os.getpid()}, python_fixture, 1 MiB\n'
  with tempfile.TemporaryDirectory() as d,patch('o4_research.common.subprocess.run',side_effect=self.fake(process=process)):
   with self.assertRaises(RuntimeError):gpu_guard(d,'4090')
if __name__=='__main__':unittest.main()
