import json,os,subprocess,tempfile,unittest
from pathlib import Path
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

 def test_retry_stale_busy_without_active_compute_process(self):
  readings=['NVIDIA GeForce RTX 4090, 83 %\\n','NVIDIA GeForce RTX 4090, 83 %\\n',
            'NVIDIA GeForce RTX 4090, 0 %\\n','NVIDIA GeForce RTX 4090, 0 %\\n']
  replies=[]
  for reading in readings:
   replies.extend([subprocess.CompletedProcess([],0,reading,''),
                   subprocess.CompletedProcess([],0,'','')])
  with tempfile.TemporaryDirectory() as d:
   with patch('o4_research.common.subprocess.run',side_effect=replies) as query,patch('o4_research.common.time.sleep') as sleep:
    gpu_guard(d,'4090',idle_wait_seconds=12,check_interval_seconds=3,idle_checks_required=2)
   evidence=json.loads((Path(d)/'gpu_guard.json').read_text())
   self.assertEqual(evidence['sample_count'],4)
   self.assertEqual(query.call_count,8)
   self.assertEqual(sleep.call_count,3)
   self.assertTrue(evidence['recovered_after_high_utilization'])
   self.assertEqual(evidence['blocked'],[])
 def test_persistently_busy_blocks_after_bounded_wait(self):
  replies=[]
  for _ in range(4):
   replies.extend([subprocess.CompletedProcess([],0,'NVIDIA GeForce RTX 4090, 83 %\\n',''),
                   subprocess.CompletedProcess([],0,'','')])
  with tempfile.TemporaryDirectory() as d:
   with patch('o4_research.common.subprocess.run',side_effect=replies),patch('o4_research.common.time.sleep') as sleep:
    with self.assertRaisesRegex(RuntimeError,'utilization exceeds5%'):
     gpu_guard(d,'4090',idle_wait_seconds=9,check_interval_seconds=3,idle_checks_required=2)
   evidence=json.loads((Path(d)/'gpu_guard.json').read_text())
   self.assertEqual(evidence['sample_count'],4)
   self.assertEqual(sleep.call_count,3)
   self.assertFalse(evidence['recovered_after_high_utilization'])
 def test_external_process_blocks_even_with_idle_retry_enabled(self):
  replies=[subprocess.CompletedProcess([],0,'NVIDIA GeForce RTX 4090, 83 %\\n',''),
           subprocess.CompletedProcess([],0,f'{os.getpid()}, python_fixture, 1 MiB\\n','')]
  with tempfile.TemporaryDirectory() as d:
   with patch('o4_research.common.subprocess.run',side_effect=replies) as query,patch('o4_research.common.time.sleep') as sleep:
    with self.assertRaisesRegex(RuntimeError,'Other GPU workload present'):
     gpu_guard(d,'4090',idle_wait_seconds=60,check_interval_seconds=3,idle_checks_required=2)
   self.assertEqual(query.call_count,2)
   sleep.assert_not_called()
 def test_single_idle_sample_is_insufficient_when_retry_policy_enabled(self):
  replies=[]
  for _ in range(2):
   replies.extend([subprocess.CompletedProcess([],0,'NVIDIA GeForce RTX 4090, 0 %\\n',''),
                   subprocess.CompletedProcess([],0,'','')])
  with tempfile.TemporaryDirectory() as d:
   with patch('o4_research.common.subprocess.run',side_effect=replies),patch('o4_research.common.time.sleep') as sleep:
    gpu_guard(d,'4090',idle_wait_seconds=6,check_interval_seconds=3,idle_checks_required=2)
   evidence=json.loads((Path(d)/'gpu_guard.json').read_text())
   self.assertEqual(evidence['sample_count'],2)
   self.assertEqual(sleep.call_count,1)

if __name__=='__main__':unittest.main()
