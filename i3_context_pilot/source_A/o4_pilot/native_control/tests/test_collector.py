import sys,tempfile,unittest,json,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from collect_controls import command
class FailureTests(unittest.TestCase):
 def test_failed_child_preserves_exit_and_command(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'child.log'
   with self.assertRaises(RuntimeError):command([sys.executable,'-c','raise SystemExit(7)'],p)
   q=json.loads(p.with_suffix('.log.command.json').read_text());self.assertEqual(q['returncode'],7);self.assertEqual(q['status'],'failed')
 def test_timeout_cleans_only_owned_child(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'child.log'
   with self.assertRaises(subprocess.TimeoutExpired):command([sys.executable,'-c','import time;time.sleep(10)'],p,timeout=.05)
   q=json.loads(p.with_suffix('.log.command.json').read_text());self.assertEqual(q['status'],'timeout_or_interrupted');self.assertLess(q['returncode'],0)
if __name__=='__main__':unittest.main()
