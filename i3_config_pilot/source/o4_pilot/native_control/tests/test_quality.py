import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from quality import interval_closure,overlapping_pairs,partition_supported,gate,strict_gate,GATES
class QualityTests(unittest.TestCase):
 def test_disjoint(self):
  q=interval_closure([(0,4),(7,10)]);self.assertEqual((q['span_ns'],q['positive_gap_ns'],q['union_overlap_ns'],q['closure_error_ns']),(10,3,0,0))
 def test_nested(self):
  q=interval_closure([(0,20),(2,4),(6,8)]);self.assertEqual((q['positive_gap_ns'],q['union_overlap_ns'],q['closure_error_ns']),(0,4,0))
 def test_triple_overlap_is_not_pair_sum(self):
  ev=[dict(uid=str(i),start=a,end=b) for i,(a,b) in enumerate([(0,10),(2,8),(4,6)])];q=interval_closure([(x['start'],x['end']) for x in ev]);self.assertEqual(q['union_overlap_ns'],8);self.assertEqual(sum(x[2] for x in overlapping_pairs(ev)),10);self.assertEqual(q['closure_error_ns'],0)
 def test_internal_gap_not_added_twice(self):
  q=interval_closure([(0,10),(12,15)]);self.assertEqual(q['positive_gap_ns'],2)
 def test_touching(self):self.assertTrue(interval_closure([(0,2),(2,3)])['nonoverlap_observed'])
 def test_invalid_or_empty(self):
  for x in ([],[(1,1)],[(2,1)],[(0.0,2.0)]):
   with self.assertRaises(ValueError):interval_closure(x)
 def test_partition_ignores_temporal_overlap(self):self.assertTrue(partition_supported(['a','a','b','b']))
 def test_interleaved_partition_rejected(self):self.assertFalse(partition_supported(['a','b','a']))
 def test_unassigned_rejected(self):self.assertFalse(partition_supported(['a',None]))
 def test_warning_prevents_strict_even_with_closure(self):
  gs={k:gate('passed',['fixture'],'fixture') for k in GATES};gs['reference_semantic_alignment']=gate('warning_accepted',['fixture'],'NVTX');self.assertEqual(strict_gate(gs,['fixture'])['status'],'unverified')
 def test_missing_work_prevents_strict(self):
  gs={k:gate('passed',['fixture'],'fixture') for k in GATES};gs['submitted_work_coverage']=gate('unverified',['fixture'],'unknown zero-byte');self.assertEqual(strict_gate(gs,['fixture'])['status'],'unverified')
 def test_pair_intersection_nested(self):
  e=[dict(uid='a',start=0,end=20),dict(uid='b',start=2,end=4)];self.assertEqual(overlapping_pairs(e)[0][2],2)
if __name__=='__main__':unittest.main()
