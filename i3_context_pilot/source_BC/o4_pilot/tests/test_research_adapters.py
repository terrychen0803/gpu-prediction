import unittest
from o4_research.adapters import SPECS,require_a,Adapter
from o4_research.audit import safe
from o4_research.quality_v4 import interval_closure,partition_supported
class AdapterContractTests(unittest.TestCase):
 def test_exact_seven_A(self):self.assertEqual(set(SPECS),{'RN-A','MV-A','SW-A','UN-A','YO-A','QP-A','QD-A'})
 def test_no_silent_substitution(self):
  for name in ['RN-B','MV-small','YOLO11n','QP-B','QD-B']:
   with self.assertRaises(ValueError):require_a(name)
 def test_numeric_predictor_poison(self):
  e={'start_ns':1,'end_ns':9,'kernel_name':'SECRET','aten':'oracle','shape':[1024,1024],'workload':'RN-A','target_GI':123,'op_scope':'oracle'};x=safe(e,0)
  for k in ['kernel_name','aten','shape','workload','target_GI','op_scope']:self.assertNotIn(k,x)
 def test_union_triple(self):
  x=interval_closure([(0,10),(2,8),(4,6)]);self.assertEqual(x['union_overlap_ns'],8);self.assertEqual(x['closure_error_ns'],0)
 def test_index_partition_is_not_temporal_overlap(self):self.assertTrue(partition_supported(['a','a','b','b']))
 def test_qwen_stale_cache_refused(self):
  class Cache:
   def get_seq_length(self):return 129
  for name in ['QP-A','QD-A']:
   x=Adapter.__new__(Adapter);x.name=name;x.cache=Cache();x.torch=None
   with self.assertRaisesRegex(RuntimeError,'Cache reuse'):x.forward()
 def test_unet_contract_CPU(self):
  import torch
  from o4_research.adapters import make_unet
  torch.set_num_threads(1);model=make_unet(torch).eval();convs=[m for m in model.modules() if isinstance(m,torch.nn.Conv2d)];self.assertEqual(len(convs),19);self.assertTrue(all(m.bias is not None for m in convs));self.assertFalse(any(isinstance(m,torch.nn.BatchNorm2d) for m in model.modules()))
  with torch.inference_mode():self.assertEqual(list(model(torch.zeros(1,1,256,256)).shape),[1,2,256,256])
if __name__=='__main__':unittest.main()
