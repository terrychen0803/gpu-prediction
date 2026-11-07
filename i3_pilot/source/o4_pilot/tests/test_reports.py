import unittest
from o4_iteration.report import category,signature
class Reports(unittest.TestCase):
 def test_linear_bias_not_silently_supported_as_gemm(self):
  self.assertEqual(category('aten.addmm.default'),'linear_bias');self.assertEqual(category('aten.mm.default'),'gemm');self.assertEqual(category('aten.native_batch_norm.default'),'batchnorm_eval')
 def test_alignment_ignores_duration_but_checks_geometry_and_name(self):
  a={'event_kind':'kernel','kernel_name':'x','grid_xyz':[2,1,1],'block_xyz':[32,1,1],'dynamic_shared_memory_bytes':0,'start_ns':1,'end_ns':2};b={**a,'start_ns':100,'end_ns':300}
  self.assertEqual(signature(a),signature(b));self.assertNotEqual(signature(a),signature({**b,'grid_xyz':[1,2,1]}));self.assertNotEqual(signature(a),signature({**b,'kernel_name':'changed'}))
 def test_memset_value_is_part_of_alignment(self):
  self.assertNotEqual(signature({'event_kind':'memset','bytes':64,'memset_value':0}),signature({'event_kind':'memset','bytes':64,'memset_value':1}))

 def test_legacy_composition_gate_derived_from_observed_events(self):
  from o4_iteration.report import composition_supported
  self.assertFalse(composition_supported({},[{'start_ns':0,'end_ns':10},{'start_ns':5,'end_ns':15}]))
  self.assertTrue(composition_supported({},[{'start_ns':0,'end_ns':5},{'start_ns':5,'end_ns':10}]))
  self.assertFalse(composition_supported({'composition_supported':False},[]))
