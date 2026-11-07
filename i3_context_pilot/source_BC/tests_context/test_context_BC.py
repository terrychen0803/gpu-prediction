import json,tempfile,unittest,copy
from pathlib import Path
import torch
from o4_i3_context.assets import generate,derive,verify,Adapter,plan,op
from o4_i3_context.protocol import protocol,schedule
class BCContextTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();r=Path(cls.tmp.name);generate(r/'base');derive(r/'base/BASE.json',r/'local','fixture',device='cpu');cls.m=r/'local/ASSETS.json'
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def test_count_and_single_scope(self):
  self.assertEqual(len(plan()['work_items']),26);self.assertEqual(sum(len(g['nodes']) for g in plan()['graphs']),18);self.assertEqual({g['scale'] for g in plan()['graphs']},{0})
 def test_new_version_and_budget(self):
  p=protocol('formal');self.assertIn('BC',p['protocol_id']);self.assertEqual((p['independent_process_repeats'],p['measured_iterations']),(3,40))
 def test_all26_correctness_FP32_and_exact_native_inputs(self):
  with torch.inference_mode():
   for w in plan()['work_items']:
    a=Adapter(torch,self.m,w['id'],device='cpu');q=a.correctness();self.assertTrue(q['passed'],w['id']);self.assertTrue(all(c['native_D_inputs_exact'] and c['native_D_output_exact'] for c in q['positions']))
 def test_depthwise_groups_and_weight_contract(self):
  g=next(g for g in plan()['graphs'] if g['family']=='depthwise_pointwise');n=g['nodes'][0];self.assertEqual(n['parameters']['groups'],n['input_shapes'][0][1]);self.assertEqual(n['parameters']['weight_shape'][1],1)
 def test_SE_broadcast_uses_original_external(self):
  with torch.inference_mode():
   a=Adapter(torch,self.m,'CTX_se_gate_0',device='cpu');a.forward();self.assertIs(a.used['scale'][0],a.b['external/x']);self.assertEqual(list(a.used['scale'][1].shape)[-2:],[1,1])
 def test_resize_concat_has_two_inputs(self):
  with torch.inference_mode():
   a=Adapter(torch,self.m,'CTX_resize_concat_conv_0',device='cpu');a.forward();self.assertEqual(len(a.used['cat']),2);self.assertIs(a.used['cat'][0],a.retained[0]);self.assertIs(a.used['cat'][1],a.b['external/skip'])
 def test_attention_matmul_softmax_order_and_dtype(self):
  with torch.inference_mode():
   a=Adapter(torch,self.m,'CTX_attention_eager_0',device='cpu');a.forward();self.assertEqual([n['op'] for n in a.graph['nodes']],['bmm','mul_scalar','softmax','bmm']);self.assertEqual(a.retained[2].dtype,torch.float16);self.assertTrue(torch.allclose(a.retained[2].float().sum(-1),torch.ones_like(a.retained[2].float().sum(-1)),atol=.001))
 def test_I_no_data_edge_but_same_native_input(self):
  with torch.inference_mode():
   a=Adapter(torch,self.m,'IND_attention_eager_0',device='cpu');a.forward();self.assertIs(a.used['scale'][0],a.inputs['scale'][0]);self.assertIsNot(a.used['scale'][0],a.retained[0]);self.assertEqual(len(a.retained),4)
 def test_layout_and_position_shape_mutation_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   import shutil
   r=Path(d)/'local';shutil.copytree(self.m.parent,r);p=r/'ASSETS.json';m=json.loads(p.read_text());m['graphs'][0]['nodes']['dw']['inputs'][0]['shape'][0]=9;p.write_text(json.dumps(m))
   with self.assertRaises(ValueError):verify(p)
 def test_corrupt_file_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   import shutil
   r=Path(d)/'local';shutil.copytree(self.m.parent,r);m=json.loads((r/'ASSETS.json').read_text());p=r/m['graphs'][0]['nodes']['dw']['output']['path'];p.write_bytes(b'bad')
   with self.assertRaises(ValueError):verify(r/'ASSETS.json')
 def test_new_bias_parameters_present(self):
  m=json.loads((self.m.parent/'base/BASE.json').read_text());se=next(e for e in m['graphs'] if e['graph']['family']=='se_gate');self.assertIn('node/reduce/bias',se['operands']);self.assertIn('node/expand/bias',se['operands'])
 def test_unknown_operation_rejected(self):
  with self.assertRaises(ValueError):op(torch,{'op':'unknown','id':'z','parameters':{}},[],{})
 def test_exact_hardswish_taxonomy(self):
  from o4_i3.taxonomy import classify
  self.assertEqual(classify('aten.hardswish.default')['subtype'],'hardswish')
 def test_CPU_fixture_not_GPU(self):self.assertEqual(json.loads(self.m.read_text())['derivation']['device'],'cpu')
 def test_schedule_cannot_accept26(self):
  with self.assertRaises(ValueError):schedule([w['id'] for w in plan()['work_items']],protocol('formal'))
if __name__=='__main__':unittest.main()
