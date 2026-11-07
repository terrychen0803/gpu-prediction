import copy,json,tempfile,unittest
from pathlib import Path
import torch
from o4_i3_context.assets import generate,derive,verify,verify_base,Adapter,plan,execute,load,work
from o4_i3_context.protocol import protocol,validate_protocol,schedule

def prepared(root):
 generate(root/'base');derive(root/'base/BASE.json',root/'local','fixture',device='cpu');return root/'local/ASSETS.json'
class ContextTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name);cls.manifest=prepared(cls.root)
 @classmethod
 def tearDownClass(cls):cls.tmp.cleanup()
 def test_14_positions_not_deduplicated(self):
  ws=plan()['work_items'];self.assertEqual(len(ws),14);self.assertEqual(sum(w['kind']=='isolated_position' for w in ws),10)
 def test_contract_scope_new_version(self):
  p=protocol('formal');self.assertIn('context_DIS',p['protocol_id']);self.assertEqual((p['measured_iterations'],p['independent_process_repeats']),(40,3));self.assertIn('all node outputs',p['scope'])
 def test_protocol_mutation_rejected(self):
  p=protocol('formal');p['independent_process_repeats']=1
  with self.assertRaises(ValueError):validate_protocol(p)
 def test_oversize_batch_rejected(self):
  with self.assertRaises(ValueError):schedule([x['id'] for x in plan()['work_items']],protocol('formal'))
 def test_all14_CPU_parity_state_and_native_inputs(self):
  with torch.inference_mode():
   for w in plan()['work_items']:
    a=Adapter(torch,self.manifest,w['id'],device='cpu');q=a.correctness();self.assertTrue(q['passed'],w['id']);self.assertTrue(all(x['native_D_inputs_exact'] and x['native_D_output_exact'] for x in q['positions']))
 def test_D_residual_consumes_skip_input(self):
  with torch.inference_mode():
   a=Adapter(torch,self.manifest,'CTX_residual_0',device='cpu');a.forward();self.assertIs(a.used['a'][1],a.b['external/x']);self.assertIs(a.used['c2'][0],a.retained[2]);self.assertEqual(len(a.retained),7)
 def test_I_does_not_consume_previous_output(self):
  with torch.inference_mode():
   a=Adapter(torch,self.manifest,'IND_residual_0',device='cpu');a.forward();self.assertIs(a.used['c2'][0],a.inputs['c2'][0]);self.assertIsNot(a.used['c2'][0],a.retained[2]);self.assertEqual(len(a.retained),7)
 def test_outputs_retained_until_prepare(self):
  with torch.inference_mode():
   a=Adapter(torch,self.manifest,'CTX_conv_bn_relu_0',device='cpu');out=a.forward();self.assertEqual(len(a.retained),3);self.assertIs(out,a.retained[-1]);a.prepare();self.assertEqual(a.retained,[])
 def test_S_position_uses_same_local_input_and_weight(self):
  with torch.inference_mode():
   a=Adapter(torch,self.manifest,'ISO_residual_0_c2',device='cpu');q=a.correctness();self.assertEqual(len(q['positions']),1);self.assertEqual(q['positions'][0]['node_id'],'c2')
 def test_shape_match_alone_not_enough_for_edge_parity(self):
  with torch.inference_mode():
   a=Adapter(torch,self.manifest,'IND_residual_0',device='cpu');a.inputs['c2'][0].add_(1)
   with self.assertRaises(RuntimeError):a.correctness()
 def test_payload_corruption_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   import shutil
   p=Path(d)/'local';shutil.copytree(self.manifest.parent,p);m=json.loads((p/'ASSETS.json').read_text());f=p/m['graphs'][0]['nodes']['c']['inputs'][0]['path'];f.write_bytes(f.read_bytes()[:-1]+b'X')
   with self.assertRaises(ValueError):verify(p/'ASSETS.json')
 def test_wrong_edge_contract_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   import shutil
   p=Path(d)/'local';shutil.copytree(self.manifest.parent,p);m=json.loads((p/'ASSETS.json').read_text());m['graphs'][1]['nodes']['a']['input_edges']=['n2','n2'];(p/'ASSETS.json').write_text(json.dumps(m))
   with self.assertRaises(ValueError):verify(p/'ASSETS.json')
 def test_base_path_escape_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   import shutil
   p=Path(d)/'local';shutil.copytree(self.manifest.parent,p);m=json.loads((p/'ASSETS.json').read_text());m['base_manifest']='../base/BASE.json';(p/'ASSETS.json').write_text(json.dumps(m))
   with self.assertRaises((ValueError,FileNotFoundError)):verify(p/'ASSETS.json')
 def test_CPU_fixture_cannot_claim_GPU(self):
  m=json.loads(self.manifest.read_text());self.assertEqual(m['derivation']['device'],'cpu');self.assertIn('NOT_GPU',m['derivation']['gpu'])
 def test_unknown_work_rejected(self):
  with self.assertRaises(ValueError):work('not-a-context')
if __name__=='__main__':unittest.main()
