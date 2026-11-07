import copy,json,tempfile,unittest
from pathlib import Path
import torch
from o4_i3_v2.protocol import protocol,validate_protocol,warmup_stable,distribution,measured_values,schedule
from o4_i3_v2.assets import configurations,lookup,generate,verify,load_tensor,operation,logical_tensor
from o4_i3_v2.collect import code_index,verify_code,summarize

class ProtocolTests(unittest.TestCase):
    def test_smoke_cannot_be_relabelled_formal(self):
        a=protocol('smoke');b=protocol('formal');self.assertNotEqual(a['protocol_id'],b['protocol_id'])
        self.assertEqual((b['independent_process_repeats'],b['measured_iterations'],b['transition_iterations']),(3,40,5))
        a['kind']='formal'
        with self.assertRaises(ValueError):validate_protocol(a)
    def test_warmup_three_windows_and_minimum(self):
        p=protocol('formal');self.assertFalse(warmup_stable([1.]*59,p));self.assertTrue(warmup_stable([1.]*60,p))
        self.assertFalse(warmup_stable([1.]*20+[2.]*20+[1.]*20,p))
    def test_invalid_warmup_rejected(self):
        with self.assertRaises(ValueError):warmup_stable([float('nan')]*60,protocol('formal'))
    def test_distribution_linear_quantiles(self):
        a=distribution([1,2,3,4]);self.assertAlmostEqual(a['p10_us'],1.3);self.assertAlmostEqual(a['p90_us'],3.7)
        self.assertEqual(a['median_us'],2.5)
    def test_nonpositive_sample_rejected(self):
        for v in [[],[0],[1,float('inf')]]:
            with self.assertRaises(ValueError):distribution(v)
    def test_duplicate_or_missing_iterations_rejected(self):
        p=protocol('formal');rows=[{'stage':'measure','iteration':i,'elapsed_us':1,'status':'completed'} for i in range(40)]
        self.assertEqual(len(measured_values(rows,p)),40);rows[-1]['iteration']=0
        with self.assertRaises(ValueError):measured_values(rows,p)
        with self.assertRaises(ValueError):measured_values(rows[:-1],p)
    def test_schedule_deterministic_and_independent_cells(self):
        ids=['a','b','c'];p=protocol('formal');a=schedule(ids,p)
        self.assertEqual(a,schedule(ids,p));self.assertEqual(len(a),27)
        self.assertEqual(len({r['folder'] for r in a}),27)
        for rep,mode in [(1,'GI'),(2,'P'),(3,'R')]:self.assertEqual(next(j for j in a if j['repeat']==rep)['mode'],mode)
    def test_batch_cap_and_duplicate_configs_rejected(self):
        for c in [[],['a','a'],[str(i) for i in range(13)]]:
            with self.assertRaises(ValueError):schedule(c,protocol('formal'))

class AssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)
    def test_catalog_exact_174_and_sentinels_train(self):
        cs=configurations();self.assertEqual(len(cs),174)
        for f in ['batchnorm_eval','gemm','bilinear_resize']:self.assertEqual(lookup('I3_'+f+'_00_base')['split_B'],'train')
    def test_all_174_CPU_shapes_and_no_mutation(self):
        with torch.inference_mode():
            for c in configurations():
                tensors={n:load_tensor(torch,logical_tensor(torch,c,n).float(),s,'cpu') for n,s in c['definition']['tensors'].items()}
                before={n:t.clone() for n,t in tensors.items()};out=operation(torch,c,tensors)
                self.assertEqual(list(out.shape),c['definition']['output_shape'],c['id']);self.assertTrue(torch.isfinite(out).all())
                for n,t in tensors.items():self.assertTrue(torch.equal(t,before[n]),(c['id'],n))
    def test_control_logical_values_equal_despite_layout(self):
        a=lookup('I3_conv_00_base');b=lookup('I3_conv_00_control2')
        va=logical_tensor(torch,a,'x');vb=logical_tensor(torch,b,'x');self.assertTrue(torch.equal(va,vb))
        t=load_tensor(torch,vb,b['definition']['tensors']['x'],'cpu')
        self.assertEqual(list(t.stride()),b['definition']['tensors']['x']['stride']);self.assertFalse(t.is_contiguous())
    def test_transpose_view_retains_values(self):
        c=lookup('I3_gemm_00_control1');v=logical_tensor(torch,c,'weight');t=load_tensor(torch,v,c['definition']['tensors']['weight'],'cpu')
        self.assertTrue(torch.equal(v,t));self.assertFalse(t.is_contiguous())
    def test_generation_reproducible_and_hash_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);a=generate(root/'a',['I3_gemm_00_base']);b=generate(root/'b',['I3_gemm_00_base']);self.assertEqual(a,b)
            self.assertEqual(verify(root/'a/ASSETS.json')['operand_files'],2)
            f=root/'a'/a['configurations'][0]['operands']['x']['path'];f.write_bytes(f.read_bytes()[:-1]+b'X')
            with self.assertRaises(ValueError):verify(root/'a/ASSETS.json')
    def test_criteria_cannot_be_relaxed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'assets';m=generate(root,['I3_gemm_00_base']);m['configurations'][0]['criteria']['relative_l2_max']=1
            (root/'ASSETS.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):verify(root/'ASSETS.json')
    def test_reject_asset_path_escape(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'assets';m=generate(root,['I3_gemm_00_base']);m['configurations'][0]['operands']['x']['path']='../outside.bin'
            (root/'ASSETS.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):verify(root/'ASSETS.json')
    def test_unknown_config_rejected(self):
        with self.assertRaises(ValueError):lookup('not_a_config')

class ProvenanceTests(unittest.TestCase):
    def test_code_mutation_detected_pycache_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'a.py').write_text('a');before=code_index(root)
            (root/'__pycache__').mkdir();(root/'__pycache__/a.pyc').write_bytes(b'bytecode');verify_code(root,before)
            (root/'a.py').write_text('changed')
            with self.assertRaises(ValueError):verify_code(root,before)
    def test_incomplete_summary_cannot_claim_stability(self):
        with tempfile.TemporaryDirectory() as d:
            inv={'protocol':protocol('formal'),'configs':['a'],'jobs':[{'config_id':'a','mode':m,'repeat':r,'status':'not_started'} for m in ['GI','P','R'] for r in [1,2,3]]}
            summarize(Path(d),inv);self.assertTrue(all(g['stability_status']=='review_required' for g in inv['mode_groups']))
            self.assertFalse(inv['P_R_pairs'])

if __name__=='__main__':unittest.main()
