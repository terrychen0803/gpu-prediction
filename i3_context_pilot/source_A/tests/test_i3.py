import copy,json,math,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from o4_i3.taxonomy import classify,registry
from o4_i3.inventory import ownership,union
from o4_i3.controlled import configs,validate_config,operation,verify_assets
from o4_i3.dispatch import arguments,inventory
from o4_i3.splits import feasibility,signature

class TaxonomyTests(unittest.TestCase):
    def test_resize_collision_and_matmul(self):
        self.assertEqual(classify('aten.upsample_bilinear2d.vec')['subtype'],'resize_bilinear')
        self.assertEqual(classify('aten.linear.default')['subtype'],'linear')
        self.assertEqual(classify('aten.matmul.default')['subtype'],'matmul_unresolved')
    def test_unknown_not_substring(self):
        for name in ['x.linear.something','aten.conv2d.not_an_overload','aten.upsample_bilinear2d.fake']:
            self.assertFalse(classify(name)['mapped'])
    def test_all_observed_70_names_explicit(self):
        import csv
        p=Path(__file__).resolve().parent/'fixtures/observed_ATen_names.json'
        names=set(json.loads(p.read_text()))
        self.assertEqual(len(names),70)
        self.assertTrue(all(classify(x)['mapped'] for x in names))
    def test_semantic_params_still_needed(self):
        self.assertIn('groups',classify('aten.conv2d.default')['required_semantic_attributes'])
        self.assertIn('stride',classify('aten.conv2d.default')['common_tensor_attributes'])

class OwnershipTests(unittest.TestCase):
    def test_missing_and_duplicate_distinct(self):
        self.assertEqual(ownership([[0,1],[1]],3),{'unassigned':[2],'multiple_owners':[1]})
    def test_bad_reference_index_rejected(self):
        for ids in [[[-1]],[[3]],[[True]]]:
            with self.assertRaises(ValueError):ownership(ids,3)
    def test_triple_overlap_union(self):
        self.assertEqual(union([(0,5),(1,4),(2,7),(10,12)]),9)

class ControlledTests(unittest.TestCase):
    def test_configs_frozen_and_holdout_is_not_repeats(self):
        cs=configs();self.assertEqual(len(cs),18);self.assertEqual(len({c['id'] for c in cs}),18)
        for c in cs:validate_config(c)
        bad=copy.deepcopy(cs[0]);bad['parameters']['groups']=3
        with self.assertRaises(ValueError):validate_config(bad)
    def test_declared_output_shapes_CPU(self):
        torch.set_num_threads(1)
        with torch.inference_mode():
            for c in configs():
                t={k:torch.randn(s) for k,s in c['operands'].items()}
                if 'running_var' in t:t['running_var']=t['running_var'].abs()+.8
                before={k:v.clone() for k,v in t.items()}
                out=operation(torch,c,t)
                self.assertEqual(list(out.shape),c['output_shape'],c['id'])
                self.assertTrue(bool(torch.isfinite(out).all()))
                for k,v in t.items():self.assertTrue(torch.equal(v,before[k]),(c['id'],k))
    def test_missing_asset_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ASSETS.json';p.write_text(json.dumps({'version':'wrong','configurations':[]}))
            with self.assertRaises(ValueError):verify_assets(p)
    def test_tolerance_cannot_be_relaxed_in_manifest(self):
        from o4_i3.controlled import VERSION
        cs=configs()
        for c in cs:c['correctness_criteria']={'relative_l2_max':100}
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ASSETS.json';p.write_text(json.dumps({'version':VERSION,'configurations':cs}))
            with self.assertRaisesRegex(ValueError,'criteria'):verify_assets(p)

class DispatchTests(unittest.TestCase):
    def test_bound_default_alpha_and_stride(self):
        x=torch.zeros(2,3).t();r=arguments(torch.ops.aten.add.Tensor,(x,x),{},torch)
        self.assertEqual(r['alpha'],1);self.assertEqual(r['self']['stride'],list(x.stride()))
        self.assertNotIn('values',r['self'])
    def test_active_capture_complete_and_output_parity(self):
        x=torch.randn(1,3,8,8);w=torch.randn(4,3,3,3);ref=torch.nn.functional.conv2d(x,w,stride=2,padding=1)
        inv=inventory(torch)
        with patch('torch.cuda.nvtx.range_push'),patch('torch.cuda.nvtx.range_pop'):
            with inv:
                inv.begin('measure',0);out=torch.nn.functional.conv2d(x,w,stride=2,padding=1);inv.end()
        self.assertTrue(torch.equal(ref,out));self.assertEqual(len(inv.rows),1)
        row=inv.rows[0];args=row['bound_arguments'];self.assertEqual(args['stride'],[2,2]);self.assertEqual(args['padding'],[1,1]);self.assertEqual(args['groups'],1)
        self.assertIn('output_metadata',row);self.assertEqual(row['metadata_role'],'R_only_never_predictor')
        json.dumps(row,allow_nan=False)

class SplitTests(unittest.TestCase):
    def test_same_config_repeats_cannot_create_split(self):
        c=configs()[0];a=copy.deepcopy(c);a['id']='repeat_on_other_device'
        f=feasibility([c,a]);self.assertEqual(len(f['groups']),1)
        self.assertFalse(f['training_ready']);self.assertIsNone(f['split_assignment'])
    def test_two_configurations_insufficient(self):
        f=feasibility(configs());self.assertTrue(all(v==2 for v in f['groups_per_subtype'].values()))
        self.assertEqual(f['status'],'insufficient_for_train_validation_test')
    def test_config_semantics_affect_group(self):
        c=configs()[0];a=copy.deepcopy(c);a['parameters']['stride']=[1,1]
        self.assertNotEqual(signature(c),signature(a))

if __name__=='__main__':unittest.main()
