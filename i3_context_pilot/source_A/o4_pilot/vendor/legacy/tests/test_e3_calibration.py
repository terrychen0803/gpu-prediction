"""Synthetic E3 split checks; these are not measured GPU results."""
import copy
import unittest

from e3_calibration.__main__ import evaluate


class E3SplitTests(unittest.TestCase):
    def test_held_out_g_label_cannot_change_its_prediction(self):
        features = []
        labels = []
        for device, scale in (('gb10', 1), ('rtx3090', 2)):
            for batch, span in ((1, 100), (8, 300), (32, 900)):
                cfg = f'r224_b{batch}'
                features.append({'label': device, 'config_id': cfg, 'gpu_span_us': span * scale,
                                 'trace_warning_runs': 0})
                labels.append({'label': device, 'config_id': cfg, 'g_us': span * scale * 1.2})
        splits, original = evaluate(features, labels)
        self.assertEqual(len(splits), 3)
        self.assertTrue(all(s['held_out_config'] not in s['training_configs'] for s in splits))
        changed = copy.deepcopy(labels)
        next(r for r in changed if r['label'] == 'gb10' and r['config_id'] == 'r224_b32')['g_us'] = 1e9
        _, altered = evaluate(features, changed)
        before = {(r['label'], r['held_out_config'], r['model']): r for r in original}
        after = {(r['label'], r['held_out_config'], r['model']): r for r in altered}
        for model in ('C0_gpu_span', 'C1_log_linear_ridge1'):
            key = ('gb10', 'r224_b32', model)
            self.assertEqual(before[key]['prediction_us'], after[key]['prediction_us'])
            self.assertNotEqual(before[key]['absolute_percentage_error'], after[key]['absolute_percentage_error'])
            other = ('rtx3090', 'r224_b32', model)
            self.assertEqual(before[other]['prediction_us'], after[other]['prediction_us'])


if __name__ == '__main__':
    unittest.main()
