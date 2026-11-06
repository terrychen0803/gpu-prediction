"""Offline regression: GB10's op140 pattern is not a cross-device invariant."""
import copy
import unittest

from o4_research.observe_memset import summarize_resnet_arguments


class ResnetMemsetTransferTests(unittest.TestCase):
    def operations(self):
        return [
            {
                "iteration": iteration,
                "op_index": index,
                "aten": "aten.conv2d.default" if index == 140 else "aten.relu_.default",
                "shapes": [[1, 16, 8, 8]],
                "dtypes": ["torch.float16"],
            }
            for iteration in range(3)
            for index in range(175)
        ]

    def observed(self, per_forward):
        return [
            {
                "scope": f"iteration{iteration}/op{index}/aten.conv2d.default",
                "count_bytes": str(count),
            }
            for iteration in range(3)
            for index, count in per_forward
        ]

    def test_3090_positive_only_is_observed_not_failed(self):
        rows = self.observed([(81, 256), (143, 128), (155, 128), (165, 128)])
        result = summarize_resnet_arguments(rows, self.operations())
        self.assertEqual(result["op140_direct_calls"], 0)
        self.assertFalse(result["op140_gb10_pattern_observed"])
        self.assertEqual(result["op140_comparison_status"], "device_specific_difference")
        self.assertEqual([x["calls"] for x in result["per_forward_memset"]], [4, 4, 4])

    def test_gb10_zero_byte_op140_is_preserved(self):
        rows = self.observed([(140, 0), (143, 128)])
        result = summarize_resnet_arguments(rows, self.operations())
        self.assertEqual(result["op140_count_bytes"], [0, 0, 0])
        self.assertTrue(result["op140_gb10_pattern_observed"])

    def test_aten_signature_change_rejected(self):
        operations = self.operations()
        operations[175]["aten"] = "aten.add.Tensor"
        with self.assertRaises(ValueError):
            summarize_resnet_arguments(self.observed([(81, 256)]), operations)

    def test_unscoped_callback_rejected(self):
        rows = self.observed([(81, 256)])
        rows.append({"scope": "outside", "count_bytes": "0"})
        with self.assertRaises(ValueError):
            summarize_resnet_arguments(rows, self.operations())

    def test_different_memset_distribution_rejected(self):
        rows = self.observed([(81, 256)])
        rows[-1]["count_bytes"] = "128"
        with self.assertRaises(ValueError):
            summarize_resnet_arguments(rows, self.operations())


if __name__ == "__main__":
    unittest.main()
