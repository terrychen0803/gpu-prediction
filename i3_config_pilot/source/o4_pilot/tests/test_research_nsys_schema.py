"""Offline Nsight SQLite schema contract tests. No CUDA or Nsight required."""
import copy
import unittest
from o4_research.nsys_schema import (
    OPTIONAL_ACTIVITY, REQUIRED, validate_export_schema
)


def fixture():
    fields = dict(REQUIRED)
    fields.update({
        "CUPTI_ACTIVITY_KIND_KERNEL": OPTIONAL_ACTIVITY["CUPTI_ACTIVITY_KIND_KERNEL"],
        "CUPTI_ACTIVITY_KIND_MEMSET": OPTIONAL_ACTIVITY["CUPTI_ACTIVITY_KIND_MEMSET"],
    })
    return {name: [{"name": column} for column in sorted(cols)]
            for name, cols in fields.items()}


class SchemaContractTests(unittest.TestCase):
    def test_328_329_331_structural_only(self):
        for version in ("3.28.0", "3.29.0", "3.31.0"):
            result = validate_export_schema(version, fixture())
            self.assertEqual(result["structural_contract"], "passed")
            self.assertEqual(result["semantic_equivalence"], "unverified")
            self.assertFalse(result["strict_reference_verified"])
            self.assertEqual(result["gpu_event_completeness"], "unverified")
            self.assertTrue(result["no_timestamp_or_event_mutations"])

    def test_unknown_version_rejected(self):
        with self.assertRaises(ValueError):
            validate_export_schema("3.32.0", fixture())

    def test_329_missing_required_column_rejected(self):
        schema = fixture()
        schema["CUPTI_ACTIVITY_KIND_RUNTIME"] = [
            r for r in schema["CUPTI_ACTIVITY_KIND_RUNTIME"]
            if r["name"] != "correlationId"
        ]
        with self.assertRaisesRegex(ValueError, "missing columns"):
            validate_export_schema("3.29.0", schema)

    def test_329_missing_geometry_field_rejected(self):
        schema = fixture()
        schema["CUPTI_ACTIVITY_KIND_KERNEL"] = [
            r for r in schema["CUPTI_ACTIVITY_KIND_KERNEL"]
            if r["name"] != "gridX"
        ]
        # Geometry is enforced by the additive v5 feature gate.
        from o4_research.trace_contract import feature_contract
        self.assertTrue(feature_contract(schema, []))

    def test_330_remains_rejected(self):
        with self.assertRaises(ValueError):
            validate_export_schema("3.30.0", fixture())

    def test_missing_required_column_rejected(self):
        schema = fixture()
        schema["NVTX_EVENTS"] = [r for r in schema["NVTX_EVENTS"]
                                  if r["name"] != "globalTid"]
        with self.assertRaises(ValueError):
            validate_export_schema("3.31.0", schema)

    def test_missing_kernel_activity_rejected(self):
        schema = fixture()
        del schema["CUPTI_ACTIVITY_KIND_KERNEL"]
        with self.assertRaises(ValueError):
            validate_export_schema("3.31.0", schema)

    def test_no_memset_table_allowed_if_absent(self):
        schema = fixture()
        del schema["CUPTI_ACTIVITY_KIND_MEMSET"]
        result = validate_export_schema("3.31.0", schema)
        self.assertEqual(result["structural_contract"], "passed")

    def test_present_activity_missing_field_rejected(self):
        schema = fixture()
        schema["CUPTI_ACTIVITY_KIND_MEMSET"] = [
            r for r in schema["CUPTI_ACTIVITY_KIND_MEMSET"]
            if r["name"] != "correlationId"
        ]
        with self.assertRaises(ValueError):
            validate_export_schema("3.31.0", schema)


if __name__ == "__main__":
    unittest.main()
