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
    def test_328_and_331_structural_only(self):
        for version in ("3.28.0", "3.31.0"):
            result = validate_export_schema(version, fixture())
            self.assertEqual(result["structural_contract"], "passed")
            self.assertEqual(result["semantic_equivalence"], "unverified")
            self.assertFalse(result["strict_reference_verified"])

    def test_unknown_version_rejected(self):
        with self.assertRaises(ValueError):
            validate_export_schema("3.32.0", fixture())

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
