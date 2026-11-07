import unittest
from o4_research.observe_memset import validate_rows
class ArgumentEvidenceTests(unittest.TestCase):
 def rows(self,count='0',exit_count=None,rc='0'):
  return [{'site':'0','correlation_id':'7','scope':'op140','count_bytes':count},{'site':'1','correlation_id':'7','scope':'op140','count_bytes':count if exit_count is None else exit_count,'return_code':rc}]
 def test_direct_zero_byte(self):self.assertEqual(validate_rows(self.rows())[0]['count_bytes'],'0')
 def test_positive_bytes_not_relabelled_noop(self):self.assertEqual(validate_rows(self.rows('64'))[0]['count_bytes'],'64')
 def test_mutated_parameter_rejected(self):
  with self.assertRaises(AssertionError):validate_rows(self.rows(exit_count='64'))
 def test_failed_call_rejected(self):
  with self.assertRaises(AssertionError):validate_rows(self.rows(rc='1'))
 def test_missing_exit_rejected(self):
  with self.assertRaises(AssertionError):validate_rows(self.rows()[:1])
if __name__=='__main__':unittest.main()
