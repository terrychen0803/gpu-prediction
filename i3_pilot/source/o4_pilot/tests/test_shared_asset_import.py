import importlib.util,json,sys,tempfile,unittest,hashlib
from pathlib import Path
from unittest.mock import patch
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/import_shared_assets.py'
spec=importlib.util.spec_from_file_location('asset_import_for_test',SCRIPT);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
class AssetImport(unittest.TestCase):
 def fixture(self,base):
  root=base/'target';src=base/'shared';(root/'manifests').mkdir(parents=True);src.mkdir();(src/'a').write_bytes(b'A');(src/'b').write_bytes(b'B')
  sha=lambda b:hashlib.sha256(b).hexdigest();manifest={'resnet':{'a':{'path':'assets/a','sha256':sha(b'A')},'b':{'path':'assets/b','sha256':sha(b'B')}},'operators':{}}
  (root/'manifests/assets.json').write_text(json.dumps(manifest));return root,src
 def invoke(self,root,src):
  with patch.object(mod,'ROOT',root),patch.object(sys,'argv',['import_shared_assets.py','--from',str(src)]):mod.main()
 def test_exact_copy_and_idempotence(self):
  with tempfile.TemporaryDirectory() as d:
   root,src=self.fixture(Path(d));self.invoke(root,src);self.invoke(root,src);self.assertEqual((root/'assets/a').read_bytes(),b'A');self.assertEqual((root/'assets/b').read_bytes(),b'B')
 def test_corrupted_source_rejected_before_any_copy(self):
  with tempfile.TemporaryDirectory() as d:
   root,src=self.fixture(Path(d));(src/'b').write_bytes(b'BAD')
   with self.assertRaises(ValueError):self.invoke(root,src)
   self.assertFalse((root/'assets').exists())
 def test_different_existing_asset_never_overwritten(self):
  with tempfile.TemporaryDirectory() as d:
   root,src=self.fixture(Path(d));(root/'assets').mkdir();(root/'assets/a').write_bytes(b'KEEP')
   with self.assertRaises(ValueError):self.invoke(root,src)
   self.assertEqual((root/'assets/a').read_bytes(),b'KEEP');self.assertFalse((root/'assets/b').exists())
