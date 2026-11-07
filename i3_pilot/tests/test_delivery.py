import hashlib,importlib.util,json,tempfile,unittest
from pathlib import Path
spec=importlib.util.spec_from_file_location('verify_delivery',Path(__file__).resolve().parents[1]/'scripts/verify_delivery.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class DeliveryTests(unittest.TestCase):
    def fixture(self,root):
        (root/'manifests').mkdir();(root/'source').mkdir();(root/'source/file.py').write_bytes(b'original')
        entry={'path':'source/file.py','sha256':hashlib.sha256(b'original').hexdigest()}
        (root/'manifests/FROZEN_SOURCE_SHA256.json').write_text(json.dumps({'files':[entry]}))
    def test_original_then_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            self.assertEqual(module.verify(root)['frozen_source_files_verified'],1)
            (root/'source/file.py').write_bytes(b'changed')
            with self.assertRaises(ValueError):module.verify(root)
    def test_unindexed_code_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);(root/'source/extra.py').write_text('extra')
            with self.assertRaises(ValueError):module.verify(root)
    def test_missing_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root);(root/'source/file.py').unlink()
            with self.assertRaises(ValueError):module.verify(root)
    def test_manifest_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            (root/'manifests/FROZEN_SOURCE_SHA256.json').write_text(json.dumps({'files':[{'path':'../outside.py','sha256':'0'*64}]}))
            with self.assertRaises(ValueError):module.verify(root)

if __name__=='__main__':unittest.main()
