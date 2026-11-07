import tempfile,unittest,json
from pathlib import Path
from o4_research.verify_assets import verify
from o4_research.common import sha
class AssetTests(unittest.TestCase):
 def test_hash_mismatch_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r=Path(d);p=r/'weights';p.write_bytes(b'fixture');item={'path':p.name,'sha256':sha(p)};manifest=r/'assets.json';manifest.write_text(json.dumps({'workloads':{'RN-A':{'status':'ready','weights':item,'input':item}}}));self.assertEqual(verify(manifest),1);p.write_bytes(b'changed')
   with self.assertRaises(ValueError):verify(manifest)
 def test_pending_asset_not_pass(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'assets.json';p.write_text(json.dumps({'workloads':{'YO-A':{'status':'pending_assets'}}}))
   with self.assertRaises(RuntimeError):verify(p)
 def test_published_manifest_hash_checked(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'assets.json';p.write_text(json.dumps({'workloads':{}}))
   with self.assertRaises(ValueError):verify(p,'0'*64)
if __name__=='__main__':unittest.main()
