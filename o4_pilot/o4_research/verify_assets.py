import argparse
from pathlib import Path
from .common import read,sha
def verify(manifest,expected_manifest_sha256=None):
 p=Path(manifest);data=read(p);checked=set()
 if expected_manifest_sha256 is not None and sha(p)!=expected_manifest_sha256:raise ValueError('Manifest differs from published alignment hash')
 for name,entry in data['workloads'].items():
  if entry['status']!='ready':raise RuntimeError(f'{name}: {entry["status"]}')
  for key in ['weights','input','config']:
   if key not in entry:continue
   item=entry[key];path=p.parent/item['path']
   if not path.is_file() or sha(path)!=item['sha256']:raise ValueError(f'Asset missing/changed: {name}/{key}')
   checked.add(item['path'])
 return len(checked)
def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--expected-manifest-sha256');a=p.parse_args();print('Verified shared files:',verify(a.manifest,a.expected_manifest_sha256))
if __name__=='__main__':main()
