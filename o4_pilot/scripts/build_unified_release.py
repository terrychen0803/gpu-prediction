"""Maintainer tool: generate a new reviewed distribution manifest, never historical raw."""
from pathlib import Path
import json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from o4_research.provenance import ROOT,source_hashes
from o4_research.common import write
if __name__=='__main__':
    write(ROOT/'manifests/unified_release.json',{'release_id':'o4-unified-engineering-v1-20261106-guard-retry-schema329-patch2',
        'previous_immutable_release_tag':'o4-unified-engineering-v1-20261106',
        'status':'source_schema329_structural_allowlist_pending_4090_reaudit','protocol_id':'o4_seven_A_40x3_engineering_v2_20261106',
        'asset_manifest_sha256':'55300ee1eb51a481421a7318ec8fbf9c1b2bffdce9e31d77d37f7383cf69cec4',
        'guard_policy':'40x3: up to 60 seconds cooldown at 3s intervals; require two idle readings; never bypass foreign GPU processes',
        'schema_policy':'3.29.0 RTX4090 RN-A smoke P/R field inspection; structural contract only, strict Reference remains unverified',
        'historical_release_manifests_preserved':True,'strict_verified':0,'files':source_hashes()})
    print('Built distribution source manifest; review diff before committing.')
