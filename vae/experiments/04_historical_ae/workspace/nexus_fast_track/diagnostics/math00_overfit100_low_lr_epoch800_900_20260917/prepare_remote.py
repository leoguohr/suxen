"""Run in the uploaded task directory; prepare links and verify deployment only."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'math00_overfit100_low_lr_continue100ep_20260916'
assert str(ROOT).startswith('/guohaoran/nexus_fast_track/diagnostics/')
assert not (ROOT/'run').exists(),'Existing run requires explicit inspection; do not overwrite'
for name in ['READY.json','data_manifest.csv','overfit100_manifest.csv','selection.json',
             'pool_provenance.json','data_validation.json','excluded.csv','pools',
             'source_archive','review_runtime','mined','augmented_pools','mining_complete.json']:
    source=PARENT/name;assert source.exists(),source
    dest=ROOT/name
    if not dest.exists():dest.symlink_to(source.resolve())
    assert dest.resolve()==source.resolve()
expected=json.loads((ROOT/'deploy_hashes.json').read_text())
for name,digest in expected.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    if name.endswith('.py'):compile((ROOT/name).read_text(),name,'exec')
checkpoint=PARENT/'run/checkpoint-update20000.pt';assert checkpoint.exists()
complete=json.loads((PARENT/'run/complete.json').read_text())
assert complete['updates']==20000 and complete['epochs']==800
assert complete['final_evaluation']['checkpoint_sha256']=='38087aeefbc12444cb51ba35dbe3715cc0f785c30cd6ed53e093a0ee7b929f80'
record=dict(files_verified=len(expected),parent_complete=True,parent_checkpoint=str(checkpoint),
            checkpoint_hash_will_be_recomputed_by_runner=True,training_started=False)
(ROOT/'deployment_verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
