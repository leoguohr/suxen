"""Prepare links and validate files before a single launch. Does not train."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
cfg=json.loads((ROOT/'config.json').read_text())
parent=Path(cfg['source_checkpoint']).parent.parent
assert str(ROOT).startswith('/guohaoran/nexus_fast_track/diagnostics/')
assert not (ROOT/'run').exists() and not (ROOT/'snapshots').exists()
for name in ['READY.json','data_manifest.csv','overfit100_manifest.csv','selection.json','pool_provenance.json','data_validation.json','pools','source_archive','review_runtime']:
    source=(parent/name).resolve();assert source.exists()
    target=ROOT/name
    if not target.exists():target.symlink_to(source)
    assert target.resolve()==source
expected=json.loads((ROOT/'deploy_hashes.json').read_text())
for name,h in expected.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==h,name
    if name.endswith('.py'):compile((ROOT/name).read_text(),name,'exec')
assert Path(cfg['source_checkpoint']).exists()
for uid in cfg['uids']:
    s=json.loads((Path(cfg['free_probe'])/'snapshots'/uid/'summary.json').read_text())
    assert s['checkpoint_sha256']==cfg['source_sha256']
(ROOT/'deployment_verification.json').write_text(json.dumps(dict(files_verified=len(expected),source_references_exist=True,training_started=False),indent=2)+'\n')
print('PREPARED',ROOT)
