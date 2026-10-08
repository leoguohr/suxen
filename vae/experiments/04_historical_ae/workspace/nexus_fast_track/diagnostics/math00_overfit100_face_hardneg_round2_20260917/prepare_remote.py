"""Prepare data links and verify uploaded code without starting training."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent
PARENT=ROOT.parent/'math00_overfit100_low_lr_continue100ep_20260916'
CONTROL=ROOT.parent/'math00_overfit100_low_lr_epoch800_900_20260917'
assert str(ROOT).startswith('/guohaoran/nexus_fast_track/diagnostics/')
assert not (ROOT/'run').exists() and not (ROOT/'mining.jsonl').exists()
for name in ['READY.json','data_manifest.csv','overfit100_manifest.csv','selection.json',
             'pool_provenance.json','data_validation.json','excluded.csv','pools','source_archive','review_runtime']:
    source=PARENT/name;assert source.exists(),source
    dest=ROOT/name
    if not dest.exists():dest.symlink_to(source.resolve())
    assert dest.resolve()==source.resolve()
for name,source in {'common_pools':PARENT/'augmented_pools',
                    'parent_mining_complete.json':PARENT/'mining_complete.json',
                    'first_round_mined':PARENT/'mined'}.items():
    assert source.exists(),source
    dest=ROOT/name
    if not dest.exists():dest.symlink_to(source.resolve())
    assert dest.resolve()==source.resolve()
expected=json.loads((ROOT/'deploy_hashes.json').read_text())
for name,digest in expected.items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    if name.endswith('.py'):compile((ROOT/name).read_text(),name,'exec')
v=json.loads((ROOT/'code_verification.json').read_text())
assert hashlib.sha256((CONTROL/'train_continue.py').read_bytes()).hexdigest()==v['source_sha256']
for name,h in v['unchanged_files'].items():assert hashlib.sha256((CONTROL/name).read_bytes()).hexdigest()==h
assert json.loads((CONTROL/'run/complete.json').read_text())['updates']==22500
assert json.loads((PARENT/'run/complete.json').read_text())['final_evaluation']['checkpoint_sha256']=='38087aeefbc12444cb51ba35dbe3715cc0f785c30cd6ed53e093a0ee7b929f80'
assert (PARENT/'run/checkpoint-update20000.pt').exists()
for epoch in range(801,901):assert (CONTROL/'run'/f'epoch-order-{epoch}.json').exists()
record=dict(files_verified=len(expected),executed_control_source_verified=True,
            parent_complete=True,control_complete=True,all100_control_orders_present=True,training_started=False)
(ROOT/'deployment_verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
