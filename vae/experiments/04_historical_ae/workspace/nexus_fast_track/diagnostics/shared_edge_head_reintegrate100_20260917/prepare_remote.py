import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parent
cfg=json.loads((root/'config.json').read_text())
parent=Path(cfg['source_checkpoint']).parent.parent
assert not (root/'run').exists()
for name in ['READY.json','data_manifest.csv','overfit100_manifest.csv','selection.json','pool_provenance.json','data_validation.json','pools','source_archive','review_runtime','augmented_pools','mining_complete.json']:
    source=(parent/name).resolve();assert source.exists(),source
    target=root/name
    if not target.exists():target.symlink_to(source)
    assert target.resolve()==source
shutil.copyfile(parent/'run/eval-epoch900.jsonl',root/'source_archived_eval.jsonl')
shutil.copyfile(parent/'run/manifest.json',root/'source_manifest.json')
for name,digest in json.loads((root/'deploy_hashes.json').read_text()).items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    if name.endswith('.py'):compile((root/name).read_text(),name,'exec')
print('PREPARED',root,flush=True)
