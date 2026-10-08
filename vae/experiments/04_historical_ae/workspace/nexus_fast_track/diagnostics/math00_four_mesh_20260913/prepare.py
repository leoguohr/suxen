"""Freeze size-selected additions and run the exact archived fixed-pool mining rule."""
from pathlib import Path
import importlib.util,json,shutil,hashlib
ROOT=Path(__file__).resolve().parent
OLD=['nexus_2k_000387','nexus_2k_001849'];NEW=['nexus_2k_001716','nexus_2k_001333']
ROOT.mkdir(exist_ok=True)
source=ROOT.parent/'aggressive_topology_probe_20260907/experiment.py'
spec=importlib.util.spec_from_file_location('archived_pool_generation',source);p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.torch.set_num_threads(1);p.torch.cuda.set_device(0)
p.torch.backends.cuda.matmul.allow_tf32=False;p.torch.backends.cudnn.allow_tf32=False;p.torch.set_float32_matmul_precision('highest')
records=[]
for uid in OLD+NEW:
 path=p.PREVIOUS/(uid+'_candidates.npz');d=p.np.load(path)
 records.append(dict(uid=uid,role='old' if uid in OLD else 'new',vertices=len(d['vertices']),faces=len(d['positives']),candidate_source=str(path),candidate_source_sha256=p.digest(path),vertices_sha256=hashlib.sha256(d['vertices'].tobytes()).hexdigest(),faces_sha256=hashlib.sha256(d['positives'].tobytes()).hexdigest()))
assert len(set(x['vertices_sha256'] for x in records))==4
selection=dict(uids=OLD+NEW,records=records,selection='Two new full meshes selected by size before inspecting Hold reconstruction; no subgraph sampling.',budget_updates=1000,training_beta=1e-4,preparation_rule='Exact aggressive_topology_probe_20260907/experiment.py prepare: archived checkpoint1000 mu+sample0 mining; preserve GT false cycles; same stable UID seed and replacement/heldout rules.',preparation_source_sha256=p.digest(source),mining_checkpoint=str(p.CHECKPOINT),mining_checkpoint_sha256=p.digest(p.CHECKPOINT))
p.write(ROOT/'selection.json',selection)
print('SELECTION',json.dumps(selection),flush=True)
for uid in OLD:shutil.copyfile(ROOT.parent/'aggressive_topology_probe_20260907'/(uid+'_pool.npz'),ROOT/(uid+'_pool.npz'))
out=ROOT/'new_candidates';out.mkdir(exist_ok=True)
p.UIDS=NEW;p.prepare(out)
for uid in NEW:shutil.copyfile(out/(uid+'_pool.npz'),ROOT/(uid+'_pool.npz'))
selection['pool_sha256']={uid:p.digest(ROOT/(uid+'_pool.npz')) for uid in OLD+NEW}
p.write(ROOT/'selection.json',selection)
print('PREPARATION_COMPLETE',flush=True)
