"""Prepare fixed20 pools; no optimizer, no training."""
import json,hashlib,shutil,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('archived_pool',ROOT.parent/'aggressive_topology_probe_20260907/experiment.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)
p.torch.set_num_threads(1);p.torch.cuda.set_device(0)
p.torch.backends.cuda.matmul.allow_tf32=False;p.torch.backends.cudnn.allow_tf32=False
sel=json.loads((ROOT/'selection.json').read_text());pools=ROOT/'pools';pools.mkdir(exist_ok=True)
old=ROOT.parent/'math00_four_mesh_lr10x_20260913'
records=[]
for row in sel['records']:
 uid=row['uid'];out=pools/(uid+'_pool.npz');original=p.PREVIOUS/(uid+'_candidates.npz')
 d=p.np.load(original)
 assert len(d['vertices'])==row['vertices'] and len(d['positives'])==row['faces'],uid
 for key,field in [('vertices','vertices_sha256'),('positives','faces_sha256')]:
  assert hashlib.sha256(d[key].tobytes()).hexdigest()==row[field],(uid,field)
 if not out.exists():
  if (old/out.name).exists():shutil.copyfile(old/out.name,out)
  else:
   work=ROOT/'pool_preparation'/uid;work.mkdir(parents=True,exist_ok=True)
   p.UIDS=[uid]
   # The archive's extra evaluation after mining does not affect the pool.
   p.evaluate=lambda *args,**kwargs:None
   p.prepare(work)
   shutil.copyfile(work/out.name,out)
 pool=p.np.load(out)
 assert p.np.array_equal(pool['vertices'],d['vertices']) and p.np.array_equal(pool['positive'],d['positives'])
 assert not p.np.intersect1d(p.keys(pool['positive'],len(d['vertices'])),p.keys(pool['mixed'],len(d['vertices']))).size
 records.append(dict(uid=uid,pool_sha256=p.digest(out),source_sha256=p.digest(original),vertices=len(d['vertices']),faces=len(d['positives']),negatives=len(pool['mixed'])))
 print('READY_POOL',uid,flush=True)
p.write(ROOT/'pools_ready.json',dict(records=records,pool_rule='archived prepare, original mining checkpoint; existing four pools reused',preparation_script_sha256=p.digest(Path(__file__)),mining_script_sha256=p.digest(Path(p.__file__))))
print('ALL_20_POOLS_READY',flush=True)
