"""Reload final saved weights and reconstruct both meshes without any updates."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('face_soft4_diagnostic',ROOT/'run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.torch.set_num_threads(1)
m.torch.cuda.set_device(0)
m.torch.set_float32_matmul_precision('highest')
m.torch.backends.cuda.matmul.allow_tf32=False
m.torch.backends.cudnn.allow_tf32=False
out=ROOT/'continue'
path=out/'checkpoint-12100.pt'
digest=m.probe.digest(path)
m.probe.UIDS=['nexus_2k_000387','nexus_2k_001849']
cp,model,batch=m.probe.setup_model(path)
last=json.loads((out/'trace.jsonl').read_text().splitlines()[-1])
assert cp['diagnostic_metrics']['rows']==last['rows']
assert cp['diagnostic_metrics']['full_reconstruction']==last['full_reconstruction']
opt=cp['optimizer']
steps=[sorted({float(opt['state'][i]['step']) for i in g['params']}) for g in opt['param_groups']]
assert steps==[[12100.],[12100.],[5600.]]
assert [g['lr'] for g in opt['param_groups']]==[1e-8,1e-7,1e-7]
with m.torch.no_grad():
    rows=m.probe.get_rows(model,batch,'mu')
    data=[m.np.load(m.teacher.modules.ab.PREVIOUS/(uid+'_pool.npz')) for uid in m.probe.UIDS]
    result=m.full_reconstruction(rows,data,model.scoring_contract(),12100)
assert all(m.torch.equal(p.cpu(),cp['model'][n]) for n,p in model.named_parameters())
assert m.probe.digest(path)==digest
m.probe.write(out/'checkpoint_verification.json',dict(checkpoint_sha256=digest,
    checkpoint_matches_final_metrics=True,adam_steps_by_group=steps,parameters_unchanged=True,
    saved_full_reconstruction=last['full_reconstruction'],replayed_full_reconstruction=result,
    note='Fresh no-update forward; any count difference is recorded, not hidden.'))
