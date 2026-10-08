"""No-update replay of each accepted checkpoint and the final checkpoint."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fourway_continuation',ROOT/'run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
m.torch.set_num_threads(1);m.torch.cuda.set_device(0)
m.torch.set_float32_matmul_precision('highest')
m.torch.backends.cuda.matmul.allow_tf32=False;m.torch.backends.cudnn.allow_tf32=False
out=ROOT/'continue'
done=json.loads((out/'complete.json').read_text())
trace={r['step']:r for r in map(json.loads,(out/'trace.jsonl').read_text().splitlines())}
steps=sorted(set(done['four_way_perfect_steps']+[done['steps']]))
results=[]
for step in steps:
    saved=trace[step]
    path=Path(saved['saved_checkpoint']);sha=m.probe.digest(path)
    m.probe.UIDS=['nexus_2k_000387','nexus_2k_001849']
    cp,model,batch=m.probe.setup_model(path)
    assert cp['diagnostic_metrics']['rows']==saved['rows']
    assert cp['diagnostic_metrics']['full_reconstruction']==saved['full_reconstruction']
    assert cp['diagnostic_metrics']['four_way_perfect_count']==saved['four_way_perfect_count']
    opt=cp['optimizer']
    adam_steps=[sorted({float(opt['state'][i]['step']) for i in g['params']}) for g in opt['param_groups']]
    assert adam_steps==[[float(step)],[float(step)],[float(step-6500)]]
    assert [g['lr'] for g in opt['param_groups']]==[1e-8,1e-7,1e-7]
    with m.torch.no_grad():
        rows=m.probe.get_rows(model,batch,'mu')
        data=[m.np.load(m.teacher.modules.ab.PREVIOUS/(uid+'_pool.npz')) for uid in m.probe.UIDS]
        replay=m.full_reconstruction(rows,data,model.scoring_contract(),step)
    assert all(m.torch.equal(p.cpu(),cp['model'][n]) for n,p in model.named_parameters())
    assert m.probe.digest(path)==sha
    results.append(dict(step=step,checkpoint=str(path),checkpoint_sha256=sha,
        checkpoint_matches_saved_metrics=True,adam_steps_by_group=adam_steps,parameters_unchanged=True,
        saved_full_reconstruction=saved['full_reconstruction'],replayed_full_reconstruction=replay))
    m.probe.write(out/'checkpoint_verification.json',results)
    del rows,cp,model,batch,opt
    m.torch.cuda.empty_cache()
