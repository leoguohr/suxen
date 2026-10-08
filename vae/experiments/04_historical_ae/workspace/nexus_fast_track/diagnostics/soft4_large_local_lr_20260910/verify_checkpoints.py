"""Verify final Adam states and replay the saved C checkpoint without updates."""
import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
import importlib.util
import json
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('large_local_lr_training_source',ROOT/'run.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
torch.set_num_threads(1);torch.manual_seed(20260910);torch.cuda.set_device(0)
torch.set_float32_matmul_precision('highest')
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
verified={}
for case,(_,_,lr_e,lr_d) in run.PHASES.items():
    path=ROOT/case/'checkpoint-6500.pt'
    cp=torch.load(path,map_location='cpu',mmap=True)
    done=json.loads((ROOT/case/'complete.json').read_text())
    assert cp['diagnostic_run']['completed_steps']==6500
    assert {float(v['step']) for v in cp['optimizer']['state'].values()}=={6500.}
    assert [g['lr'] for g in cp['optimizer']['param_groups']]==[lr_e,lr_d]
    assert cp['diagnostic_metrics']['rows']==done['final']['rows']
    verified[case]=dict(checkpoint_sha256=run.probe.digest(path),adam_step=6500,metrics_match_trace=True)
    del cp
run.probe.UIDS=['nexus_2k_000387','nexus_2k_001849']
cp,model,batch=run.probe.setup_model(ROOT/'C/checkpoint-6500.pt')
assert all(torch.equal(p.cpu(),cp['model'][n]) for n,p in model.named_parameters())
model.eval()
with torch.no_grad():
    rows=run.probe.get_rows(model,batch,'mu')
    replay=[]
    for i,uid in enumerate(run.probe.UIDS):
        _,counts=run.teacher.paper_edge_loss_all_pairs(rows[2][i],batch.edge_index[i],
            pair_chunk_size=cp['args']['pair_chunk_size'],counts_on_device=True,
            logit_scale=model.scoring_contract()['edge_logit_scale'])
        c={k:int(v) for k,v in counts.items()}
        replay.append(dict(uid=uid,**c,edge_f1=2*c['tp']/(2*c['tp']+c['fp']+c['fn'])))
assert all(torch.equal(p.cpu(),cp['model'][n]) for n,p in model.named_parameters())
result=dict(final_checkpoints=verified,C_fresh_load_forward=replay,optimizer_steps_in_verification=0)
(ROOT/'checkpoint_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
