"""Reload final complete H model/Adam/RNG, then verify all50 predictions; zero updates."""
import json
from pathlib import Path
import sys

PROJECT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/'H_terminal_ffn'))
from runtime import ROOT,T,np,setup,write,sha,tensor_hash
from helpers import PARENT,rng,restore_rng,equal,group_steps,evaluate
from terminal_ffn import load_extended_state

checkpoint=ROOT/'checkpoint-new0100-step2600.pt'
assert json.loads((ROOT/'complete.json').read_text())['stopped_at_budget']
out=ROOT/'cold_verify'
assert not out.exists(),'Refuse to overwrite a completed verification'
out.mkdir()
model,opt,groups,uids,pools,forward,objective,capture,args,batches=setup(False)
saved=T.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
load_extended_state(model,saved)
block=model.autoencoder.terminal_ffn
template=opt.param_groups[1]
extra={k:v for k,v in template.items() if k not in ('params','name','lr')}
extra.update(params=list(block.parameters()),name='terminal_ffn',lr=3e-5)
opt.add_param_group(extra)
opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng']);model.eval()
assert equal(model.state_dict(),saved['model'])
assert equal(opt.state_dict(),saved['optimizer']) and equal(rng(),saved['rng'])
assert group_steps(opt)==dict(encoder_mu=2600,decoder=2600,edge_head=2600,face_head=2600,terminal_ffn=100)
parent_eval=json.loads((PARENT/'B_lr03/eval-new0500.json').read_text())
before=tensor_hash(model.named_parameters())
actual=evaluate(model,uids,pools,objective,capture,batches,out,100,checkpoint,parent_eval)
reference=json.loads((ROOT/'eval-new0100.json').read_text())
assert actual['counts']==reference['counts'] and actual['perfect_uids']==reference['perfect_uids']
checks=[]
for now,old in zip(actual['meshes'],reference['meshes']):
    assert now['uid']==old['uid'] and now['parts']==old['parts']
    with np.load(ROOT/now['prediction_path']) as a,np.load(ROOT/old['prediction_path']) as b:
        assert a.files==b.files
        assert all(a[k].dtype==b[k].dtype and a[k].tobytes()==b[k].tobytes() for k in a.files),now['uid']
    checks.append(dict(uid=now['uid'],all_arrays_bitwise_equal=True,prediction_sha256=now['prediction_sha256']))
assert before==tensor_hash(model.named_parameters())
assert equal(opt.state_dict(),saved['optimizer']) and equal(rng(),saved['rng'])
result=dict(passed=True,optimizer_updates=0,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
    model_adam_rng_restored_exactly=True,adam_steps=group_steps(opt),
    execution_placement='terminal FFN prehook reinstalled before strict model load',
    counts=actual['counts'],joint_perfect=actual['joint_perfect'],all50_arrays_bitwise_equal=True,meshes=checks)
write(PROJECT/'repro_outputs/COLD_VERIFY.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='meshes'},indent=2),flush=True)
