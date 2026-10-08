"""Explain FP32 delta-subtract/add reconstruction without executing an optimizer."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
import hashlib,json,math
from pathlib import Path
import torch as T
T.set_num_threads(1)
R=Path(__file__).resolve().parent;S=R.parent/'decoder_last2_joint_fixed100_20260920'
G=R.parent/'decoder_last2_gradient_groups_20260920';F=R.parent/'decoder_last2_adam_displacement_20260921'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
cp=T.load(S/'run/checkpoint-new0500-tail1500.pt',map_location='cpu',weights_only=False)
saved=R/'A_control/failure-state.pt';after=T.load(saved,map_location='cpu',weights_only=False)
finite=T.load(F/'points/plus1/tail-fp32.pt',map_location='cpu',weights_only=False)
v=T.load(G/'gradient_vectors.pt',map_location='cpu',weights_only=False)
total_diff=0;max_abs=0.;records=[]
for n,expected in zip(v['parameter_names'],v['cloned_adam_actual_fp32_displacement']):
    p=after['tail'][n];p0=cp['tail'][n];actual=p-p0
    assert T.equal(actual,expected)
    assert T.equal(p0+actual,finite[n])
    diff=p.double()-finite[n].double();count=int(T.count_nonzero(diff));maximum=float(diff.abs().max())
    records.append(dict(parameter=n,different_elements=count,max_absolute_difference=maximum))
    total_diff+=count;max_abs=max(max_abs,maximum)
record=json.loads((R/'A_control/run/updates.jsonl').read_text())
old=json.loads((G/'result.json').read_text())['adam_candidate']
assert record['global_gradient_norm']==old['clip_norm'] and record['clip_coefficient']==old['clip_coefficient']
assert record['actual_delta_norm']==old['actual_displacement_norm']
result=dict(cause='FP32 subtraction of saved delta then re-addition is not an exact inverse at tiny values',
    actual_fp32_adam_displacement_bitwise_equal_previous=True,gradient_norm_and_clip_exact=True,
    fp32_source_plus_actual_delta_bitwise_equals_finite_plus1=True,different_parameter_elements=total_diff,
    max_parameter_difference=max_abs,per_parameter=records,failure_checkpoint_sha256=sha(saved),
    completed_A_updates=1,completed_B_updates=0,resume_from_A_step1_no_optimizer_reset_or_replay=True)
(R/'resume_first_step.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
