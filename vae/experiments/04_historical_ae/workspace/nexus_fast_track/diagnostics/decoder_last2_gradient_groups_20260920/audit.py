"""Independent CPU reconstruction from saved gradient vectors and source Adam."""
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
import copy, json, math
from pathlib import Path
import torch
torch.set_num_threads(1)
R = Path(__file__).resolve().parent
r = json.loads((R/'result.json').read_text())
v = torch.load(R/'gradient_vectors.pt', map_location='cpu', weights_only=True)
cp = torch.load(r['source_checkpoint'], map_location='cpu', weights_only=False)
names = v['parameter_names']
parts = r['parts']
grads = [v['gradients'][p] for p in parts]
total = v['original_total_gradient']
delta = v['cloned_adam_actual_fp32_displacement']
def dot(a,b):
    return sum(float((x.double()*y.double()).sum()) for x,y in zip(a,b))
def close(x,y):
    assert math.isclose(x,y,rel_tol=2e-10,abs_tol=1e-13),(x,y)
for name,g in zip(parts,grads):
    close(dot(g,delta),r['impacts'][name]['adam_actual_displacement_dot'])
    close(math.sqrt(dot(g,g)),r['impacts'][name]['gradient_norm'])
for i,g in enumerate(grads):
    for j,h in enumerate(grads):
        close(dot(g,h),r['matrices']['all']['dot'][i][j])
params = {n:torch.nn.Parameter(cp['tail'][n].clone()) for n in names}
sets = [['block','final_norm'],['head'],['face'],['penultimate']]
groups = [dict(params=[params[n] for n in names if n.split('.')[0] in prefixes]) for prefixes in sets]
opt = torch.optim.Adam(groups)
opt.load_state_dict(copy.deepcopy(cp['optimizer']))
for n,g in zip(names,v['clipped_total_gradient']):
    params[n].grad = g.clone()
opt.step()
cpu_delta = [params[n].detach()-cp['tail'][n] for n in names]
error = math.sqrt(dot([x-y for x,y in zip(cpu_delta,delta)],[x-y for x,y in zip(cpu_delta,delta)]))/math.sqrt(dot(delta,delta))
impacts = {name:dot(g,cpu_delta) for name,g in zip(parts,grads)}
assert error < 0.002,error  # CPU/GPU Adam kernels may differ at FP32 ulps.
assert all(math.copysign(1,impacts[p]) == math.copysign(1,r['impacts'][p]['adam_actual_displacement_dot']) for p in parts)
out = dict(gradient_norms_gram_and_predictions_recomputed=True,
           independent_cpu_adam_from_source_moments=True,
           cpu_gpu_actual_displacement_relative_l2=error,
           cpu_first_order_impacts=impacts,
           same_four_prediction_signs=True,source_optimizer_updates=0)
(R/'independent_audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
