"""Reload first/final saved embeddings and verify labels directly from GT faces."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--snapshot',type=Path,required=True)
parser.add_argument('--run',type=Path,required=True)
args=parser.parse_args()
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
spec=importlib.util.spec_from_file_location('scoring',args.snapshot/'effective_code/effective_loss_and_scoring.py')
scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
source=np.load(args.snapshot/'representations_and_gradients.npz')
faces=source['gt_faces']
gt=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[0,2]]]),axis=1),axis=0)
n=len(source['vertices'])
assert len(gt)==json.loads((args.snapshot/'summary.json').read_text())['gt_edges']
scale=json.loads((args.snapshot/'effective_code/runtime_contract.json').read_text())['scales']['edge_logit_scale']
results={}
names=['checkpoint-step2000']
if (args.run/'first-perfect.pt').exists():names.insert(0,'first-perfect')
for name in names:
    cp=torch.load(args.run/(name+'.pt'),map_location='cpu',weights_only=False)
    saved=np.load(args.run/(name+'.npz'))
    pairs=saved['pairs']; labels=np.isin(pairs[:,0]*n+pairs[:,1],gt[:,0]*n+gt[:,1])
    assert np.array_equal(labels,saved['labels'])
    assert torch.equal(cp['edge_head_raw'],torch.from_numpy(saved['edge_head_raw']))
    assert int(cp['optimizer']['state'][0]['step']) == cp['completed_updates']
    raw=cp['edge_head_raw'].cuda(); pair=torch.from_numpy(pairs).cuda()
    def forward():
        e=raw-raw.mean(0,keepdim=True)
        return e,scoring.first_order_interval(e[pair[:,0]],e[pair[:,1]])*scale
    with torch.no_grad():
        e,logit=forward();_,repeat=forward()
    assert torch.equal(logit,repeat)
    assert np.array_equal(e.cpu().numpy(),saved['edge_embedding_scoring'])
    assert np.array_equal(logit.cpu().numpy(),saved['logits'])
    pred=logit.cpu().numpy()>0
    fp=int(np.sum(pred & ~labels));fn=int(np.sum(~pred & labels))
    assert fp==cp['metrics']['fp'] and fn==cp['metrics']['fn']
    if name=='first-perfect':assert fp==fn==0
    results[name]=dict(step=cp['completed_updates'],tp=int(np.sum(pred&labels)),fp=fp,fn=fn,
                       pairs=len(pairs),labels_rebuilt_from_gt_faces=True,
                       saved_logits_reproduced_exactly=True,repeated_forward_exact=True,
                       optimizer_step_matches_checkpoint=True)
(args.run/'saved_checkpoint_verification.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results))
