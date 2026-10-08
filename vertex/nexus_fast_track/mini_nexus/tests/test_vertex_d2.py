from collections import Counter
import hashlib,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from mini_nexus.octree import build_octree_levels
from mini_nexus.data_2k import load_nexus2k_sample
from scripts.train_vertex_d2 import pair_for, condition_pair, first_split, Evidence
from scripts.train_vertex_c import gt_at_depth,occupancy_for
from test_vertex_b2_frozen import make_checkpoint
from test_data_2k import _write_manifest,_write_sample


def test_pair_schedule_covers_each_mesh_and_depth():
    assert Counter(pair_for(u,m) for u in range(1,10) for m in range(8))=={(i,d):4 for i in range(2) for d in range(1,10)}
    assert Counter(pair_for(u,m) for u in range(1,3601) for m in range(8))=={(i,d):1600 for i in range(2) for d in range(1,10)}


def test_common_parents_condition_switch_and_condition_ignorant_control(tmp_path):
    a=torch.tensor([[x,y,z] for x in [0,511] for y in [0,511] for z in [0,511]])
    b=torch.tensor([[x,y,z] for x in [128,383] for y in [128,383] for z in [128,383]])
    samples=[SimpleNamespace(uid=str(i),quantized_vertices=v,octree_levels=build_octree_levels(v,9)) for i,v in enumerate([a,b])]
    contexts=[torch.tensor([[[float(i)]]]) for i in range(2)]
    assert first_split(samples)[0]==2
    class Oracle:
        def __init__(self,ignore=False):self.ignore=ignore
        def flow(self,x,t,parents,depth,context):
            index=0 if self.ignore else int(context[0,0,0])
            target=occupancy_for(parents[0],gt_at_depth(samples[index].quantized_vertices,int(depth[0])))
            return (target-x)/(1-t[:,None,None])
    good=condition_pair(Oracle(),contexts,samples,42,tmp_path/'good')
    assert good['full_match_matrix']==good['common_match_matrix']==[[True,False],[False,True]]
    left=np.load(tmp_path/'good/common-0.npz');right=np.load(tmp_path/'good/common-1.npz')
    assert np.array_equal(left['noise'],right['noise']) and np.array_equal(left['parents'],right['parents'])
    bad=condition_pair(Oracle(True),contexts,samples,42,tmp_path/'bad')
    assert not bad['full_pair_correct'] and not bad['common_pair_correct']
    assert bad['full_match_matrix']==[[True,False],[True,False]]


def test_d2_restores_adam_trains_vecset_and_persists_all_evidence(tmp_path):
    cp,original_manifest=make_checkpoint(tmp_path)
    import csv
    rows=list(csv.DictReader(original_manifest.open()))
    b=torch.tensor([[x,y,z] for x in [128,383] for y in [128,383] for z in [128,383]])
    rows.append(_write_sample(tmp_path,'nexus_2k_000195','train',b))
    manifest=tmp_path/'two.csv';_write_manifest(manifest,rows)
    samples=[load_nexus2k_sample(row) for row in rows]
    depth,parents,_=first_split(samples)
    selection={'uids':[s.uid for s in samples],'first_different_depth':depth,'common_parents':parents.tolist(),
        'records':[{'uid':s.uid,'vertices':len(s.quantized_vertices),
                    'condition_sha256':hashlib.sha256(s.condition.numpy().tobytes()).hexdigest(),
                    'quantized_vertices_sha256':hashlib.sha256(s.quantized_vertices.numpy().tobytes()).hexdigest()} for s in samples]}
    sel=tmp_path/'selection.json';sel.write_text(json.dumps(selection))
    state=torch.load(cp,weights_only=False);state.update(step=3800,c_update=1800)
    state['config'].update(phase='C',precision='fp32')
    state['model']['flow.output.bias'].fill_(-20)
    torch.save(state,cp);sha=hashlib.sha256(cp.read_bytes()).hexdigest()
    provenance=tmp_path/'provenance.tar.gz';provenance.write_bytes(b'CPU test provenance')
    root=Path(__file__).resolve().parents[1];out=tmp_path/'out';durable=tmp_path/'durable'
    cmd=[sys.executable,str(root/'scripts/train_vertex_d2.py'),'--code-root',str(root),'--checkpoint',str(cp),
        '--manifest',str(manifest),'--selection',str(sel),'--output',str(out),'--durable-dir',str(durable),
        '--provenance',str(provenance),'--expected-sha',sha,'--updates','18','--eval-every','9','--device','cpu']
    import time
    console=tmp_path/'console.txt'
    with console.open('w') as stream:
        proc=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+60
        checkpoint9=None
        while time.monotonic()<deadline:
            marker=out/'checkpoint_verified.json'
            if marker.exists() and json.loads(marker.read_text())['d2_update']==9:
                checkpoint9=tmp_path/'checkpoint9.pt'
                checkpoint9.write_bytes((out/'checkpoint-last.pt').read_bytes());break
            if proc.poll() is not None:break
            time.sleep(.01)
        assert checkpoint9 is not None,console.read_text()
        assert proc.wait(timeout=60)==0,console.read_text()
    logs=[json.loads(line) for line in (out/'train.jsonl').read_text().splitlines()]
    assert Counter((m['mesh_index'],m['depth']) for r in logs for m in r['microbatches'])=={(i,d):8 for i in range(2) for d in range(1,10)}
    assert all(r['vecset_forward_calls']==8 for r in logs)
    assert any(r['component_update_norms']['vecset_point_embedding']>0 for r in logs)
    checkpoint=torch.load(out/'checkpoint-last.pt',weights_only=False)
    assert checkpoint['step']==3818 and checkpoint['d2_update']==18
    assert all(s['step']==19 for s in checkpoint['optimizer']['state'].values())
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==sha
    for name in ['train.jsonl','checkpoint-last.pt','config.json','result.json','evaluation_ledger.jsonl','training_complete.json','final_evaluation_complete.json']:
        assert (out/name).read_bytes()==(durable/name).read_bytes()
    final=json.loads((out/'final_once.json').read_text())
    assert len(final['pairs'])==16 and final['pair_count']==16
    assert not final['full_condition_switch_passed']
    for relative,sha in final['array_sha256'].items():assert hashlib.sha256((durable/relative).read_bytes()).hexdigest()==sha

    resumed=tmp_path/'resumed';resumed_durable=tmp_path/'resumed_durable'
    resume_cmd=cmd.copy()
    for flag,value in [('--checkpoint',checkpoint9),('--expected-sha',hashlib.sha256(checkpoint9.read_bytes()).hexdigest()),('--output',resumed),('--durable-dir',resumed_durable)]:
        resume_cmd[resume_cmd.index(flag)+1]=str(value)
    resume_cmd += ['--resume-update','9','--resume-evaluation',str(out/'evaluation-000009.json'),'--resume-training-log',str(out/'train.jsonl')]
    recovered=subprocess.run(resume_cmd,capture_output=True,text=True)
    assert recovered.returncode==0,recovered.stderr
    replay=[json.loads(x) for x in (resumed/'train.jsonl').read_text().splitlines()]
    assert [r['update'] for r in replay]==list(range(10,19))
    assert all(r['matches_original_replayed_input'] for r in replay)
    restored=torch.load(resumed/'checkpoint-last.pt',weights_only=False)
    assert restored['step']==3818 and restored['d2_update']==18
    assert all(s['step']==19 for s in restored['optimizer']['state'].values())
    assert torch.equal(restored['torch_rng'],checkpoint['torch_rng'])
    assert all(torch.equal(value,restored['model'][name]) for name,value in checkpoint['model'].items())
    assert json.loads((resumed/'recovery_verification.json').read_text())['compared_prior_evaluation_arrays']==84
