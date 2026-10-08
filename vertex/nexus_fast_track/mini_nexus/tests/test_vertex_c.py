from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import torch
from mini_nexus.octree import build_octree_levels
from scripts.train_vertex_c import depth_for, gt_at_depth, occupancy_for, sample_tree
from test_vertex_b2_frozen import make_checkpoint


def test_balanced_depth_schedule():
    assert Counter(depth_for(u,m) for u in range(1,10) for m in range(8)) == {d:8 for d in range(1,10)}
    assert Counter(depth_for(u,m) for u in range(1,1801) for m in range(8)) == {d:1600 for d in range(1,10)}


def test_oracle_tree_root_all_children_and_generated_parent_failure(tmp_path):
    vertices = torch.tensor([[x,y,z] for x in [0,511] for y in [0,511] for z in [0,511]])
    levels = build_octree_levels(vertices, depth=9)
    sample = SimpleNamespace(quantized_vertices=vertices, octree_levels=levels)
    context = torch.zeros(1,1,1)
    class Oracle:
        def __init__(self, broken=False): self.broken=broken
        def flow(self, x, t, parents, depths, context):
            depth = int(depths[0]); target = occupancy_for(parents[0], gt_at_depth(vertices,depth))
            if self.broken and depth==1: target[:,:,0]=0
            return (target-x)/(1-t[:,None,None])
    result = sample_tree(Oracle(), context, sample, 42, tmp_path/'correct')
    assert result['all_levels_exact'] and result['first_mismatch_depth'] is None
    assert [r['input_parent_count'] for r in result['levels']] == [1]+[8]*8
    assert all(r['predicted_count']==8 for r in result['levels'])
    assert occupancy_for(levels[0].parent_codes, gt_at_depth(vertices,1)).sum()==8
    bad = sample_tree(Oracle(True), context, sample, 42, tmp_path/'bad')
    assert not bad['exact_coordinate_set'] and bad['first_mismatch_depth']==1
    assert bad['levels'][1]['input_parent_count']==7
    assert all(r['missing_count']==1 for r in bad['levels'])


def test_c_cli_keeps_b2_optimizer_and_uses_new_depths(tmp_path):
    cp, manifest = make_checkpoint(tmp_path)
    # Keep this optimizer execution test small: a strongly empty synthetic model.
    state = torch.load(cp,weights_only=False)
    for name,value in state['model'].items():
        if name=='flow.output.bias':value.fill_(-20)
    torch.save(state,cp)
    sha = hashlib.sha256(cp.read_bytes()).hexdigest()
    gate = tmp_path/'gate.json';gate.write_text(json.dumps({'sampling_passed':True,'exact_count':64,
            'source_step':2000,'source_b2_update':1000,'checkpoint_sha256':sha}))
    root=Path(__file__).resolve().parents[1];out=tmp_path/'out'
    result=subprocess.run([sys.executable,str(root/'scripts/train_vertex_c.py'),'--code-root',str(root),
        '--checkpoint',str(cp),'--manifest',str(manifest),'--output',str(out),'--frozen-report',str(gate),
        '--updates','1','--eval-every','1','--device','cpu','--backup-dir',str(tmp_path/'backup')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    row=json.loads((out/'train.jsonl').read_text())
    assert row['depths']==list(range(1,9)) and len(set(row['times']))==8
    final=torch.load(out/'checkpoint-last.pt',weights_only=False)
    assert final['step']==2001 and final['c_update']==1
    assert all(s['step']==2 for s in final['optimizer']['state'].values())
    assert hashlib.sha256(cp.read_bytes()).hexdigest()==sha
    assert (tmp_path/'backup/checkpoint-last.pt').read_bytes()==(out/'checkpoint-last.pt').read_bytes()
    report=json.loads((out/'fresh_once.json').read_text())
    assert len(report['full_generated_parents'])==8
    assert not report['full_tree_passed']
