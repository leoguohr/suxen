"""Fixed50 scheduling, exact epoch-boundary continuation, and safe retention on CPU fixtures."""
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from _faces_reference import atomic_json, file_sha
from flow_training import (batch_cursor, build_training_state, train_update, checkpoint_state,
    restore_training, atomic_checkpoint, rng_state, training_recipe)
from train_flow import prune_checkpoints, read_stage_budget
import train_flow


def same(a,b):
    if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a: same(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b): same(x,y)
    else: assert a==b,(a,b)


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    uids = [f'synthetic_{i:03d}' for i in range(50)]
    tests = []
    for epoch in (0,1,99):
        groups = [batch_cursor(uids,10*epoch+i,5,0) for i in range(10)]
        assert all(g['epoch']==epoch and g['group']==i for i,g in enumerate(groups))
        assert sorted(sum([g['next_uids'] for g in groups],[])) == uids
    counts = {uid:0 for uid in uids}
    for step in range(1000):
        for uid in batch_cursor(uids,step,5,0)['next_uids']: counts[uid] += 1
    assert set(counts.values()) == {100}
    try: batch_cursor(uids[:-1]+[uids[0]],0,5,0); raise AssertionError('Duplicate UID accepted')
    except ValueError: pass
    tests.append('50 unique UIDs exactly once per 10-update epoch; 1000 updates imply 100 direct participations each')
    rng = torch.Generator().manual_seed(71)
    items = {uid:dict(mu=torch.randn(3+i%2,512,generator=rng)*.1,
        logvar=torch.full((3+i%2,512),-2.),vertices=torch.randn(3+i%2,3,generator=rng),
        points=torch.randn(4,6,generator=rng),point_mask=torch.ones(4,dtype=torch.bool)) for i,uid in enumerate(uids)}
    cache = SimpleNamespace(uids=uids,get=items.__getitem__,sha256='synthetic-cache',
        stats=dict(mean=[0.]*512,std=[1.]*512),manifest=dict(source_checkpoint_sha256='synthetic-vae'))
    config = dict(model=dict(hidden_dim=24,num_layers=1,num_heads=3,condition_dim=24,condition_layers=1,
        condition_heads=3,condition_tokens=2,rope_scale=1.,latent_dim=512,recompute=True),
        seed=0,batch_meshes=5,lr=1e-4,weight_decay=.01,warmup_updates=100,clip=1.)
    device = torch.device('cpu')
    model,opt,rngs = build_training_state(config,device)
    fps = {uid:torch.tensor([[0,1]]) for uid in uids}
    for step in range(9): train_update(model,opt,rngs,cache,config,step,fps=fps)
    with tempfile.TemporaryDirectory(prefix='topology-fixed50-training-') as temporary:
        root = Path(temporary)
        checkpoint = root/'boundary9.pt'
        digest = atomic_checkpoint(checkpoint,checkpoint_state(model,opt,rngs,cache,config,9,1.,{'fixture':'only'}))
        assert digest == file_sha(checkpoint)
        state = torch.load(checkpoint,weights_only=False)
        assert state['cursor']['epoch']==0 and state['cursor']['group']==9
        assert not {'max_updates','max_seconds','save_every'} & state['config'].keys()
        uninterrupted = [train_update(model,opt,rngs,cache,config,step,fps=fps) for step in (9,10)]
        extended = dict(config,max_updates=1000,max_seconds=28800,save_every=100)
        resumed,ropt,rrngs = build_training_state(extended,device)
        restore_training(state,resumed,ropt,rrngs,cache,extended,{'fixture':'only'})
        for step,expected in zip((9,10),uninterrupted):
            actual = train_update(resumed,ropt,rrngs,cache,extended,step,fps=fps)
            same(actual,expected)
            assert actual['lr'] == config['lr']*((step+1)/100)
        same(model.state_dict(),resumed.state_dict()); same(opt.state_dict(),ropt.state_dict())
        same(rng_state(rngs,device),rng_state(rrngs,device))
        assert batch_cursor(uids,10,5,0)['group']==0 and batch_cursor(uids,10,5,0)['epoch']==1
        tests.append('step9 checkpoint crosses epoch boundary with identical model, AdamW, draws, UID cursor and unreset warmup after budget extension')
        for corrupted in (dict(config,lr=2e-4),dict(config,seed=3)):
            try: restore_training(state,resumed,ropt,rrngs,cache,corrupted,{'fixture':'only'}); raise AssertionError('Recipe change accepted')
            except ValueError: pass
        broken = copy.deepcopy(state); broken['cursor']['group']=0
        try: restore_training(broken,resumed,ropt,rrngs,cache,config,{'fixture':'only'}); raise AssertionError('Cursor change accepted')
        except ValueError: pass
        tests.append('changed recipe and corrupted cursor fail closed')
        before = copy.deepcopy(checkpoint_state(resumed,ropt,rrngs,cache,config,11,0.,{'fixture':'only'}))
        checks = 0
        def stop():
            nonlocal checks
            checks += 1
            return checks == 3
        assert train_update(resumed,ropt,rrngs,cache,config,11,stop=stop,fps=fps) is None
        same(before,checkpoint_state(resumed,ropt,rrngs,cache,config,11,0.,{'fixture':'only'}))
        assert all(p.grad is None for p in resumed.parameters())
        progress = {}
        profiled = train_update(resumed,ropt,rrngs,cache,config,11,fps=fps,profile=True,progress=progress)
        assert profiled['completed_updates']==12 and profiled['adam_steps']==[12]
        assert profiled['profile']['accumulated_meshes']==5 and progress['phase']=='complete'
        assert all(profiled['profile'][k]>=0 for k in ('forward_seconds','backward_seconds','clip_seconds','adam_step_seconds','complete_update_seconds'))
        tests.append('partial five-mesh accumulation rolls back RNG/gradients; profile covers all five meshes and one counted Adam step')
        first = root/'stage1.json'; second = root/'stage2.json'
        for path,target in ((first,500),(second,1000)):
            atomic_json(path,dict(max_updates=target,max_seconds=7200,save_every=100,checkpoint_reserve_seconds=600))
        assert read_stage_budget(first)['max_updates']==500 and read_stage_budget(second)['max_updates']==1000
        assert training_recipe(extended)==config
        (root/'checkpoints').mkdir()
        entries = []
        for step in (0,3,100,500,600,700,800,900,1000):
            path=root/'checkpoints'/f'flow-{step:07d}.pt'; path.write_bytes(b'fixture')
            entries.append(dict(path=str(path),completed_updates=step,bytes=7,
                protected=['scheduled_evaluation'] if step in (500,1000) else ['best'] if step==600 else [],retained=True))
        Path(entries[1]['path']).with_suffix('.protect.json').write_text('{"reason":"final"}')
        prune_checkpoints(root,entries)
        assert {e['completed_updates'] for e in entries if e['retained']}=={3,500,600,1000}
        assert all(Path(e['path']).exists()==e['retained'] for e in entries)
        tests.append('separate stage files extend cumulative allowance; retention preserves newest one, eval500/1000, best and explicit final protection')
        # Exercise the actual CLI control path with only the source/cache replaced by CPU fixtures.
        cache.root = root/'fixture-cache'
        config_path = root/'recipe.json'; atomic_json(config_path,config)
        stage_path = root/'cli-stage.json'; output = root/'cli-run'
        atomic_json(stage_path,dict(max_updates=2,max_seconds=7200,save_every=100,checkpoint_reserve_seconds=600))
        argv = ['train_flow.py','--config',str(config_path),'--stage-budget',str(stage_path),
            '--cache',str(cache.root),'--output',str(output),'--device','cpu']
        with patch.object(train_flow,'LatentCache',return_value=cache),patch.object(sys,'argv',argv): train_flow.main()
        first_latest = json.loads((output/'latest.json').read_text())
        first_state = torch.load(first_latest['path'],weights_only=False)
        assert first_state['completed_updates']==2 and first_state['optimizer']['state']
        index = json.loads((output/'checkpoint-manifest.json').read_text())['checkpoints']
        assert any(e['completed_updates']==1 and len(e['sha256'])==64 for e in index)
        atomic_json(stage_path,dict(max_updates=3,max_seconds=7200,save_every=100,checkpoint_reserve_seconds=600))
        resumed_argv = argv+['--resume',first_latest['path'],'--resume-sha256',first_latest['sha256']]
        with patch.object(train_flow,'LatentCache',return_value=cache),patch.object(sys,'argv',resumed_argv): train_flow.main()
        final_latest = json.loads((output/'latest.json').read_text())
        final_state = torch.load(final_latest['path'],weights_only=False)
        assert final_state['completed_updates']==3
        assert {int(s['step']) for s in final_state['optimizer']['state'].values()}=={3}
        assert final_state['config']==config and final_state['stage_budget']['max_updates']==3
        assert file_sha(final_latest['path'])==final_latest['sha256']
        assert json.loads((output/'status.json').read_text())['complete']
        tests.append('CPU CLI saves real populated fixture Adam at update1, extends stage2 to3, verifies SHA and preserves Adam cursor')
    evidence = dict(passed=True,device='cpu',tests=tests,synthetic_optimizer_updates=17,
        real_mesh_optimizer_updates=0,gpu_used=False,production_scale_tested=False,temporary_fixtures_removed=True)
    atomic_json(Path(__file__).resolve().parent.parent/'evidence/training50_cpu_tests.json',evidence)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__': main()
