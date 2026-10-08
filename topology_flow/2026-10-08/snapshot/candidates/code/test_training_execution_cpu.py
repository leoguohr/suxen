"""Small CPU CLI fixtures for execution-policy identity and per-attempt profiling."""
import copy
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from _faces_reference import atomic_json, file_sha
import train_flow
from train_flow import read_execution_policy, check_resume_execution_policy


def expect(kind, call):
    try: call()
    except kind: return
    raise AssertionError('Expected '+kind.__name__)


def main():
    torch.set_num_threads(1)
    tests = []
    config = dict(model=dict(hidden_dim=24,num_layers=2,num_heads=3,condition_dim=24,
        condition_layers=2,condition_heads=3,condition_tokens=2,rope_scale=1.,latent_dim=512,recompute=True),
        seed=0,batch_meshes=5,lr=1e-4,weight_decay=.01,warmup_updates=100,clip=1.)
    with tempfile.TemporaryDirectory(prefix='topology-execution-cpu-') as temporary:
        root = Path(temporary)
        policy_path = root/'policy.json'
        policy = dict(flow_direct_blocks=[1],condition_direct_blocks=[1])
        atomic_json(policy_path,policy)
        execution = read_execution_policy(policy_path,config['model'])
        formatted = root/'reformatted.json'
        formatted.write_text(json.dumps(policy,indent=4)+'\n')
        same_policy = read_execution_policy(formatted,config['model'])
        assert execution['execution_policy_sha256'] == same_policy['execution_policy_sha256']
        assert execution['execution_policy_file_sha256'] != same_policy['execution_policy_file_sha256']
        check_resume_execution_policy(execution,same_policy)
        empty = read_execution_policy(None,config['model'])
        check_resume_execution_policy({},empty)
        expect(ValueError,lambda:check_resume_execution_policy({},execution))
        damaged = dict(execution,execution_policy_sha256='0'*64)
        expect(ValueError,lambda:check_resume_execution_policy(damaged,execution))
        damaged = dict(execution,execution_policy=dict(policy,flow_direct_blocks=[True]))
        expect(ValueError,lambda:check_resume_execution_policy(damaged,execution))
        for invalid in ([],{},dict(policy,other=[]),dict(policy,flow_direct_blocks=[True]),
                dict(policy,flow_direct_blocks=[-1]),dict(policy,flow_direct_blocks=[2]),
                dict(policy,flow_direct_blocks=[1,1]),dict(policy,flow_direct_blocks=[1,0]),
                dict(policy,condition_direct_blocks=[0.]),dict(policy,condition_direct_blocks=[[0]])):
            atomic_json(root/'invalid.json',invalid)
            expect(ValueError,lambda:read_execution_policy(root/'invalid.json',config['model']))
        tests.append('Strict policy schema; sorted unique bounded integers; canonical and file SHA identities; legacy empty-policy compatibility only')

        generator = torch.Generator().manual_seed(71)
        uids = [f'synthetic_execution_{i:03d}' for i in range(50)]
        items = {uid:dict(mu=torch.randn(3+i%2,512,generator=generator)*.1,
            logvar=torch.full((3+i%2,512),-2.),vertices=torch.randn(3+i%2,3,generator=generator),
            points=torch.randn(4,6,generator=generator),point_mask=torch.ones(4,dtype=torch.bool)) for i,uid in enumerate(uids)}
        cache = SimpleNamespace(uids=uids,get=items.__getitem__,sha256='synthetic-execution-cache',
            stats=dict(mean=[0.]*512,std=[1.]*512),manifest=dict(source_checkpoint_sha256='synthetic-vae'),
            root=root/'fixture-cache')
        recipe = root/'recipe.json'; atomic_json(recipe,config)
        stage = root/'stage.json'
        atomic_json(stage,dict(max_updates=4,max_seconds=7200,save_every=1000,checkpoint_reserve_seconds=600))
        output = root/'run'
        argv = ['train_flow.py','--config',str(recipe),'--stage-budget',str(stage),'--cache',str(cache.root),
            '--output',str(output),'--device','cpu','--execution-policy',str(policy_path),'--profile-first','2']
        def run(arguments):
            with patch.object(train_flow,'LatentCache',return_value=cache),patch.object(sys,'argv',arguments):
                train_flow.main()
        first_update_probes = []
        actual_atomic_checkpoint = train_flow.atomic_checkpoint
        def check_first_write(path, state):
            digest = actual_atomic_checkpoint(path,state)
            if state['completed_updates']==1:
                stored = torch.load(path,weights_only=False)
                assert stored['optimizer']['state'] and {int(s['step']) for s in stored['optimizer']['state'].values()}=={1}
                assert {'model','optimizer','rng','cursor'} <= set(stored)
                assert file_sha(path)==digest
                first_update_probes.append(digest)
            return digest
        with patch.object(train_flow,'atomic_checkpoint',side_effect=check_first_write): run(argv)
        assert len(first_update_probes)==1
        records = [json.loads(line) for path in output.glob('updates-*.jsonl') for line in path.read_text().splitlines()]
        assert [r['attempt_update'] for r in records] == [1,2,3,4]
        assert ['profile' in r for r in records] == [True,True,False,False]
        assert all(r['execution_policy']==policy and r['execution_policy_sha256']==execution['execution_policy_sha256'] for r in records)
        first_latest = json.loads((output/'latest.json').read_text())
        state = torch.load(first_latest['path'],weights_only=False)
        assert state['completed_updates']==4 and state['execution_policy']==policy
        assert state['execution_policy_file_sha256']==file_sha(policy_path)
        assert state['execution_policy_sha256']==execution['execution_policy_sha256']
        index = json.loads((output/'checkpoint-manifest.json').read_text())['checkpoints']
        assert {0,1,3,4} <= {entry['completed_updates'] for entry in index}
        assert all(len(entry['sha256'])==64 for entry in index)
        assert state['optimizer']['state'] and {int(s['step']) for s in state['optimizer']['state'].values()}=={4}
        tests.append('Fresh CLI records policy in logs/checkpoint; profiles exactly first two full updates; fixed update1/3 checkpoints remain despite save_every=1000')

        atomic_json(stage,dict(max_updates=6,max_seconds=7200,save_every=1000,checkpoint_reserve_seconds=600))
        resumed = argv[:-4]+['--execution-policy',str(formatted),'--profile-first','1',
            '--resume',first_latest['path'],'--resume-sha256',first_latest['sha256']]
        run(resumed)
        resumed_records = [json.loads(line) for path in output.glob('updates-from0000004-*.jsonl') for line in path.read_text().splitlines()]
        assert [r['completed_updates'] for r in resumed_records]==[5,6]
        assert [r['attempt_update'] for r in resumed_records]==[1,2]
        assert ['profile' in r for r in resumed_records]==[True,False]
        assert [r['lr'] for r in resumed_records]==[config['lr']*.05,config['lr']*.06]
        latest = json.loads((output/'latest.json').read_text())
        state = torch.load(latest['path'],weights_only=False)
        assert state['completed_updates']==6 and state['cursor']['group']==6
        assert {int(s['step']) for s in state['optimizer']['state'].values()}=={6}
        assert state['execution_policy_file_sha256']==file_sha(formatted)
        assert json.loads((output/'status.json').read_text())['attempt_start_completed_updates']==4
        tests.append('Resume profiles invocation-first updates5/6 without resetting Adam, warmup or cursor; formatting-only policy changes retain canonical identity')

        altered = root/'changed-policy.json'; atomic_json(altered,empty['execution_policy'])
        rejected = copy.copy(resumed)
        rejected[rejected.index('--execution-policy')+1] = str(altered)
        expect(ValueError,lambda:run(rejected))
        failure = json.loads((output/'failure.json').read_text())
        assert failure['phase']=='model_or_resume_setup' and 'policy differs' in failure['error']
        assert json.loads((output/'latest.json').read_text())==latest
        tests.append('Changed execution policy refuses resume before training and leaves latest durable checkpoint unchanged')

        for label,error in (('oom',torch.cuda.OutOfMemoryError('synthetic CPU OOM path')),
                ('nonfinite',FloatingPointError('synthetic nonfinite path'))):
            failed_output = root/label
            failed_args = copy.copy(argv)
            failed_args[failed_args.index('--output')+1] = str(failed_output)
            with patch.object(train_flow,'train_update',side_effect=error):
                expect(type(error),lambda:run(failed_args))
            failure = json.loads((failed_output/'failure.json').read_text())
            assert failure['completed_updates']==0 and failure['last_durable_checkpoint']['completed_updates']==0
            assert failure['execution_policy']==policy and 'fallback' in failure['note']
            assert not (failed_output/'status.json').exists()
        tests.append('Injected CPU OOM/nonfinite exceptions record failure and policy, preserve last durable state, and never claim completion or fall back')
    evidence = dict(passed=True,device='cpu',tests=tests,synthetic_optimizer_updates=6,
        real_mesh_optimizer_updates=0,gpu_used=False,production_scale_tested=False,temporary_fixtures_removed=True)
    atomic_json(Path(__file__).resolve().parent.parent/'evidence/training_execution_cpu_tests.json',evidence)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__': main()
