"""CPU-only parent and sampling preparation; no optimizer update or network forward."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from run_support import torch, read, write, sha, tensor_hash
from data_objective import load_dataset, epoch_batches, array_sha, negative_faces
from runtime_fixed100 import next_state
from pair_negatives import select_negatives, load_hard_pool, self_test

PARENT_SHA = '7fb1e2a5d128762e147a4b87524d4cc212909625b706841c80e3519576e05562'


def main(parent_root, root):
    assert not torch.cuda.is_initialized()
    self_test()
    parent = parent_root/'run/checkpoints/vae-0500.pt'
    assert sha(parent) == PARENT_SHA
    cp = torch.load(parent, map_location='cpu', weights_only=False)
    assert cp['completed_updates'] == 34720 and cp['beta'] == 1e-6 and cp['no_scheduler']
    assert cp['model_config']['model_variant'] == 'B_v2_teacher_blocks'
    assert cp['code_sha256'] == {p.name:sha(p) for p in sorted((parent_root/'code').glob('*.py'))}
    items, data = load_dataset(cp['config']['data_source'])
    assert data == cp['data'] and cp['cursor'] == next_state(items, data['uids'], 34720, 0)
    assert cp['participation'] == {u: 1736 for u in items}
    groups = cp['optimizer']['param_groups']
    assert len(groups) == 2 and [len(g['params']) for g in groups] == [412, 2]
    for g in groups:
        assert g['lr'] == 1e-4 and g['betas'] == (.9, .999) and g['eps'] == 1e-8
        assert g['weight_decay'] == .01 and g['foreach'] is True
    steps = [sorted({int(cp['optimizer']['state'][p]['step']) for p in g['params']}) for g in groups]
    assert steps == [[34720], [500]]
    assert cp['optimizer_group_parameter_names'][1] == ['log_variance.weight', 'log_variance.bias']
    baseline_path = parent_root/'run/evaluations/update-00034720/mu/complete.json'
    baseline = read(baseline_path)
    assert baseline['complete'] and baseline['identity']['checkpoint_sha256'] == PARENT_SHA
    assert (baseline['edge']['fp'], baseline['edge']['fn'], baseline['face']['fp'], baseline['face']['fn'], baseline['joint_perfect']) == (59,28,1509,58,28)
    pool = dict(parent_checkpoint_sha256=PARENT_SHA, source_condition='mu',
                source_baseline=str(baseline_path), source_baseline_sha256=sha(baseline_path),
                selection_rule='descending parent mu logit; lexicographic triple tie break; cap GT face count',
                refresh=False, evaluation_noise_used=False, meshes={}, selected_total=0)
    for uid, item in items.items():
        directory = baseline_path.parent/uid
        state = read(directory/'face_shards/progress.json')
        assert state['complete'] and state['binding']['identity'] == dict(baseline['identity'],uid=uid)
        scored = []
        shards = []
        gt = set(map(tuple, item['gt_faces'].tolist()))
        for index in range(state['shards']):
            path = directory/f'face_shards/part-{index:08d}.npz'
            digest = sha(path)
            if index == state['shards']-1: assert digest == state['last_shard_sha256']
            shards.append(dict(path=str(path),sha256=digest))
            with np.load(path, allow_pickle=False) as z:
                assert np.isfinite(z['logits']).all()
                assert np.array_equal(z['labels'], np.array([tuple(t) in gt for t in z['ids']]))
                mask = (z['logits'] > 0) & ~z['labels']
                scored.extend((float(v), tuple(map(int,t))) for t,v in zip(z['ids'][mask],z['logits'][mask]))
        assert len(scored) == state['fp'] and len({t for _,t in scored}) == len(scored)
        scored.sort(key=lambda x: (-x[0], x[1]))
        chosen = scored[:len(gt)]
        assert len(chosen) == len(scored), 'This experiment expects all1509 parent FP fit the cap'
        ids = np.asarray([t for _,t in chosen], dtype=np.int64).reshape(-1,3)
        pool['meshes'][uid] = dict(vertices=len(item['vertices']),gt_faces=len(gt),parent_mu_fp=len(scored),
            ids=ids.tolist(),logits=[v for v,_ in chosen],ids_sha256=array_sha(ids),source_shards=shards)
        pool['selected_total'] += len(chosen)
    assert pool['selected_total'] == 1509
    write(root/'hard_pool.json',pool)
    pools, manifest = load_hard_pool(root/'hard_pool.json', items, PARENT_SHA)
    plan = []
    for completed in range(34720,35220):
        epoch, group = divmod(completed,20)
        batch = epoch_batches(data['uids'],epoch,0)[group]
        for uid in batch:
            a,ma = select_negatives(items[uid],epoch,'A_uniform',pools)
            b,mb = select_negatives(items[uid],epoch,'B_hard',pools)
            assert len(a)==len(b) and ma['uniform_negative_sha256']==mb['uniform_negative_sha256']
            assert len(b) == min(int(np.ceil(1.5*len(items[uid]['gt_faces']))),
                                 len(a))
            plan.append(dict(update=completed+1,epoch=epoch,group=group,uid=uid,A=ma,B=mb))
    write(root/'negative_schedule.json',plan)
    evidence = dict(parent_checkpoint=str(parent),parent_sha256=PARENT_SHA,parent_bytes=parent.stat().st_size,
        completed_updates=34720,parent_model_sha256=tensor_hash(cp['model']),
        parent_code_hashes=cp['code_sha256'],data=data,optimizer_steps=steps,
        optimizer_group_parameter_names=cp['optimizer_group_parameter_names'],
        optimizer_options=[{k:v for k,v in g.items() if k!='params'} for g in groups],
        train_noise_rng_sha256=hashlib.sha256(cp['train_noise_rng'].numpy().tobytes()).hexdigest(),
        hard_pool_sha256=sha(root/'hard_pool.json'),negative_schedule_sha256=sha(root/'negative_schedule.json'),
        hard_negatives=1509,meshes_with_hard_negatives=sum(bool(len(v)) for v in pools.values()),
        planned_mesh_participations=len(plan),parent_baseline=str(baseline_path),
        parent_baseline_sha256=sha(baseline_path),cpu_only=True,cuda_initialized=torch.cuda.is_initialized())
    assert not evidence['cuda_initialized'] and len(plan)==2500
    write(root/'evidence/preparation.json',evidence)
    print(json.dumps({k:evidence[k] for k in ['completed_updates','optimizer_steps','hard_negatives','meshes_with_hard_negatives','planned_mesh_participations','cuda_initialized']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent-root',type=Path,required=True);p.add_argument('--root',type=Path,required=True)
    a=p.parse_args();main(a.parent_root,a.root)
