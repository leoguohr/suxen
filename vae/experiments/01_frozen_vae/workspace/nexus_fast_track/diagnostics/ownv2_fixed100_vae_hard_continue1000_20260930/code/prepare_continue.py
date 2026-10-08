"""Verify the complete B parent and continued CPU schedule; no GPU initialization."""
import argparse
import math
from pathlib import Path
from run_support import torch, read, write, sha, tensor_hash
from data_objective import load_dataset, epoch_batches
from pair_negatives import load_hard_pool, select_negatives, self_test
from train_continue import START, UPDATES, BETA, MILESTONES, EVAL_SEEDS, PARENT_SHA, branch_cursor

ROOT = Path(__file__).resolve().parent.parent
PARENT_ROOT = ROOT.parent/'ownv2_fixed100_vae_negative_pair_20260930'


def main(gpu_uuid):
    assert not torch.cuda.is_initialized()
    self_test()
    parent = PARENT_ROOT/'B_hard/checkpoints/vae-0500.pt'
    assert sha(parent) == PARENT_SHA
    cp = torch.load(parent, map_location='cpu', weights_only=False)
    assert cp['completed_updates'] == START and cp['beta'] == BETA and cp['no_scheduler']
    assert cp['config']['branch'] == 'B_hard'
    assert cp['code_sha256'] == {p.name: sha(p) for p in sorted((PARENT_ROOT/'code').glob('*.py'))}
    items, data = load_dataset(cp['config']['data_source'])
    assert cp['data'] == data and cp['participation'] == {u: START//20 for u in items}
    hard_path = ROOT/'hard_pool.json'
    assert sha(hard_path) == sha(PARENT_ROOT/'hard_pool.json') == cp['hard_pool_sha256']
    pool_source = read(hard_path)['parent_checkpoint_sha256']
    pools, pool = load_hard_pool(hard_path, items, pool_source)
    assert cp['cursor'] == branch_cursor(items, data, START, cp['config'], pools)
    assert cp['model_config']['model_variant'] == 'B_v2_teacher_blocks'
    groups = cp['optimizer']['param_groups']
    assert [len(g['params']) for g in groups] == [412, 2]
    steps = [sorted({int(cp['optimizer']['state'][p]['step']) for p in g['params']}) for g in groups]
    assert steps == [[35220], [1000]]
    for g in groups:
        assert (g['lr'], g['betas'], g['eps'], g['weight_decay'], g['foreach']) == (1e-4, (.9, .999), 1e-8, .01, True)
    baseline_path = PARENT_ROOT/'B_hard/evaluations/update-00035220/mu/complete.json'
    baseline = read(baseline_path)
    assert baseline['complete'] and baseline['identity']['checkpoint_sha256'] == PARENT_SHA
    assert (baseline['edge']['fp'], baseline['edge']['fn'], baseline['face']['fp'], baseline['face']['fn'], baseline['joint_perfect']) == (97, 52, 1430, 98, 31)
    for seed in EVAL_SEEDS:
        result = read(baseline_path.parent.parent/f'noise-{seed}'/'complete.json')
        assert result['complete'] and result['identity']['checkpoint_sha256'] == PARENT_SHA
        assert result['identity']['evaluation_seed'] == seed and len(result['meshes']) == 100
    schedule, counts = [], {u: 0 for u in items}
    # Compute each epoch/UID selection once. Stateless epoch/UID RNG never touches training RNG.
    for epoch in range(START//20, (START+UPDATES)//20):
        batches = epoch_batches(data['uids'], epoch, 0)
        assert len(batches) == 20 and len(set(sum(batches, []))) == 100
        for group, batch in enumerate(batches):
            for uid in batch:
                selected, metadata = select_negatives(items[uid], epoch, 'B_hard', pools)
                n, f = len(items[uid]['vertices']), len(items[uid]['gt_faces'])
                assert len(selected) == min(math.ceil(1.5*f), math.comb(n, 3)-f)
                schedule.append(dict(update=epoch*20+group+1, epoch=epoch, group=group, uid=uid, B=metadata))
                counts[uid] += 1
    assert len(schedule) == UPDATES*5 and set(counts.values()) == {50}
    first = schedule[:5]
    assert [r['uid'] for r in first] == cp['cursor']['next_uids']
    assert {r['uid']: r['B']['negative_sha256'] for r in first} == cp['cursor']['next_negative_hashes']
    write(ROOT/'negative_schedule.json', schedule)
    config = dict(cp['config'])
    config.update(gpu_uuid=gpu_uuid, parent_checkpoint=str(parent), parent_sha256=PARENT_SHA,
        parent_model_sha256=tensor_hash(cp['model']), parent_code_hashes=cp['code_sha256'],
        parent_baseline=str(baseline_path), parent_baseline_sha256=sha(baseline_path),
        run_directory=str(ROOT/'run'), hard_pool_path=str(hard_path), hard_pool_sha256=sha(hard_path),
        hard_pool_source_sha256=pool_source, negative_schedule_path=str(ROOT/'negative_schedule.json'),
        negative_schedule_sha256=sha(ROOT/'negative_schedule.json'), source_step=START,
        new_updates=UPDATES, mu_checkpoints=list(MILESTONES), noise_checkpoints=[0, UPDATES],
        train_noise='Continue B35220 training generator exactly; no reseeding or replay',
        primary_metric='actual Face micro-F1 >= 0.997; mu and each noise condition reported separately',
        strict_zero_error_is_required_for_stage_pass=False, reused_step0_noise_evaluations=True,
        authorization='User: continue B35220 for1000 effective updates to36220, then stop; use the one available GPU; no persistent assistant supervision after startup.')
    write(ROOT/'config.json', config)
    write(ROOT/'evidence/preparation.json', dict(cpu_only=True, cuda_initialized=torch.cuda.is_initialized(),
        parent_checkpoint=str(parent), parent_sha256=PARENT_SHA, parent_model_sha256=config['parent_model_sha256'],
        source_update=START, target_update=START+UPDATES, optimizer_steps=steps,
        optimizer_options=[{k:v for k,v in g.items() if k != 'params'} for g in groups],
        data=data, hard_pool_sha256=sha(hard_path), hard_negatives=1509, hard_pool_refreshed=False,
        planned_updates=UPDATES, planned_participations=len(schedule), each_mesh_new_participations=50,
        first_batch_matches_parent_cursor=True, next_negative_hashes_match_parent_cursor=True,
        unchanged_modules={p.name: sha(p) for p in (ROOT/'code').glob('*.py')
                           if (PARENT_ROOT/'code'/p.name).exists() and sha(p) == sha(PARENT_ROOT/'code'/p.name)}))
    assert not torch.cuda.is_initialized()
    print('CPU preparation passed: complete B35220, Adam35220/1000, same1509 negatives, next cursor exact,1000 updates/5000 participations.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-uuid', required=True)
    main(parser.parse_args().gpu_uuid)
