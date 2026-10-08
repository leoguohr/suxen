"""Read original interrupted evidence; create an isolated, budgeted recovery."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import copy
import shutil
import fcntl
from run_support import *
from native_models import NativeTopologyAE, Config
from data_objective import load_dataset

ORIGINAL = ROOT.with_name('own512_v2_recipe_pair_20260923')
VARIANTS = ['A_v1_recipe_control', 'B_v2_teacher_blocks']
GPU = 'GPU-0a371e11-9041-9b22-d2d5-8469622c4681'


def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'RECOVERY_PLAN.json').exists(), 'Refuse duplicate preparation'
    configure()
    locks = []
    for relative in ['queue/worker.lock','repro_outputs/unattended.lock'] + [v+'/execution.lock' for v in VARIANTS]:
        lock = (ORIGINAL/relative).open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(lock)
    for name in ['data','pools']:
        (ROOT/name).symlink_to((ORIGINAL/name).resolve(), target_is_directory=True)
    for name in ['reference','frozen_code']:
        shutil.copytree(ORIGINAL/name, ROOT/name)
    shutil.copytree(ORIGINAL/'repro_outputs', ROOT/'provenance/original_repro_outputs')
    for name in ['SOURCE_COMMIT.json','INITIALIZATION.json','CORE_TESTS.json','SKILL_LOAD.json']:
        shutil.copy2(ORIGINAL/'repro_outputs'/name, OUT/name)
    (ROOT/'queue').mkdir(exist_ok=True)
    cpu_items, manifest = load_dataset(ROOT/'data', ROOT/'pools')
    plan = dict(original_root=str(ORIGINAL), recovery_root=str(ROOT), gpu_uuid=GPU,
        node=os.uname().nodename, created=time.time(), physical_update_cap_per_branch=20000,
        max_gpu_hours_including_original=24, extra_uncertain_update_reserved_per_branch=1,
        old_locks_free=True, branches={}, unchanged_runtime_code=code_hashes())
    for variant in VARIANTS:
        old, new = ORIGINAL/variant, ROOT/variant
        new.mkdir()
        for name in ['config.json','preflight.json','ready.json']:
            shutil.copy2(old/name, new/name)
        shutil.copytree(old/'evaluations', new/'evaluations')
        for pattern in ['checkpoint-*.json','best_*.json','TARGET_REACHED.json']:
            for path in old.glob(pattern): shutil.copy2(path, new/path.name)
        receipt = read(sorted(old.glob('checkpoint-*.json'))[-1])
        assert sha(receipt['path']) == receipt['sha256']
        cp = torch.load(receipt['path'], map_location='cpu', mmap=True, weights_only=False)
        start = cp['completed_updates']; cfg = cp['config']
        assert cp['model_variant'] == variant and cfg['data'] == manifest
        assert str(torch.__version__) == cfg['torch'] and torch.version.cuda == cfg['cuda']
        assert cfg['code_sha256'] == code_hashes()
        with torch.device('meta'): model = NativeTopologyAE(Config(**cp['model_config']))
        names = [n for n,p in model.named_parameters() if p.requires_grad]
        assert names == cfg['optimizer_parameter_names']
        state = model.state_dict()
        assert state.keys() == cp['model'].keys()
        assert all(state[n].shape == cp['model'][n].shape for n in state)
        groups = cp['optimizer']['param_groups']; assert len(groups) == 1
        group = groups[0]
        assert len(group['params']) == len(names) and group['lr'] == 1e-4
        assert tuple(group['betas']) == (.9,.999) and group['eps'] == 1e-8 and group['weight_decay'] == .01
        for name, pid in zip(names,group['params']):
            slot = cp['optimizer']['state'][pid]
            assert int(slot['step']) == start
            for key in ['exp_avg','exp_avg_sq']:
                assert slot[key].shape == cp['model'][name].shape and torch.isfinite(slot[key]).all()
        assert {'python','numpy','torch','cuda'} <= cp['rng'].keys() and len(cp['rng']['cuda']) == 1
        assert cp['data_generator']['next_epoch'] == start//10 and cp['data_generator']['next_batch'] == start%10
        assert sum(cp['participation'].values()) == start*5
        original_bytes = (old/'updates.jsonl').read_bytes()
        log_lines = original_bytes.splitlines(keepends=True)
        records = [json.loads(line) for line in log_lines]
        last = records[-1]['update']
        assert all(row['update'] == i for i,row in enumerate(records,1))
        assert last == read(old/'status.json')['completed_updates']
        (new/'updates.jsonl').write_bytes(b''.join(log_lines[:start]))
        lost = last-start
        plan['branches'][variant] = dict(parent=receipt, parent_step=start,
            original_recorded_updates=last, unsaved_recorded_updates=lost,
            conservative_original_updates=last+1, possible_unlogged_updates_reserved=1,
            total_recovery_update_allowance=20000-last-1, restored_lineage_log_prefix_steps=start,
            original_log_sha256=hashlib.sha256(original_bytes).hexdigest(),
            complete_model_optimizer_rng_verified=True, optimizer_slot_count=len(names))
        write(new/'recovery_current.json',dict(step=start,checkpoint=receipt))
        for request in sorted((ORIGINAL/'queue').glob('*.json')):
            if variant not in request.name:continue
            q = read(request)
            if not (old/'evaluations'/f"eval-{q['step']:05d}.json").exists():
                assert q['step'] <= start and q.get('kind','evaluation') == 'evaluation'
                write(ROOT/'queue'/request.name, q)
        del cp, model
    plan['common_logical_stop'] = min(v['parent_step']+v['total_recovery_update_allowance'] for v in plan['branches'].values())
    # Conservatively charge even disconnected time up to this read-only recovery.
    resources = read(ORIGINAL/'repro_outputs/RESOURCE_ACCOUNT.json')
    write(OUT/'ORIGINAL_RESOURCE_SNAPSHOT.json',resources)
    now = time.time()
    for job in resources['jobs'].values():
        if job['status'] == 'active':
            job.update(status='interrupted',finished=now,recovery_note='Upper-bound accounting; includes offline time before recovery verified old locks free')
        job['charged_seconds'] = job['finished']-job['started']
    resources['total_charged_gpu_seconds'] = sum(j['charged_seconds'] for j in resources['jobs'].values())
    assert resources['total_charged_gpu_seconds'] < TRAIN_GPU_SECONDS
    plan['inherited_conservative_gpu_seconds'] = resources['total_charged_gpu_seconds']
    write(OUT/'RESOURCE_ACCOUNT.json',resources)
    write(OUT/'RECOVERY_PLAN.json',plan)
    print(json.dumps(plan),flush=True)


if __name__ == '__main__':main()
