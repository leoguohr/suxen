"""CPU-only preparation of explicit XYZ recovery; training authorization remains pending."""
import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protocol import (LIMIT, MILESTONES, PARENTS, ROOT, START, acquire, check_log,
    info, read, require, verify_code, write)

RESTORED = 9000
OLD_LAST = {'XYZ_LN_post': 9989, 'XYZ_LN_pre': 9977}


def copy_log_prefix(source, destination, restored):
    """Preserve exact prefix bytes, validate the entire old log, never overwrite."""
    expected = START+1
    with Path(source).open('rb') as original, Path(destination).open('xb') as prefix:
        for line in original:
            row = json.loads(line)
            require(row['update_after'] == expected and row['update_before'] == expected-1,
                    'Noncontiguous/duplicate original update log')
            if expected <= restored:
                prefix.write(line)
            expected += 1
        require(expected-1 >= restored, 'Original log does not reach restored checkpoint')
        prefix.flush()
        os.fsync(prefix.fileno())
    check_log(destination, restored)
    return dict(old_log_count=expected-1-START, old_last_update=expected-1,
                restored_update=restored, lost_updates=expected-1-restored)


def audit_original(folder, arm, last, code):
    require(not (folder/'complete.json').exists(), f'{arm}: branch already complete')
    require(not (folder/'failure.json').exists(), f'{arm}: failure requires separate review')
    cfg = read(folder/'config.json')
    require(cfg['arm'] == arm and cfg['source_code'] == code and cfg['max_updates'] == LIMIT
            and cfg['start_updates'] == START and cfg['checkpoints'] == list(MILESTONES),
            f'{arm}: original continuation contract differs')
    baseline = read(folder/'startup-baseline-verification.json')
    require(baseline['passed'] is True and baseline['source_code'] == code
            and baseline['parent_sha256'] == cfg['parent']['sha256'] == PARENTS[arm]['sha256'],
            f'{arm}: original startup verification differs')
    require(read(folder/'status.json')['completed_updates'] == last, f'{arm}: status boundary changed')
    check_log(folder/'updates.jsonl', last)
    # Refuse to discard a newly discovered durable tail. Inspection is read-only.
    for path in folder.rglob('*.pt'):
        require(path.name not in ('final.pt',) and not path.name.startswith('boundary-'),
                f'{arm}: additional durable state requires review: {path}')
        if path.name.startswith('checkpoint-'):
            require(int(path.stem.split('-')[1]) <= RESTORED,
                    f'{arm}: newer durable checkpoint requires review: {path}')
    sidecar = read(folder/'checkpoint-09000.json')
    require(sidecar['resumable'] is True and sidecar['completed_updates'] == RESTORED,
            f'{arm}: checkpoint is not a resumable 9000 boundary')
    checkpoint = info(folder/'checkpoint-09000.pt', sidecar['sha256'])
    best = read(folder/'best.json')
    require(type(best['completed_updates']) is int and START <= best['completed_updates'] <= RESTORED,
            f'{arm}: best contains a newer durable state requiring review')
    best_file = info(folder/'best.pt', best['checkpoint']['sha256'])
    old_best = info(best['checkpoint']['path'], best_file['sha256'])
    require(folder in Path(old_best['path']).parents, f'{arm}: best checkpoint is outside original branch')
    evaluation_path = Path(best['evaluation_path']).resolve()
    require(folder in evaluation_path.parents, f'{arm}: best evaluation is outside original branch')
    result = read(evaluation_path)
    require(result['complete'] is True and len(result['meshes']) == 50
            and result['checkpoint']['sha256'] == best['evaluation_checkpoint']['sha256'],
            f'{arm}: best evaluation metadata is invalid')
    names = ['config.json', 'startup-baseline-verification.json', 'checkpoint-09000.json', 'best.json']
    files = {name: info(folder/name) for name in names}
    files.update({'checkpoint-09000.pt': checkpoint, 'best.pt': best_file,
                  'updates.jsonl': info(folder/'updates.jsonl')})
    return files


def main(args):
    runs = ROOT/'runs'
    destination = Path(args.outdir).resolve()
    require(ROOT in destination.parents and destination != runs and runs not in destination.parents,
            'Recovery output must be inside ROOT and outside original runs')
    require(not destination.exists(), 'Recovery output already exists; no overwrite')
    with ExitStack() as stack:
        for lock_path in [runs/'controller.lock']+[runs/arm/'execution.lock' for arm in OLD_LAST]:
            require(lock_path.is_file(), f'Original lock file missing: {lock_path}')
            stack.enter_context(acquire(lock_path))
        code = verify_code()
        originals = {arm: audit_original((runs/arm).resolve(), arm, last, code)
                     for arm, last in OLD_LAST.items()}
        destination.mkdir(parents=True, exist_ok=False)
        branches = {}
        for arm, files in originals.items():
            folder = destination/arm
            folder.mkdir()
            for name, entry in files.items():
                if name == 'updates.jsonl':
                    continue
                if name.endswith('.pt'):
                    os.link(entry['path'], folder/name)  # No cross-filesystem copy fallback.
                else:
                    shutil.copy2(entry['path'], folder/name)
                info(folder/name, entry['sha256'])
            summary = copy_log_prefix(files['updates.jsonl']['path'], folder/'updates.jsonl', RESTORED)
            require(summary['old_last_update'] == OLD_LAST[arm], f'{arm}: original tail changed')
            for entry in files.values():
                info(entry['path'], entry['sha256'])
            branches[arm] = dict(summary, original_directory=str(runs/arm),
                recovery_directory=str(folder), original_files=files,
                resume=info(folder/'checkpoint-09000.pt'), prefix_log=info(folder/'updates.jsonl'),
                proposed_updates=LIMIT-RESTORED, authorized_additional_updates=LIMIT-START,
                original_actual_additional_updates_if_finished=summary['old_log_count']+LIMIT-RESTORED,
                best_metadata_preserved_exactly=True, best_paths_remain_in_original_branch=True)
        manifest = dict(schema='cad50_xyz_explicit_recovery_v1', state='prepared',
            authorization='pending', worker_launch_performed=False, original_runs_read_only=True,
            actual_full_checkpoint_restore_verified=False, source_code=code,
            management_source=info(__file__), branches=branches,
            warning='Recovering from 9000 repeats logged tail work and exceeds the original actual update budget; explicit authorization required.')
        write(destination/'RECOVERY.json', manifest)
        return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--outdir', required=True, help='New independent directory inside ROOT, outside runs')
    result = main(parser.parse_args())
    print(json.dumps(dict(recovery=str(Path(result['branches']['XYZ_LN_post']['recovery_directory']).parent/'RECOVERY.json'),
                          authorization=result['authorization'], worker_launch_performed=False), indent=2))
