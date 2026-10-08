"""Run the authorized C1 then C2 campaign; never resets its existing 4-hour clock."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def check_gpu(uuid):
    def query(arguments):
        return subprocess.check_output(['nvidia-smi', *arguments], text=True, timeout=15)
    available = query(['--query-gpu=uuid', '--format=csv,noheader']).splitlines()
    if uuid not in [line.strip() for line in available]:
        raise RuntimeError('Authorized GPU UUID is not present')
    occupied = query(['--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    if any(line.split(',')[0].strip() == uuid for line in occupied.splitlines()):
        raise RuntimeError('Authorized GPU is occupied; refusing to interfere')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'cache', 'baseline', 'vae-checkpoint', 'gpu-uuid'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    for name in ('cache', 'baseline', 'vae_checkpoint'):
        setattr(args, name, str(Path(getattr(args, name)).resolve()))
    root = Path(args.root).resolve()
    code = root/'code'
    policy = root/'perf/execution_policy.json'
    with (root/'CANDIDATES.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        campaign = read(root/'campaign.json')
        started, deadline = campaign['started_unix'], campaign['deadline_unix']
        if (campaign['max_gpu_seconds'] != 14400 or campaign['max_updates_per_group'] != 8000
                or not all(math.isfinite(v) for v in (started, deadline))
                or deadline <= started or deadline-started > 14400.001
                or started > time.time()+1 or campaign['authorized_gpu_uuid'] != args.gpu_uuid
                or not args.gpu_uuid.startswith('GPU-')):
            raise ValueError('Invalid existing campaign clock or GPU authorization')
        # Wall time across all phases, including the completed foreground benchmark,
        # is a conservative upper bound on this serial campaign's GPU reservation.
        monotonic_deadline = time.monotonic()+deadline-time.time()
        def remaining():
            return max(0., min(deadline-time.time(), monotonic_deadline-time.monotonic()))
        if (root/'runner-receipt.json').exists():
            raise RuntimeError('Campaign already attempted; no automatic clock/cursor reset or replay')
        execution = read(policy)
        if set(execution) != {'flow_direct_blocks', 'condition_direct_blocks'}:
            raise ValueError('Missing or invalid completed benchmark execution policy')
        benchmark = read(root/'perf/report.json')
        if (benchmark.get('complete') is not True or benchmark.get('adam_steps') != [0]
                or benchmark.get('selected_policy') != execution):
            raise ValueError('Benchmark must complete without Adam steps and select this execution policy')
        for directory, pattern, key in ((code, '*.py', 'source_code_sha256'),
                (root/'configs', '*.json', 'config_sha256')):
            if {p.name: digest(p) for p in sorted(directory.glob(pattern))} != campaign[key]:
                raise ValueError('Uploaded files differ from the campaign: '+key)
        for path in (Path(args.cache)/'manifest.json', Path(args.baseline)/'summary.json',
                     Path(args.vae_checkpoint), code/'train_flow.py', code/'evaluate_flow.py'):
            if not path.is_file(): raise FileNotFoundError(path)
        configs = {'c1': root/'configs/c1_fourier.json', 'c2': root/'configs/c2_teacher.json'}
        config_sha = {group: digest(path) for group, path in configs.items()}
        check_gpu(args.gpu_uuid)
        receipt = dict(pid=os.getpid(), started_unix=time.time(), campaign_sha256=digest(root/'campaign.json'),
            original_started_unix=started, deadline_unix=deadline, max_gpu_seconds=14400,
            per_group_max_optimizer_updates=8000,
            authorized_gpu_uuid=args.gpu_uuid, execution_policy_sha256=digest(policy),
            config_sha256=config_sha, runner_sha256=digest(__file__),
            order=['c1_train', 'c1_eval500', 'c1_eval1000', 'c1_eval_final_if_distinct',
                   'c2_train', 'c2_eval500', 'c2_eval1000', 'c2_eval_final_if_distinct'],
            benchmark='Already performed in foreground; its time remains inside the original campaign clock',
            accounting='Serial campaign wall deadline, including setup/saving/evaluation; separate group update cursors')
        status = dict(state='started', groups={g: dict(state='pending') for g in ('c1', 'c2')},
            phases=[], deadline_unix=deadline)
        write(root/'runner-receipt.json', receipt)
        def commit():
            status.update(updated_unix=time.time(), remaining_seconds=remaining())
            write(root/'runner-status.json', status)
        commit()
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid,
            PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python', PYTHONUNBUFFERED='1')

        def run_phase(name, arguments, stop_directory, ledger, phase_left):
            if (root/'STOP').exists(): raise InterruptedError('Campaign STOP requested')
            if phase_left() <= 75: raise InterruptedError('Phase shutdown reserve reached')
            check_gpu(args.gpu_uuid)
            if read(ledger).get('active') is not None:
                raise RuntimeError('Unclosed group GPU attempt; no automatic reconciliation')
            command = [sys.executable, '-B', *map(str, arguments)]
            record = dict(name=name, command=command, log=str(root/(name+'.log')), started_unix=time.time())
            status['phases'].append(record); status['state']=name; commit()
            print('Starting '+name, flush=True)
            with (root/(name+'.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=root, env=environment,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record['pid']=child.pid; commit()
                try:
                    while child.poll() is None:
                        if (root/'STOP').exists() and Path(stop_directory).is_dir():
                            (Path(stop_directory)/'STOP').touch(exist_ok=True)
                        if phase_left() <= 60:
                            raise TimeoutError('Absolute phase/campaign deadline minus 60 seconds reached')
                        try: child.wait(timeout=min(5., max(.1, phase_left()-60)))
                        except subprocess.TimeoutExpired: pass
                except BaseException:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try: child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL); child.wait()
                    record.update(returncode=child.returncode, ended_unix=time.time(), interrupted=True)
                    commit()
                    raise
            record.update(returncode=child.returncode, ended_unix=time.time())
            record['gpu_budget']=read(ledger)
            commit()
            if child.returncode != 0 or record['gpu_budget'].get('active') is not None:
                raise RuntimeError(name+' failed; saved progress retained, no retry')
            if (root/'STOP').exists(): raise InterruptedError('Campaign STOP requested')

        try:
            for group in ('c1', 'c2'):
                # This permits a bounded partial C2, without promising 8000 updates.
                if remaining() < 1200 or (root/'STOP').exists():
                    status['groups'][group]=dict(state='not_started', reason='insufficient_time_or_STOP')
                    commit(); continue
                allowance = remaining()/2 if group == 'c1' else remaining()
                group_deadline = time.time()+allowance
                group_monotonic_deadline = time.monotonic()+allowance
                def group_left():
                    return max(0., min(remaining(), group_deadline-time.time(),
                        group_monotonic_deadline-time.monotonic()))
                evaluation_reserve = 1400
                stage = dict(max_updates=8000, max_seconds=int(group_left()-60-evaluation_reserve),
                    save_every=8000, checkpoint_reserve_seconds=600)
                if stage['max_seconds'] <= stage['checkpoint_reserve_seconds']:
                    status['groups'][group]=dict(state='not_started', reason='insufficient_training_save_evaluation_reserves')
                    commit(); continue
                directory = root/group
                directory.mkdir(exist_ok=False)
                ledger = directory/'budget.json'
                write(ledger, dict(schema_version=1, authorized_gpu_uuid=args.gpu_uuid,
                    max_gpu_seconds=max(1, int(group_left()-60)), max_optimizer_updates=8000,
                    charged_gpu_seconds=0., effective_optimizer_updates=0, last_completed_updates=0,
                    attempts=[], active=None, campaign_sha256=receipt['campaign_sha256'], group=group))
                write(directory/'stage.json', stage)
                result = dict(state='training', deadline_unix=group_deadline,
                    evaluation_reserve_seconds=evaluation_reserve, evaluations={})
                status['groups'][group]=result; commit()
                run = directory/'run'
                run_phase(group+'_train', [code/'train_flow.py', '--config', configs[group],
                    '--stage-budget', directory/'stage.json', '--execution-policy', policy,
                    '--budget-file', ledger, '--cache', args.cache, '--output', run, '--device', 'cuda:0'], run, ledger, group_left)
                training = read(run/'status.json')
                completed = training.get('completed_updates', 0)
                if not isinstance(completed, int) or not 0 <= completed <= 8000:
                    raise ValueError('Invalid training completed_updates')
                if completed != read(ledger)['last_completed_updates']:
                    raise ValueError('Training status/ledger cursor mismatch')
                if bool(training.get('complete')) != (completed == 8000):
                    raise ValueError('Training completion flag contradicts update count')
                result.update(state='evaluating', training=training)
                entries = read(run/'checkpoint-manifest.json')['checkpoints']
                evaluation_deadline = time.monotonic()+1400
                def evaluation_left():
                    return max(0., min(group_left(), evaluation_deadline-time.monotonic()))
                def evaluate(checkpoint, seed):
                    update = checkpoint['completed_updates']
                    key = str(update)+('-seed1' if seed else '')
                    destination = directory/('eval'+key)
                    if evaluation_left() <= 150:
                        result['evaluations'][key]=dict(complete=False, seed=seed,
                            checkpoint_sha256=checkpoint['sha256'], reason='insufficient_shared_evaluation_time')
                        commit(); return
                    path = Path(checkpoint['path']).resolve()
                    if path.parent != (run/'checkpoints').resolve() or path.stat().st_size != checkpoint['bytes']:
                        raise ValueError('Checkpoint manifest path or size mismatch')
                    seconds = min(1000, max(1, int(evaluation_left()-75)))
                    run_phase(group+'_eval'+key, [code/'evaluate_flow.py', '--checkpoint', path,
                        '--checkpoint-sha256', checkpoint['sha256'], '--vae-checkpoint', args.vae_checkpoint,
                        '--cache', args.cache, '--vae-baseline', args.baseline, '--output', destination,
                        '--device', 'cuda:0', '--seed', str(seed), '--steps', '50', '--max-seconds', seconds,
                        '--budget-file', ledger], destination, ledger, evaluation_left)
                    summary = read(destination/'summary.json')
                    if summary.get('complete') and (summary.get('expected_meshes') != 50
                            or len(set(summary.get('evaluated_uids', []))) != 50):
                        raise ValueError('Complete evaluation does not cover all 50 unique UIDs')
                    result['evaluations'][key]=dict(complete=bool(summary.get('complete')),
                        evaluated_uids=summary.get('evaluated_uids', []), summary=str(destination/'summary.json'),
                        checkpoint_sha256=checkpoint['sha256'], seed=seed, euler_steps=50,
                        metrics=summary.get('metrics') if summary.get('complete') else None,
                        note=None if summary.get('complete') else 'Incomplete; no full-50 score')
                    commit()
                for update in [500, 1000]+([completed] if completed > 0 and completed not in (500, 1000) else []):
                    matches = [v for v in entries if v.get('retained', True) and v['completed_updates']==update]
                    if not matches:
                        result['evaluations'][str(update)]=dict(complete=False, reason='checkpoint_not_reached')
                        commit(); continue
                    checkpoint = matches[-1]
                    evaluate(checkpoint, 0)
                    primary = result['evaluations'][str(update)]
                    if primary['complete'] and primary['metrics'].get('strict50_under_this_noise') is True:
                        protection = dict(reason='strict50_under_seed0', checkpoint_sha256=checkpoint['sha256'],
                            completed_updates=update, evaluation=primary['summary'])
                        write(Path(checkpoint['path']).with_suffix('.protect.json'), protection)
                        result.setdefault('first_strict', protection)
                        evaluate(checkpoint, 1)
                result['state']='finished' if completed==8000 and all(
                    item['complete'] for item in result['evaluations'].values()) else 'incomplete_with_saved_progress'
                commit()
            status['state']='finished' if all(v['state']=='finished' for v in status['groups'].values()) else 'incomplete_with_saved_progress'
            commit()
        except BaseException as error:
            status.update(state='stopped_with_saved_progress' if isinstance(error, (InterruptedError, TimeoutError)) else 'failed_no_retry',
                error=dict(type=type(error).__name__, message=str(error)))
            for result in status['groups'].values():
                if result['state'] == 'pending':
                    result.update(state='not_started', reason='earlier_phase_stopped_or_failed')
            commit()
            raise
        print(status['state'], flush=True)


if __name__ == '__main__':
    main()
