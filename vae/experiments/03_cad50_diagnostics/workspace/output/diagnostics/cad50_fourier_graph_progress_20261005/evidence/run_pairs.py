"""Two processes per approved GPU; alternate pairs at every common milestone."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from protocol import (BASE, LIMIT, MILESTONES, PAIRS, PARENTS, ROOT, START, acquire, check_log, completed_run,
    info, launch_action, output_folder, read, require, schedule, verify_code, write)


def main(args):
    require(re.fullmatch(r'GPU-[0-9a-fA-F-]{36}', args.gpu_uuid), 'Explicit full GPU UUID required')
    require(args.budget_updates == LIMIT, 'Each branch has a locked 10000-update budget')
    outdir = output_folder(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    controller_lock = acquire(outdir/'controller.lock')
    code = verify_code()
    resume_map = read(args.resume_map) if args.resume_map else {}
    require(set(resume_map) <= set(PARENTS), 'Unknown resume-map branch')
    # Reserve ALL branches before starting any child. Locks are inherited, so a dead controller
    # cannot cause a duplicate launch while orphan children are still alive.
    locks = {arm: acquire(outdir/arm/'execution.lock') for arm in PARENTS}
    states = {}
    for arm in PARENTS:
        folder = outdir/arm
        resume = resume_map.get(arm)
        action = launch_action(folder, resume, LIMIT, arm)
        if action == 'skip_complete':
            states[arm] = dict(done=LIMIT, complete=True)
        elif action == 'resume':
            cfg = read(folder/'config.json')
            require(cfg['arm'] == arm and cfg['source_code'] == code and cfg['max_updates'] == LIMIT,
                    f'{arm}: resumed run contract differs')
            cp = info(resume['path'], resume['sha256'])
            require(folder.resolve() in Path(cp['path']).parents, f'{arm}: resume checkpoint outside its branch directory')
            sidecar = read(Path(cp['path']).with_suffix('.json'))
            require(sidecar['sha256'] == cp['sha256'], f'{arm}: checkpoint sidecar differs')
            require(sidecar['resumable'] is True and type(sidecar['completed_updates']) is int
                    and START <= sidecar['completed_updates'] <= LIMIT, 'Resume sidecar is not a valid continuation boundary')
            check_log(folder/'updates.jsonl', sidecar['completed_updates'])
            # No torch/CUDA in the controller. Worker validates the actual complete state.
            states[arm] = dict(done=sidecar['completed_updates'], complete=False, resume=cp)
        else:
            states[arm] = dict(done=START, complete=False, resume=None)
    attempt = str(time.time_ns())
    active = {}
    stopped = []

    def request_stop(signum, frame):
        stopped.append(signum)
        for child in active.values():
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    launch = dict(gpu_uuid=args.gpu_uuid, allocator_fraction_each=.44, max_updates_each=LIMIT,
        milestones=list(MILESTONES), pairs=[list(p) for p in PAIRS], source_code=code,
        resume_map=resume_map, failure_policy='stop active pair at safe boundaries; no automatic retry/reset')
    write(outdir/f'controller-{attempt}.json', launch)
    try:
        for milestone in schedule(LIMIT):
            for pair in PAIRS:
                require(not stopped, 'Controller stop requested')
                active = {}
                for arm in pair:
                    state = states[arm]
                    folder = outdir/arm
                    if state['complete']:
                        continue
                    receipt_path = folder/f'stage-{milestone:05d}.json'
                    if receipt_path.exists() and milestone != LIMIT:
                        receipt = read(receipt_path)
                        require(receipt['state'] == 'stage_complete' and receipt['completed_updates'] == milestone,
                                f'{arm}: invalid stage receipt')
                        info(receipt['checkpoint']['path'], receipt['checkpoint']['sha256'])
                        result = read(receipt['evaluation_path'])
                        require(result['complete'] and len(result['meshes']) == 50
                                and result['checkpoint']['sha256'] == receipt['checkpoint']['sha256'], 'Invalid stage evaluation')
                        # A user-supplied resume must not silently be advanced or rolled back by a marker.
                        if state['done'] is not None:
                            require(state['done'] >= milestone, 'Stage receipt ahead of selected state')
                        continue
                    command = [sys.executable, '-B', '-u', str(ROOT/'train_continue.py'),
                        '--arm', arm, '--source', args.source, '--parent-base', args.parent_base,
                        '--gpu-uuid', args.gpu_uuid, '--outdir', str(folder), '--budget-updates', str(LIMIT),
                        '--stop-at', str(milestone), '--lock-fd', str(locks[arm].fileno())]
                    if state['resume']:
                        command += ['--resume', state['resume']['path'], '--resume-sha256', state['resume']['sha256']]
                    log_path = folder/f'stage-{milestone:05d}-{attempt}.log'
                    with log_path.open('x') as log:
                        active[arm] = subprocess.Popen(command, cwd=ROOT,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu_uuid, PYTHONDONTWRITEBYTECODE='1'),
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                            pass_fds=(controller_lock.fileno(), locks[arm].fileno()))
                    write(outdir/f'active-{attempt}.json', dict(milestone=milestone, gpu_uuid=args.gpu_uuid,
                        pids={a: p.pid for a, p in active.items()}, logs={a: str(outdir/a/f'stage-{milestone:05d}-{attempt}.log') for a in active}))
                failed = False
                while any(child.poll() is None for child in active.values()):
                    if stopped or any(child.poll() not in (None, 0) for child in active.values()):
                        if not failed:
                            failed = True
                            for child in active.values():
                                if child.poll() is None:
                                    child.send_signal(signal.SIGTERM)
                    time.sleep(.5)
                codes = {arm: child.wait() for arm, child in active.items()}
                write(outdir/f'exit-{milestone:05d}-{pair[0]}-{attempt}.json', codes)
                require(not stopped and not failed and not any(codes.values()), 'Pair failed/stopped; inspect preserved logs/states, no retry')
                for arm in active:
                    folder = outdir/arm
                    receipt = read(folder/f'stage-{milestone:05d}.json')
                    require(receipt['completed_updates'] == milestone and receipt['state'] == 'stage_complete', 'Worker did not finish its stage')
                    checkpoint = info(receipt['checkpoint']['path'], receipt['checkpoint']['sha256'])
                    result = read(receipt['evaluation_path'])
                    require(result['complete'] and len(result['meshes']) == 50
                            and result['checkpoint']['sha256'] == checkpoint['sha256'], 'Worker stage evaluation invalid')
                    states[arm] = dict(done=milestone, resume=checkpoint,
                        complete=completed_run(folder, LIMIT, arm) if milestone == LIMIT else False)
                active = {}
            write(outdir/f'milestone-{milestone:05d}-{attempt}.json', dict(state='all_four_stage_complete',
                milestone=milestone, branches=states, budget_each=LIMIT))
        require(all(completed_run(outdir/arm, LIMIT, arm) for arm in PARENTS), 'Four complete final states required')
        write(outdir/'controller-complete.json', dict(state='complete', max_updates_each=LIMIT,
            additional_updates_each=8000, branches=states, stop_reason='fixed_budget_no_early_elimination'))
    except BaseException as error:
        # This includes launch/evidence errors. Already-running siblings finish a safe boundary.
        for child in active.values():
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        for child in active.values():
            child.wait()
        write(outdir/f'controller-failure-{attempt}.json', dict(error=str(error),
            returncodes={arm: child.poll() for arm, child in active.items()}, states=states))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--parent-base', default=BASE)
    parser.add_argument('--outdir', default=str(ROOT/'runs'))
    parser.add_argument('--budget-updates', type=int, default=LIMIT, choices=[LIMIT])
    parser.add_argument('--resume-map', help='Explicit JSON mapping incomplete branches to {path, sha256}')
    main(parser.parse_args())
