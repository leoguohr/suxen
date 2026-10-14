"""Two endpoint evaluations, then five independent original A24000 continuations."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

import eval_continue as worker

ROOT = Path(__file__).resolve().parent
GPU_UUIDS = worker.GPU_UUIDS


def original_lock(path):
    """Hold every older launch inode without overwriting its evidence."""
    stream = path.open('r')
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        stream.close()
        raise RuntimeError(f'previous sweep launch lock held: {path}')
    return stream


def assert_uuid_mapping(gpus):
    assert tuple(gpus) == GPU_UUIDS
    output = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, check=True).stdout
    mapping = {int(line.split(',')[0]): line.split(',')[1].strip() for line in output.splitlines()}
    assert tuple(mapping[index] for index in (0, 1)) == GPU_UUIDS


def branch_command(args, branch, original):
    command = [sys.executable, str(ROOT / 'eval_continue.py'),
               'evaluate-endpoint' if branch in worker.TERMINAL_SHA else 'train',
               '--branch', branch, '--output', str(args.output / branch),
               '--start', str(args.start), '--start-sha', original.SOURCE_SHA, '--microgroup', '2']
    for name in ('base_root', 'original_runs', 'first_code', 'previous_code', 'deps', 'data'):
        command += ['--' + name.replace('_', '-'), str(getattr(args, name))]
    return command


def branch_complete(folder, exit_code, original):
    if not original.branch_complete(folder, exit_code):
        return False
    status = json.loads((folder / 'status.json').read_text())
    summary_path = folder / 'evaluation/summary.json'
    if not summary_path.is_file():
        return False
    summary = json.loads(summary_path.read_text())
    protocol = (summary.get('scored_trees') == summary.get('expected_trees') == 100
                and summary.get('steps') == 20 and summary.get('threshold') == .5
                and summary.get('gt_parents_used') is False
                and summary.get('sampler') == 'source Euler left endpoint'
                and summary.get('paired_control_seeds') == [97029000, 97029001]
                and summary.get('checkpoint_identity') == status['checkpoint'])
    if folder.name in worker.TERMINAL_SHA:
        return protocol and status.get('training_updates_this_attempt') == 0 and (
            status['checkpoint'].get('sha256') == worker.TERMINAL_SHA[folder.name])
    return protocol


def main(args):
    assert_uuid_mapping(args.gpus)
    train, environment = worker.bootstrap(args)
    import run_sweep_queue as original
    assert args.microgroup == 2 and args.start_sha == original.SOURCE_SHA
    locks, created = [], False
    try:
        for path in (args.base_root / '.sweep_launch.lock',
                     args.first_code / '.sweep_resume_launch.lock',
                     args.previous_code / '.sweep_resume_launch.lock'):
            locks.append(original_lock(path))
        locks.append(original.acquire_lock(ROOT / '.sweep_eval_continue_launch.lock'))
        for gpu in args.gpus:
            locks.append(original.acquire_lock(Path(tempfile.gettempdir()) / f'nexus_sweep_{gpu}.lock'))
            original.assert_gpu_idle(gpu)
        assert original.sha(args.start) == original.SOURCE_SHA
        sources = None
        for branch in worker.TERMINAL_SHA:
            _, _, config = worker.terminal_headers(args, branch)
            checked = worker.code_sources(args, config, train)
            assert sources is None or checked == sources
            sources = checked
        for branch in original.BRANCHES[2:]:
            assert not (args.original_runs / branch).exists(), f'{branch} is no longer unstarted'
        assert not args.output.exists(), 'new continuation output must be fresh'
        args.output.mkdir(parents=True)
        created = True
        audit = args.output / 'audit'
        audit.mkdir()
        for path in args.original_runs.iterdir():
            if path.is_file() and path.suffix in worker.EVIDENCE_SUFFIXES:
                target = audit / 'previous_queue' / path.name
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(path, target)
        if (args.original_runs / 'audit').is_dir():
            worker.archive_evidence(args.original_runs / 'audit', audit / 'previous_queue/audit')
        for name, item in sources.items():
            destination = audit / 'effective_code' / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item['path'], destination)
            assert worker.sha(destination) == item['sha256']
        for path in ROOT.glob('test_*.py'):
            shutil.copy2(path, audit / 'effective_code/actual_runtime_wrapper' / path.name)
        original.write_json(audit / 'code_identity.json', {'sources': sources, 'environment': environment})
        original.write_json(args.output / 'queue_config.json', {
            'start': str(args.start), 'start_sha256': original.SOURCE_SHA, 'data': str(args.data),
            'gpus': args.gpus, 'branches': original.BRANCHES, 'microgroups': {b: 2 for b in original.BRANCHES},
            'updates_per_branch': 10000, 'global_micros': 8, 'automatic_retry': False,
            'training_updates_this_attempt': {b: 0 if b in worker.TERMINAL_SHA else 10000 for b in original.BRANCHES},
            'original_runs': str(args.original_runs), 'endpoint_checkpoint_sha256': worker.TERMINAL_SHA,
            'base_root': str(args.base_root), 'first_code_root': str(args.first_code),
            'previous_code_root': str(args.previous_code), 'code_sources': sources, 'environment': environment})
        pending, running, results, events = list(original.BRANCHES), {}, {}, []
        stop_requested, fatal = False, None

        def stop(_number, _frame):
            nonlocal stop_requested
            stop_requested = True

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        while pending or running:
            for gpu, (branch, process) in list(running.items()):
                code = process.poll()
                if code is not None:
                    results[branch] = code if branch_complete(args.output / branch, code, original) else (code or 1)
                    del running[gpu]
            failed = fatal is not None or stop_requested or any(results.values())
            for gpu in args.gpus:
                if gpu in running or not pending or failed:
                    continue
                try:
                    original.assert_gpu_idle(gpu)
                    branch = pending[0]
                    command = branch_command(args, branch, original)
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=gpu, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',
                               PYTHONDONTWRITEBYTECODE='1')
                    with (audit / f'{branch}.log').open('x') as log:
                        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                                   start_new_session=True)
                    pending.pop(0)
                    running[gpu] = branch, process
                    events.append({'branch': branch, 'pid': process.pid, 'gpu': gpu, 'command': command,
                                   'started_at': datetime.now().astimezone().isoformat(),
                                   'mode': 'frozen_endpoint_evaluation_only' if branch in worker.TERMINAL_SHA else 'fresh_A24000'})
                    original.write_json(audit / 'commands.json', events)
                except Exception as error:
                    fatal, failed = {'error': repr(error), 'traceback': traceback.format_exc()}, True
            original.write_json(args.output / 'queue_status.json', {
                'state': 'failed_no_automatic_retry' if failed else ('running' if running or pending else 'aggregating'),
                'running': {gpu: {'branch': branch, 'pid': process.pid} for gpu, (branch, process) in running.items()},
                'pending': pending, 'exit_codes': results, 'finished': [b for b, code in results.items() if code == 0],
                'failed': [b for b, code in results.items() if code != 0], 'automatic_retry': False,
                'stop_requested': stop_requested, 'launch_failure': fatal, 'running_branches_allowed_to_finish': True})
            if failed and not running:
                raise RuntimeError('Continuation failed or stopped; pending branches not launched')
            if running or pending:
                time.sleep(5)
        assert set(results) == set(original.BRANCHES) and not any(results.values())
        original.aggregate(args.output)
        status = json.loads((args.output / 'queue_status.json').read_text())
        original.write_json(args.output / 'queue_status.json', {**status,
                            'state': 'all_training_and_evaluation_complete_packaging', 'evidence_audit_passed': True})
        original.package(args.output)
        status = json.loads((args.output / 'queue_status.json').read_text())
        original.write_json(args.output / 'queue_status.json', {**status, 'state': 'complete', 'evidence_audit_passed': True})
    except Exception as error:
        if created:
            path = args.output / 'queue_status.json'
            status = json.loads(path.read_text()) if path.exists() else {}
            original.write_json(path, {**status, 'state': 'failed_no_automatic_retry'})
            original.write_json(args.output / 'queue_failure.json', {
                'error': repr(error), 'traceback': traceback.format_exc(), 'automatic_retry': False})
        raise
    finally:
        for lock in reversed(locks):
            lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('base-root', 'original-runs', 'first-code', 'previous-code', 'deps', 'output', 'start', 'data'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--start-sha', default='2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba')
    parser.add_argument('--gpus', nargs=2, default=GPU_UUIDS)
    parser.add_argument('--microgroup', type=int, choices=(2,), default=2)
    main(parser.parse_args())
