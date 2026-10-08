"""Supervise two explicitly authorized XYZ replay workers and verify overlapping updates."""
import argparse, hashlib, json, os, re, signal, subprocess, sys, time
from contextlib import ExitStack
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from protocol import (BASE, LIMIT, PARENTS, ROOT, acquire, check_log, completed_run, info,
    read, require, verify_code, write)
from prepare_recovery import OLD_LAST, RESTORED
def comparison_bits(row):
    values = dict(uids=row['uids'], mesh_uids=[mesh['uid'] for mesh in row['meshes']],
        mesh_loss_fp32_bits_before=row['mesh_loss_fp32_bits_before'], adam_step_after=row['adam_step_after'])
    require(values['uids'] == values['mesh_uids'], 'UID list and per-mesh UID order disagree')
    require(len(values['uids']) == len(values['mesh_loss_fp32_bits_before']), 'UID/loss bit count differs')
    return json.dumps(values, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
class ReplayVerifier:
    def __init__(self, arm, original, replay, old_last):
        self.arm, self.path, self.old_last = arm, Path(replay), old_last
        check_log(original, old_last)
        check_log(replay, RESTORED)
        self.expected = {}
        with Path(original).open() as stream:
            for line in stream:
                row = json.loads(line)
                if row['update_after'] > RESTORED:
                    self.expected[row['update_after']] = comparison_bits(row)
        self.offset = self.path.stat().st_size
        self.done, self.matched = RESTORED, 0
    def poll(self, ledger):
        with self.path.open('rb') as stream:
            stream.seek(self.offset)
            while True:
                line = stream.readline()
                if not line.endswith(b'\n'):
                    break  # A worker can be in the middle of appending its current row.
                row = json.loads(line)
                step = row['update_after']
                require(step == self.done+1 and row['update_before'] == self.done and step <= LIMIT,
                        f'{self.arm}: replay log boundary/budget differs')
                bits = comparison_bits(row)
                expected = self.expected.get(step)
                match = row['adam_step_after'] == step and (expected is None or bits == expected)
                event = dict(arm=self.arm, update_after=step,
                    state='mismatch' if not match else ('new_update' if expected is None else 'match'),
                    actual_sha256=hashlib.sha256(bits).hexdigest(),
                    expected_sha256=hashlib.sha256(expected).hexdigest() if expected is not None else None,
                    fields=['uids', 'meshes.uid', 'mesh_loss_fp32_bits_before', 'adam_step_after'])
                ledger.write(json.dumps(event, separators=(',', ':'))+'\n')
                ledger.flush()
                os.fsync(ledger.fileno())
                require(match, f'{self.arm}: replay mismatch at update {step}; stop both workers')
                self.offset += len(line)
                self.done = step
                self.matched += expected is not None
    def finish(self, ledger):
        self.poll(ledger)
        require(self.offset == self.path.stat().st_size, f'{self.arm}: incomplete final log row')
        require(self.done == LIMIT and self.matched == self.old_last-RESTORED,
                f'{self.arm}: incomplete replay verification')
        check_log(self.path, LIMIT)
        return dict(completed_updates=self.done, overlapping_updates_verified=self.matched,
                    new_updates_after_old_tail=LIMIT-self.old_last, passed=True)
def main(args):
    for gpu in (args.gpu0_uuid, args.gpu1_uuid):
        require(re.fullmatch(r'GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}', gpu),
                'Explicit full GPU UUID required')
    require(args.gpu0_uuid.lower() != args.gpu1_uuid.lower(), 'Two different GPU UUIDs required')
    recovery = Path(args.recovery_root).resolve()
    runs = ROOT/'runs'
    require(ROOT in recovery.parents and recovery != runs and runs not in recovery.parents,
            'Independent recovery root must be inside ROOT, outside original runs')
    manifest = read(recovery/'RECOVERY.json')
    require(manifest['state'] == 'prepared' and manifest['authorization'] in ('pending', 'approved'),
            'Prepared recovery required; this launcher records the explicit human authorization separately')
    require(set(manifest['branches']) == set(OLD_LAST), 'Exactly the two XYZ recovery branches required')
    gpu_map = dict(zip(OLD_LAST, (args.gpu0_uuid, args.gpu1_uuid)))
    attempt = str(time.time_ns())
    active, stopped, verifiers = {}, [], {}
    with ExitStack() as stack:
        require(all((runs/arm/'execution.lock').is_file() for arm in PARENTS), 'Original branch lock missing')
        controller_lock = stack.enter_context(acquire(runs/'controller.lock'))
        old_locks = {arm: stack.enter_context(acquire(runs/arm/'execution.lock')) for arm in PARENTS}
        new_locks = {arm: stack.enter_context(acquire(recovery/arm/'execution.lock')) for arm in OLD_LAST}
        code = verify_code()
        require(manifest['source_code'] == code, 'Immutable scientific source differs from recovery manifest')
        for arm, branch in manifest['branches'].items():
            folder = recovery/arm
            require(Path(branch['recovery_directory']).resolve() == folder
                    and Path(branch['original_directory']).resolve() == (runs/arm).resolve(), 'Recovery branch path differs')
            require(branch['restored_update'] == RESTORED and branch['old_last_update'] == OLD_LAST[arm]
                    and branch['proposed_updates'] == LIMIT-RESTORED, 'Recovery update contract differs')
            require(not (folder/'complete.json').exists(), f'{arm}: recovery already complete; no relaunch')
            require(read(folder/'config.json')['source_code'] == code, f'{arm}: copied config source differs')
            info(runs/arm/'updates.jsonl', branch['original_files']['updates.jsonl']['sha256'])
            for name in ('config.json', 'startup-baseline-verification.json', 'best.json', 'checkpoint-09000.json'):
                info(folder/name, branch['original_files'][name]['sha256'])
            checkpoint = info(branch['resume']['path'], branch['resume']['sha256'])
            require(folder in Path(checkpoint['path']).parents, f'{arm}: resume outside recovery branch')
            info(folder/'best.pt', branch['original_files']['best.pt']['sha256'])
            info(folder/'updates.jsonl', branch['prefix_log']['sha256'])
            verifiers[arm] = ReplayVerifier(arm, runs/arm/'updates.jsonl', folder/'updates.jsonl', OLD_LAST[arm])
        ledger_path = recovery/f'replay-verification-{attempt}.jsonl'
        ledger = stack.enter_context(ledger_path.open('x'))
        def stop_workers(signum=signal.SIGTERM, frame=None):
            stopped.append(signum)
            for child in active.values():
                if child.poll() is None:
                    child.send_signal(signal.SIGTERM)
        previous = {sig: signal.signal(sig, stop_workers) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            write(recovery/f'authorization-{attempt}.json', dict(state='authorized', new_updates_each=1000,
                repeated_updates_allowed={arm: last-RESTORED for arm, last in OLD_LAST.items()},
                effective_finish=LIMIT, no_other_budget=True, authority='Explicit human approval in this chat'))
            write(recovery/f'supervisor-{attempt}.json', dict(source_code=code, management_source=info(__file__),
                gpu_map=gpu_map, resume={arm: b['resume'] for arm, b in manifest['branches'].items()},
                stop_at=LIMIT, authorization='approved', verification_ledger=str(ledger_path),
                failure_policy='SIGTERM both workers at safe boundaries; no retry'))
            for arm, gpu in gpu_map.items():
                require(not stopped, 'Supervisor stop requested before launch')
                branch = manifest['branches'][arm]
                command = [sys.executable, '-B', '-u', str(ROOT/'train_continue.py'), '--arm', arm,
                    '--source', args.source, '--parent-base', BASE, '--gpu-uuid', gpu,
                    '--outdir', str(recovery/arm), '--budget-updates', str(LIMIT), '--stop-at', str(LIMIT),
                    '--resume', branch['resume']['path'], '--resume-sha256', branch['resume']['sha256'],
                    '--lock-fd', str(new_locks[arm].fileno())]
                with (recovery/arm/f'worker-{attempt}.log').open('x') as log:
                    active[arm] = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, PYTHONDONTWRITEBYTECODE='1'),
                        pass_fds=tuple([controller_lock.fileno()]+[lock.fileno() for lock in old_locks.values()]+[new_locks[arm].fileno()]))
                write(recovery/f'active-{attempt}.json', dict(gpu_map=gpu_map, pids={a: p.pid for a, p in active.items()}))
            while True:
                for verifier in verifiers.values():
                    verifier.poll(ledger)
                require(not stopped, 'Supervisor stopped; workers preserve safe boundaries')
                codes = {arm: child.poll() for arm, child in active.items()}
                require(all(code in (None, 0) for code in codes.values()), f'Worker failed: {codes}')
                if all(code is not None for code in codes.values()):
                    break
                time.sleep(.5)
            verified = {arm: verifier.finish(ledger) for arm, verifier in verifiers.items()}
            require(all(completed_run(recovery/arm, LIMIT, arm) for arm in OLD_LAST), 'Both final checkpoint/evaluation receipts required')
            write(recovery/f'verification-complete-{attempt}.json', dict(passed=True, branches=verified,
                ledger=info(ledger_path), source_code=code, gpu_map=gpu_map, authorization='approved'))
        except BaseException as error:
            stop_workers()
            for child in active.values():
                child.wait()
            write(recovery/f'supervisor-failure-{attempt}.json', dict(error=str(error),
                returncodes={arm: child.poll() for arm, child in active.items()}, verification_ledger=str(ledger_path),
                observed={arm: dict(done=v.done, matched=v.matched) for arm, v in verifiers.items()}, no_automatic_retry=True))
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-root', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--gpu0-uuid', required=True, help='XYZ_LN_post GPU UUID')
    parser.add_argument('--gpu1-uuid', required=True, help='XYZ_LN_pre GPU UUID')
    main(parser.parse_args())
