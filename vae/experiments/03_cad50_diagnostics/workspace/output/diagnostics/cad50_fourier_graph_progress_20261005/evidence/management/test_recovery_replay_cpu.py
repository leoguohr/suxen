"""Synthetic CPU-only replay ledger checks; no worker or GPU is launched."""
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_recovery_dual_gpu as supervisor

MANAGEMENT = Path(__file__).resolve().parent


class CPULedger(io.StringIO):
    def fileno(self):
        return -1  # fsync is mocked; this toy ledger never touches a real descriptor.


def row(step):
    return dict(update_before=step-1, update_after=step, uids=['a', 'b'],
        meshes=[dict(uid='a'), dict(uid='b')], mesh_loss_fp32_bits_before=['01020304', '05060708'],
        adam_step_after=step, seconds=1.)


def line(value):
    return (json.dumps(value)+'\n').encode()


class ReplayTests(unittest.TestCase):
    def test_timing_ignored_partial_append_and_exact_overlap(self):
        with tempfile.TemporaryDirectory(dir=MANAGEMENT) as directory, \
                patch.object(supervisor, 'RESTORED', 2000), patch.object(supervisor, 'LIMIT', 2003):
            original, replay = Path(directory)/'old.jsonl', Path(directory)/'new.jsonl'
            original.write_bytes(line(row(2001))+line(row(2002)))
            replay.write_bytes(b'')
            verifier = supervisor.ReplayVerifier('toy', original, replay, 2002)
            ledger = CPULedger()
            changed = row(2001)
            changed['seconds'] = 99.
            encoded = line(changed)
            with replay.open('ab') as stream:
                stream.write(encoded[:-1])
            verifier.poll(ledger)
            self.assertEqual(verifier.done, 2000)
            with replay.open('ab') as stream:
                stream.write(b'\n'+line(row(2002))+line(row(2003)))
            with patch.object(supervisor.os, 'fsync'):
                verifier.poll(ledger)
                result = verifier.finish(ledger)
            self.assertEqual(result['overlapping_updates_verified'], 2)
            self.assertEqual([json.loads(s)['state'] for s in ledger.getvalue().splitlines()], ['match', 'match', 'new_update'])

    def test_uid_loss_bits_and_adam_mismatch_are_rejected(self):
        for field in ('uid', 'loss_bits', 'adam_step'):
            with self.subTest(field=field), tempfile.TemporaryDirectory(dir=MANAGEMENT) as directory, \
                    patch.object(supervisor, 'RESTORED', 2000):
                original, replay = Path(directory)/'old.jsonl', Path(directory)/'new.jsonl'
                original.write_bytes(line(row(2001)))
                replay.write_bytes(b'')
                verifier = supervisor.ReplayVerifier('toy', original, replay, 2001)
                changed = row(2001)
                if field == 'uid':
                    changed['uids'].reverse()
                    changed['meshes'].reverse()
                elif field == 'loss_bits':
                    changed['mesh_loss_fp32_bits_before'][0] = 'ffffffff'
                else:
                    changed['adam_step_after'] = 2002
                replay.write_bytes(line(changed))
                ledger = CPULedger()
                with patch.object(supervisor.os, 'fsync'), self.assertRaises(RuntimeError):
                    verifier.poll(ledger)

    def test_budget_overrun_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=MANAGEMENT) as directory, \
                patch.object(supervisor, 'RESTORED', 2000), patch.object(supervisor, 'LIMIT', 2001):
            original, replay = Path(directory)/'old.jsonl', Path(directory)/'new.jsonl'
            original.write_bytes(line(row(2001)))
            replay.write_bytes(b'')
            verifier = supervisor.ReplayVerifier('toy', original, replay, 2001)
            replay.write_bytes(line(row(2001))+line(row(2002)))
            with patch.object(supervisor.os, 'fsync'), self.assertRaisesRegex(RuntimeError, 'budget'):
                verifier.poll(CPULedger())

    def test_locked_spawn_passes_fds_and_mismatch_stops_both_workers(self):
        import argparse
        import signal
        import test_prepare_recovery_cpu as fixtures
        from protocol import acquire, PARENTS
        with tempfile.TemporaryDirectory(dir=MANAGEMENT) as directory:
            root = Path(directory).resolve()
            fixtures.RecoveryTests().prepare_original(root)
            for arm in PARENTS:
                (root/'runs'/arm).mkdir(exist_ok=True)
                (root/'runs'/arm/'execution.lock').touch(exist_ok=True)
            for arm, last in supervisor.OLD_LAST.items():
                (root/'runs'/arm/'updates.jsonl').write_bytes(b''.join(line(row(s)) for s in range(2001, last+1)))
            with patch.object(fixtures.recovery, 'ROOT', root), patch.object(fixtures.recovery, 'verify_code', return_value=fixtures.CODE):
                fixtures.recovery.main(argparse.Namespace(outdir=str(root/'recovery')))
            children = []
            test = self

            class Child:
                def __init__(self, command, **kwargs):
                    options = dict(zip(command[4::2], command[5::2]))
                    arm = options['--arm']
                    test.assertEqual(options['--stop-at'], '10000')
                    test.assertEqual(options['--budget-updates'], '10000')
                    test.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], options['--gpu-uuid'])
                    test.assertEqual(len(kwargs['pass_fds']), 6)
                    test.assertIn(int(options['--lock-fd']), kwargs['pass_fds'])
                    for path in [root/'runs/controller.lock', root/'recovery'/arm/'execution.lock']+[root/'runs'/a/'execution.lock' for a in PARENTS]:
                        with test.assertRaises(RuntimeError):
                            acquire(path)
                    self.pid, self.code, self.signals = 100+len(children), None, []
                    children.append(self)
                    value = row(9001)
                    if arm == 'XYZ_LN_post':
                        value['mesh_loss_fp32_bits_before'][0] = 'ffffffff'
                    with (root/'recovery'/arm/'updates.jsonl').open('ab') as stream:
                        stream.write(line(value))

                def poll(self):
                    return self.code

                def send_signal(self, sig):
                    self.signals.append(sig)
                    self.code = -sig

                def wait(self):
                    return self.code

            args = argparse.Namespace(recovery_root=str(root/'recovery'), source='CPU fixture',
                gpu0_uuid='GPU-80f199d2-afab-fad3-824d-6d2482a4c882',
                gpu1_uuid='GPU-0634fd68-4a4d-facb-1b7f-0f9789679b47')
            with patch.object(supervisor, 'ROOT', root), patch.object(supervisor, 'verify_code', return_value=fixtures.CODE), \
                    patch.object(supervisor.subprocess, 'Popen', Child), self.assertRaisesRegex(RuntimeError, 'replay mismatch'):
                supervisor.main(args)
            self.assertEqual(len(children), 2)
            self.assertTrue(all(child.signals == [signal.SIGTERM] for child in children))
            self.assertTrue(list((root/'recovery').glob('authorization-*.json')))


if __name__ == '__main__':
    unittest.main(verbosity=2)
