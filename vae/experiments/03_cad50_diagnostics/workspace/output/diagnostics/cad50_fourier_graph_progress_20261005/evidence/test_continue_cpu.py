"""CPU protocol/controller tests; exact production sampling isolated without Torch when absent."""
import argparse
import ast
from contextlib import ExitStack
import importlib.util
import itertools
import math
import random
import tempfile
import types
import unittest
from unittest.mock import patch
from pathlib import Path
import hashlib
import json
import os
from protocol import (LIMIT, MILESTONES, PAIRS, PARENTS, ROOT, acquire, check_cursor, check_log,
    info, launch_action, read, require, schedule, write)
import run_pairs

HAS_NUMPY = importlib.util.find_spec('numpy') is not None
HAS_TORCH = importlib.util.find_spec('torch') is not None
if HAS_NUMPY:
    import numpy as np
    # Execute the UNMODIFIED production stateless sampling functions. Only their final
    # torch.from_numpy wrapper is replaced by identity for this no-Torch CPU test.
    tree = ast.parse((ROOT/'data_objective.py').read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in ('array_sha', 'seed_for', 'epoch_batches', 'negative_faces')]
    ns = dict(np=np, hashlib=hashlib, math=math, itertools=itertools,
              torch=types.SimpleNamespace(from_numpy=lambda array: array))
    exec(compile(ast.Module(body=selected, type_ignores=[]), 'production_sampling', 'exec'), ns)
    batches = ns['epoch_batches']
    negatives = ns['negative_faces']

    class CPUFaces:
        def __init__(self, rows):
            self.rows = rows

        def cpu(self):
            return self

        def tolist(self):
            return self.rows

UIDS = [f'teacher_cad50_{i:02d}' for i in range(50)]


def synthetic_batches(uids, epoch):
    return [uids[i:i+5] for i in range(0, 50, 5)]


class ProtocolTests(unittest.TestCase):
    def test_locked_budget_and_milestones(self):
        self.assertEqual(schedule(10000), [3000, 4000, 5000, 7500, 9000, 10000])
        self.assertEqual(10000-2000, 8000)
        self.assertEqual(sum(b-a for a, b in zip(MILESTONES, MILESTONES[1:])), 8000)
        for invalid in (2000, 5000, 10001, 11000):
            with self.assertRaises(RuntimeError):
                schedule(invalid)

    def test_exact_partial_cursor_and_final_participation(self):
        for done in (2000, 2001, 2009, 2010, 5007, 7500, 9999, 10000):
            epoch, cursor = divmod(done, 10)
            count = {u: epoch+(i < 5*cursor) for i, u in enumerate(UIDS)}
            cp = dict(completed_updates=done, next_epoch=epoch, next_batch=cursor,
                      negative_seed=0, participation=count)
            check_cursor(cp, UIDS, synthetic_batches, LIMIT)
            cp['participation'][UIDS[-1]] += 1
            with self.assertRaises(RuntimeError):
                check_cursor(cp, UIDS, synthetic_batches, LIMIT)

    def test_logs_refuse_silent_replay_or_unlogged_updates(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory)/'updates.jsonl'
            check_log(path, 2000)
            path.write_text(''.join(json.dumps(dict(update_before=i-1, update_after=i))+'\n' for i in range(2001, 2004)))
            check_log(path, 2003)
            for checkpoint in (2000, 2002, 2004):
                with self.assertRaises(RuntimeError):
                    check_log(path, checkpoint)
            path.write_text(json.dumps(dict(update_before=2001, update_after=2002))+'\n')
            with self.assertRaises(RuntimeError):
                check_log(path, 2002)

    def test_explicit_resume_and_lock_prevent_duplicate_start(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            folder = Path(directory)
            self.assertEqual(launch_action(folder, None, LIMIT, 'XYZ_LN_pre'), 'start_from_parent')
            write(folder/'config.json', {})
            with self.assertRaises(RuntimeError):
                launch_action(folder, None, LIMIT, 'XYZ_LN_pre')
            self.assertEqual(launch_action(folder, dict(path='state.pt', sha256='hash'), LIMIT, 'XYZ_LN_pre'), 'resume')
            lock = acquire(folder/'execution.lock')
            with self.assertRaises(RuntimeError):
                acquire(folder/'execution.lock')
            lock.close()


@unittest.skipUnless(HAS_NUMPY, 'NumPy unavailable: actual PCG64 sampling not executed')
class SamplingTests(unittest.TestCase):
    def test_every_continuation_epoch_covers_all50_once(self):
        for epoch in range(200, 1000):
            groups = batches(UIDS, epoch)
            self.assertEqual([len(g) for g in groups], [5]*10)
            self.assertEqual(sorted(sum(groups, [])), UIDS)
        self.assertEqual(batches(UIDS, 200)[0], [f'teacher_cad50_{i}' for i in ('31', '28', '38', '37', '00')])

    def test_resume_next_uid_and_negative_bytes_equal(self):
        item = dict(uid=UIDS[31], vertices=np.zeros((12, 3), dtype=np.float32),
                    gt_faces=CPUFaces([list(row) for row in list(itertools.combinations(range(12), 3))[:12]]))
        for done in (2000, 2001, 2009, 2010, 4999, 5000, 7500, 9999):
            epoch, cursor = divmod(done, 10)
            original_batch = batches(UIDS, epoch)[cursor]
            first, first_hash = negatives(item, epoch)
            random.seed(done)
            np.random.seed(done)
            np.random.random(100)
            restored = dict(next_epoch=epoch, next_batch=cursor)
            second, second_hash = negatives(item, restored['next_epoch'])
            self.assertEqual(original_batch, batches(UIDS, restored['next_epoch'])[restored['next_batch']])
            self.assertEqual(first_hash, second_hash)
            self.assertTrue(np.array_equal(first, second))
        self.assertNotEqual(negatives(item, 200)[1], negatives(item, 201)[1])
        self.assertEqual(len(negatives(item, 200)[0]), 18)

    def test_real_order_partial_participation(self):
        for done in (2000, 2001, 7507, 9999, 10000):
            epoch, cursor = divmod(done, 10)
            count = dict.fromkeys(UIDS, epoch)
            for uid in sum(batches(UIDS, epoch)[:cursor], []):
                count[uid] += 1
            check_cursor(dict(completed_updates=done, next_epoch=epoch, next_batch=cursor,
                negative_seed=0, participation=count), UIDS, batches, LIMIT)


class ControllerTests(unittest.TestCase):
    def fake_worker(self, launches, fail=False):
        class Child:
            def __init__(self, command, **kwargs):
                values = dict(zip(command[4::2], command[5::2]))
                # command is [python, -B, -u, script, --key, value, ...]
                arm = values['--arm']
                step = int(values['--stop-at'])
                folder = Path(values['--outdir'])
                launches.append((step, arm, values.get('--resume'), values['--gpu-uuid']))
                self.pid = len(launches)+50000
                self.code = 1 if fail and len(launches) == 1 else 0
                if fail:
                    return
                cfg = dict(arm=arm, max_updates=LIMIT, source_code={'test': 'CPU simulation'})
                write(folder/'config.json', cfg)
                checkpoint = folder/f'checkpoint-{step:05d}.pt'
                checkpoint.write_bytes(f'CPU fake checkpoint {arm} {step}'.encode())
                entry = info(checkpoint)
                entry.update(completed_updates=step, resumable=True)
                write(checkpoint.with_suffix('.json'), entry)
                evaluation_path = folder/f'evaluations/step-{step:05d}/evaluation.json'
                write(evaluation_path, dict(complete=True, meshes=[dict(uid=u) for u in UIDS], checkpoint=entry))
                write(folder/f'stage-{step:05d}.json', dict(state='stage_complete', completed_updates=step,
                    checkpoint=entry, evaluation_path=str(evaluation_path), budget=LIMIT))
                # Fake only controller I/O; this is not model/checkpoint/GPU verification.
                if step == LIMIT:
                    write(folder/'complete.json', dict(state='complete', completed_updates=step,
                        next_epoch=1000, next_batch=0, final_checkpoint=entry, final_evaluation_path=str(evaluation_path)))

            def poll(self):
                return self.code

            def wait(self):
                return self.code

            def send_signal(self, signum):
                self.code = -signum
        return Child

    def args(self, folder):
        return argparse.Namespace(source='CPU-only-unused', parent_base='CPU-only-unused',
            gpu_uuid='GPU-80f199d2-afab-fad3-824d-6d2482a4c882', budget_updates=LIMIT,
            resume_map=None, outdir=str(folder))

    def test_all_four_rotate_at_common_milestones_and_completed_skip(self):
        launches = []
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, ExitStack() as stack:
            folder = Path(directory)/'runs'
            stack.enter_context(patch.object(run_pairs, 'verify_code', return_value={'test': 'CPU simulation'}))
            stack.enter_context(patch.object(run_pairs.subprocess, 'Popen', self.fake_worker(launches)))
            run_pairs.main(self.args(folder))
            self.assertEqual([(s, a) for s, a, _, _ in launches], [(s, a) for s in MILESTONES[1:] for pair in PAIRS for a in pair])
            self.assertTrue(all(resume is None for _, _, resume, _ in launches[:4]))
            self.assertTrue(all(resume is not None for _, _, resume, _ in launches[4:]))
            self.assertTrue(all(gpu == self.args(folder).gpu_uuid for _, _, _, gpu in launches))
            run_pairs.main(self.args(folder))
            self.assertEqual(len(launches), 24)  # No duplicate launch of completed arms.

    def test_pair_failure_stops_before_other_pair_or_later_stage(self):
        launches = []
        with tempfile.TemporaryDirectory(dir=ROOT) as directory, ExitStack() as stack:
            stack.enter_context(patch.object(run_pairs, 'verify_code', return_value={'test': 'CPU simulation'}))
            stack.enter_context(patch.object(run_pairs.subprocess, 'Popen', self.fake_worker(launches, fail=True)))
            with self.assertRaises(RuntimeError):
                run_pairs.main(self.args(Path(directory)/'runs'))
            self.assertEqual([arm for _, arm, _, _ in launches], list(PAIRS[0]))


@unittest.skipUnless(HAS_TORCH and HAS_NUMPY, 'Torch unavailable: model/Adam/RNG CPU audit tests not executed')
class TorchAuditTests(unittest.TestCase):
    def test_adam_mapping_steps_and_rng_raw_bytes(self):
        import torch
        from resume_audit import equal_state, validate_adam
        from protocol import ADAM
        names = [f'p{i}' for i in range(412)]
        model = {n: torch.zeros(1) for n in names}
        state = dict(param_groups=[dict(ADAM, params=list(range(412)))], state={i:
            dict(step=torch.tensor(2000.), exp_avg=torch.zeros(1), exp_avg_sq=torch.zeros(1)) for i in range(412)})
        validate_adam(state, names, model, 2000)
        state['state'][411]['step'] += 1
        with self.assertRaises(RuntimeError):
            validate_adam(state, names, model, 2000)
        with self.assertRaises(RuntimeError):
            equal_state(torch.tensor([0.]), torch.tensor([-0.]), 'raw signed zero')


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    suite = unittest.defaultTestLoader.loadTestsFromModule(__import__(__name__))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write(ROOT/'LOCAL_CPU_TESTS.json', dict(passed=result.wasSuccessful(), tests_run=result.testsRun,
        skipped=[dict(test=str(test), reason=reason) for test, reason in result.skipped],
        numpy_available=HAS_NUMPY, torch_available=HAS_TORCH,
        scope='CPU budget/cursor/log/lock/controller simulation; real PCG64 production sampling with identity tensor wrapper',
        GPU_execution_verified=False, actual_full_checkpoint_restore_verified=False))
    raise SystemExit(0 if result.wasSuccessful() else 1)
