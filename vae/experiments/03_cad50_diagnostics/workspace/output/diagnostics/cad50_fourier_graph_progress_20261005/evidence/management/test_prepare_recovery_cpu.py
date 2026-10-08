"""CPU-only prefix/no-overwrite and independent recovery preparation tests."""
import argparse
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_recovery as recovery
from protocol import LIMIT, MILESTONES, PARENTS, START, info, read, write

MANAGEMENT = Path(__file__).resolve().parent
CODE = {'test': 'CPU-only fixture'}


def log_bytes(last):
    return b''.join((json.dumps(dict(update_before=step-1, update_after=step), indent=None)+'\n').encode()
                    for step in range(START+1, last+1))


class RecoveryTests(unittest.TestCase):
    def test_exact_toy_prefix_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=MANAGEMENT) as directory:
            source, target = Path(directory)/'original.jsonl', Path(directory)/'prefix.jsonl'
            original = log_bytes(2004)
            source.write_bytes(original)
            summary = recovery.copy_log_prefix(source, target, 2002)
            self.assertEqual(target.read_bytes(), log_bytes(2002))
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(summary, dict(old_log_count=4, old_last_update=2004,
                                          restored_update=2002, lost_updates=2))
            with self.assertRaises(FileExistsError):
                recovery.copy_log_prefix(source, target, 2003)
            self.assertEqual(target.read_bytes(), log_bytes(2002))

    def test_invalid_tail_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory(dir=MANAGEMENT) as directory:
            source, target = Path(directory)/'original.jsonl', Path(directory)/'prefix.jsonl'
            original = log_bytes(2002)+b'{"update_before":2003,"update_after":2004}\n'
            source.write_bytes(original)
            with self.assertRaisesRegex(RuntimeError, 'Noncontiguous'):
                recovery.copy_log_prefix(source, target, 2002)
            self.assertEqual(source.read_bytes(), original)

    def prepare_original(self, root):
        write(root/'runs/controller.lock', {})
        for arm, last in recovery.OLD_LAST.items():
            folder = root/'runs'/arm
            write(folder/'execution.lock', {})
            parent_sha = PARENTS[arm]['sha256']
            write(folder/'config.json', dict(arm=arm, source_code=CODE, max_updates=LIMIT,
                start_updates=START, checkpoints=list(MILESTONES), parent=dict(sha256=parent_sha)))
            write(folder/'startup-baseline-verification.json', dict(passed=True, source_code=CODE, parent_sha256=parent_sha))
            write(folder/'status.json', dict(completed_updates=last))
            (folder/'updates.jsonl').write_bytes(log_bytes(last))
            cp = folder/'checkpoint-09000.pt'
            cp.write_bytes(b'CPU fixture, not a Torch checkpoint')
            entry = dict(info(cp), resumable=True, completed_updates=9000)
            write(cp.with_suffix('.json'), entry)
            os.link(cp, folder/'best.pt')
            evaluation = folder/'evaluations/step-09000/evaluation.json'
            write(evaluation, dict(complete=True, meshes=[{}]*50, checkpoint=entry))
            write(folder/'best.json', dict(score=[0, .1, .2], completed_updates=9000,
                checkpoint=info(folder/'best.pt'), evaluation_path=str(evaluation), evaluation_checkpoint=entry))

    def test_prepare_preserves_originals_budget_disclosure_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir=MANAGEMENT) as directory:
            root = Path(directory).resolve()
            self.prepare_original(root)
            before = {str(p): p.read_bytes() for p in (root/'runs').rglob('*') if p.is_file()}
            args = argparse.Namespace(outdir=str(root/'recovery'))
            with patch.object(recovery, 'ROOT', root), patch.object(recovery, 'verify_code', return_value=CODE):
                manifest = recovery.main(args)
                self.assertEqual(manifest['authorization'], 'pending')
                self.assertFalse(manifest['worker_launch_performed'])
                for arm, lost, actual in [('XYZ_LN_post', 989, 8989), ('XYZ_LN_pre', 977, 8977)]:
                    branch = manifest['branches'][arm]
                    self.assertEqual(branch['lost_updates'], lost)
                    self.assertEqual(branch['restored_update'], 9000)
                    self.assertEqual(branch['proposed_updates'], 1000)
                    self.assertEqual(branch['authorized_additional_updates'], 8000)
                    self.assertEqual(branch['original_actual_additional_updates_if_finished'], actual)
                    old, new = root/'runs'/arm, root/'recovery'/arm
                    for name in ['config.json', 'startup-baseline-verification.json', 'checkpoint-09000.json', 'best.json']:
                        self.assertEqual((new/name).read_bytes(), (old/name).read_bytes())
                    for name in ['checkpoint-09000.pt', 'best.pt']:
                        self.assertEqual((new/name).stat().st_ino, (old/name).stat().st_ino)
                    self.assertEqual((new/'updates.jsonl').read_bytes(), log_bytes(9000))
                    self.assertEqual(read(new/'best.json')['checkpoint']['path'], str(old/'best.pt'))
                self.assertEqual(before, {str(p): p.read_bytes() for p in (root/'runs').rglob('*') if p.is_file()})
                marker = (root/'recovery/RECOVERY.json').read_bytes()
                with self.assertRaisesRegex(RuntimeError, 'no overwrite'):
                    recovery.main(args)
                self.assertEqual((root/'recovery/RECOVERY.json').read_bytes(), marker)
                self.assertEqual(before, {str(p): p.read_bytes() for p in (root/'runs').rglob('*') if p.is_file()})


if __name__ == '__main__':
    unittest.main(verbosity=2)
