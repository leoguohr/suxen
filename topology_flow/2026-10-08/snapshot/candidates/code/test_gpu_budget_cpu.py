"""Mocked-clock GPU-ledger tests: no CUDA calls, devices, real budgets or network."""
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import gpu_budget
from gpu_budget import GPUBudget
from _faces_reference import atomic_json


class Clock:
    def __init__(self): self.now=100.
    def monotonic(self): return self.now
    def time(self): return 1_000_000.+self.now
    def advance(self,seconds): self.now+=seconds


def expect(kind,call):
    try: call()
    except kind: return
    raise AssertionError(f'Expected {kind.__name__}')


def ledger(path,**changes):
    state=dict(schema_version=1,authorized_gpu_uuid='GPU-synthetic-budget-test',
        max_gpu_seconds=30.,max_optimizer_updates=2)
    state.update(changes);atomic_json(path,state)


def enter_only(path,device='cuda:0'):
    with GPUBudget(path,phase='test_only_no_gpu',device=device): pass


def main():
    tests=[];clock=Clock()
    with tempfile.TemporaryDirectory(prefix='topology-budget-cpu-') as tmp, \
            patch.object(gpu_budget.time,'monotonic',clock.monotonic), \
            patch.object(gpu_budget.time,'time',clock.time), \
            patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'GPU-synthetic-budget-test'}):
        root=Path(tmp);path=root/'budget.json';ledger(path)
        with GPUBudget(None,phase='test',device='cpu') as budget:
            clock.advance(100)
            assert budget.snapshot()==dict(gpu_used=False,charged_gpu_seconds=0.)
            assert budget.remaining_seconds()==float('inf') and not budget.stop()
            assert not budget.updates_exhausted()
            budget.record_update(999);budget.require_training_cursor(999)
        expect(ValueError,lambda:enter_only(path,'cpu'))
        expect(ValueError,lambda:enter_only(None))
        tests.append('CPU mode performs no GPU accounting; CUDA requires an explicit ledger')

        original=path.read_text()
        for visible in ('0','GPU-other','GPU-synthetic-budget-test,GPU-other'):
            with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':visible}):
                expect(ValueError,lambda:enter_only(path))
            assert path.read_text()==original
        expect(ValueError,lambda:enter_only(path,'cuda:1'))
        assert path.read_text()==original
        tests.append('Reject numeric, different or multiple visible GPUs and nonzero CUDA mapping without modifying the ledger')

        with GPUBudget(path,phase='export',device='cuda:0') as first:
            clock.advance(12)
            assert first.remaining_seconds()==18
            expect(BlockingIOError,lambda:enter_only(path))
            assert json.loads(path.read_text())['active']['phase']=='export'
            expect(ValueError,lambda:first.stop(-1))
        saved=json.loads(path.read_text())
        assert saved['charged_gpu_seconds']==12 and saved['active'] is None
        assert saved['attempts'][0]['seconds']==12
        clock.advance(200) # Idle wall time between GPU phases is not GPU reservation time.
        with GPUBudget(path,phase='train',device='cuda:0') as second:
            assert second.remaining_seconds()==18
            second.require_training_cursor(0)
            expect(ValueError,lambda:second.require_training_cursor(1))
            expect(ValueError,lambda:second.record_update(2))
            second.record_update(1)
            assert json.loads(path.read_text())['last_completed_updates']==1
            expect(ValueError,lambda:second.record_update(1))
            clock.advance(5);second.record_update(2)
            assert second.updates_exhausted()
            expect(RuntimeError,lambda:second.record_update(3))
            second.require_training_cursor(2)
            assert second.snapshot()['effective_optimizer_updates']==2
        saved=json.loads(path.read_text())
        assert saved['charged_gpu_seconds']==17 and len(saved['attempts'])==2
        assert saved['effective_optimizer_updates']==saved['last_completed_updates']==2
        tests.append('Exclusive ledger lock; export and train charge cumulative reservation time; idle gaps excluded; cursor/update writes durable')
        tests.append('Reject replayed/skipped cursors; exact update cap stops further optimizer accounting')

        with GPUBudget(path,phase='flow_evaluation',device='cuda:0') as third:
            assert third.updates_exhausted() and not third.stop()
            assert third.stop(reserve_seconds=13)
            clock.advance(13)
            assert third.stop() and third.remaining_seconds()==0
        saved=path.read_text()
        expect(RuntimeError,lambda:enter_only(path))
        assert path.read_text()==saved
        assert json.loads(saved)['charged_gpu_seconds']==30
        tests.append('Evaluation can consume remaining time after the update cap; reserve/equality time boundary stops; exhausted reentry refused')

        crash=root/'crash.json';ledger(crash)
        def failed_attempt():
            with GPUBudget(crash,phase='train',device='cuda:0'):
                clock.advance(4)
                raise RuntimeError('synthetic failure')
        expect(RuntimeError,failed_attempt)
        saved=json.loads(crash.read_text())
        assert saved['charged_gpu_seconds']==4 and saved['active'] is None
        assert saved['attempts'][0]['outcome']=='RuntimeError'
        with GPUBudget(crash,phase='vae_baseline',device='cuda:0') as next_phase:
            assert next_phase.remaining_seconds()==26
        unclosed=root/'unclosed.json'
        ledger(unclosed,charged_gpu_seconds=7.,active={'phase':'interrupted_test'})
        original=unclosed.read_text()
        expect(RuntimeError,lambda:enter_only(unclosed))
        assert unclosed.read_text()==original
        tests.append('Exceptions are charged and release the lock; unclosed attempts refuse restart without silently resetting time')

        overrun=root/'overrun.json';ledger(overrun)
        with GPUBudget(overrun,phase='test_atomic_boundary',device='cuda:0') as last:
            clock.advance(31)
            assert last.stop()
        saved=json.loads(overrun.read_text())
        assert saved['charged_gpu_seconds']==31 and saved['attempts'][0]['exceeded_time_budget']
        tests.append('Boundary overrun remains charged and explicitly flagged, never clamped out of recorded usage')
    evidence=dict(passed=True,device='cpu',tests=tests,clock='mocked',cuda_device_calls=0,
        real_gpu_used=False,real_budget_files_modified=False,temporary_fixtures_removed=True)
    atomic_json(Path(__file__).resolve().parent.parent/'evidence/gpu_budget_cpu_tests.json',evidence)
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
