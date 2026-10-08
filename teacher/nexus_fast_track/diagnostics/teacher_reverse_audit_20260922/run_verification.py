"""Bounded read-only replay of unmodified downloaded candidate code."""
import os, sys, json, hashlib, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'repro_outputs'
GPU = 'GPU-e4d3702b-901b-8690-efa2-4f2a7a81f277'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    assert not (OUT / 'fresh_runs.json').exists(), 'Do not overwrite completed verification'
    manifest = json.loads((OUT / 'INPUT_VERIFICATION.json').read_text())
    for r in manifest['assets']:
        assert sha(ROOT / 'teacher_assets' / r['path']) == r['sha256'], r['path']
    for name, value in manifest['code_sha256'].items():
        assert sha(ROOT / 'code' / name) == value, name
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    assert GPU not in processes, 'Assigned GPU occupied; do not preempt'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    commands = [
        ('final_ae_cpu', 'replay_three_ae.py', ['--phase', 'final']),
        ('cascade_gpu_seed1', 'replay.py', ['--stage', 'cascade', '--device', 'cuda:0', '--point-seed', '34567', '--topology-seed', '12345']),
        ('cascade_gpu_seed2', 'replay.py', ['--stage', 'cascade', '--device', 'cuda:0', '--point-seed', '98765', '--topology-seed', '23456']),
    ]
    records = []
    for name, script, args in commands:
        cmd = [sys.executable, '-u', str(ROOT / 'code' / script), '--teacher-root', str(ROOT / 'teacher_assets'), '--out', str(OUT / name), *args]
        record = dict(name=name, command=cmd, started=time.time(), status='running')
        records.append(record)
        (OUT / 'fresh_runs.json').write_text(json.dumps(records, indent=2))
        print('START', name, flush=True)
        with (OUT / (name + '.log')).open('w') as log:
            try:
                result = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=600)
                record.update(returncode=result.returncode, status='completed' if result.returncode == 0 else 'failed')
            except subprocess.TimeoutExpired:
                record.update(status='timeout')
        record['seconds'] = time.time() - record['started']
        (OUT / 'fresh_runs.json').write_text(json.dumps(records, indent=2))
        print('END', name, record['status'], round(record['seconds'], 1), flush=True)
        if record['status'] != 'completed':
            return 1
    for r in manifest['assets']:
        assert sha(ROOT / 'teacher_assets' / r['path']) == r['sha256'], r['path']
    (OUT / 'READ_ONLY_VERIFIED.json').write_text(json.dumps(dict(optimizer_updates=0, asset_hashes_unchanged=True, code_hashes_verified=True, gpu_uuid=GPU, runs=3), indent=2))
    return 0

if __name__ == '__main__':
    sys.exit(main())
