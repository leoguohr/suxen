"""CPU-only completion coordinator; never starts/resumes training or changes code."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'repro_outputs'
VARIANTS = ['A_v1_recipe_control', 'B_v2_teacher_blocks']
GPUS = ['GPU-0ae7719a-74e5-08ae-75de-0a6205381365', 'GPU-7ea0dcc5-8893-8144-31c6-df3c9b25b351']
PYTHON = '/opt/conda/bin/python'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def terminal(variant):
    return any((ROOT / variant / name).exists() for name in ['training_complete.json', 'failure.json'])


def worker_active():
    with (ROOT / 'queue/worker.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def pending():
    return [p for p in sorted((ROOT / 'queue').glob('*.json'))
            if p.name != 'worker_complete.json' and not p.with_suffix('.done').exists()]


def maybe_fallback():
    receipt = OUT / 'evaluation_fallback_launch.json'
    if receipt.exists() or worker_active() or not pending():
        return
    for variant, gpu in zip(VARIANTS, GPUS):
        if not terminal(variant):
            continue
        processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
        if gpu in processes:
            continue
        before = subprocess.check_output(['nvidia-smi', '--id=' + gpu,
            '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        fields = [x.strip() for x in before.split(',')]
        if fields[0] != gpu or int(fields[1]) != 0 or int(fields[2]) != 0:
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python',
            CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
        with (OUT / 'evaluation_fallback.log').open('x') as log:
            process = subprocess.Popen([PYTHON, '-u', str(ROOT / 'eval_worker.py')], cwd=ROOT,
                env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write(receipt, dict(pid=process.pid, gpu_uuid=gpu, released_training_branch=variant,
            gpu_before=before, launched=time.time(), reason='Original third-card worker absent; reuse a released allocated GPU',
            optimizer_updates=0, training_source_changed=False))
        return


def summarize(incomplete_reason=None):
    audit = subprocess.run([PYTHON, str(ROOT / 'postprocess/check_paired_logs.py')],
        cwd=ROOT, capture_output=True, text=True)
    (OUT / 'FINAL_PAIRED_LOG_AUDIT.log').write_text(audit.stdout + audit.stderr)
    rows = {}
    for variant in VARIANTS:
        branch = ROOT / variant
        evaluations = [read(p) for p in sorted((branch / 'evaluations').glob('eval-*.json'))]
        for row in evaluations:
            assert row['complete'] and row['optimizer_updates'] == 0 and len(row['meshes']) == 50
            for task in ['edge', 'face']:
                for key in ['tp', 'fp', 'fn']:
                    assert row['counts'][task][key] == sum(m[task][key] for m in row['meshes'])
            assert row['joint_perfect'] == sum(m['joint_perfect'] for m in row['meshes'])
        rows[variant] = dict(evaluations=evaluations,
            completion=read(branch / 'training_complete.json') if (branch / 'training_complete.json').exists() else None,
            failures={p.name: read(p) for p in [branch / 'failure.json', branch / 'evaluation_failure.json'] if p.exists()})
    common = sorted(set(r['step'] for r in rows[VARIANTS[0]]['evaluations']) &
                    set(r['step'] for r in rows[VARIANTS[1]]['evaluations']))
    common_step = common[-1] if common else None
    lines = ['# CAD50 own512 architecture comparison', '',
        'No teacher or B2500 weights were used. A is the original backbone with the new recipe; B is the Fourier / changed Graph order / per-block FFN V2 with the same recipe.', '',
        f'Latest jointly evaluated budget: {common_step} optimizer updates, each involving five complete meshes.', '',
        '| Branch | Step | Actual Face F1 | Edge FP/FN | Large16 Face F1 | Strict /50 |',
        '|---|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        selected = next((r for r in rows[variant]['evaluations'] if r['step'] == common_step), None)
        if selected:
            e = selected['counts']['edge']
            lines.append(f"| {variant} | {common_step} | {selected['counts']['face']['micro_f1']:.9f} | {e['fp']}/{e['fn']} | {selected['large16']['counts']['face']['micro_f1']:.9f} | {selected['joint_perfect']} |")
    lines += ['', '## Stop status and scope', '', f'Paired recorded batch/negative/LR audit exit code: {audit.returncode}.']
    for variant in VARIANTS:
        row = rows[variant]
        lines.append(f"- {variant}: completion={json.dumps(row['completion'])}; failures={json.dumps(row['failures'])}")
        if row['evaluations']:
            best = max(row['evaluations'], key=lambda x: x['counts']['face']['micro_f1'])
            strict = max(row['evaluations'], key=lambda x: x['joint_perfect'])
            lines.append(f"  Highest observed Face F1={best['counts']['face']['micro_f1']:.9f} at step {best['step']}; highest strict coverage={strict['joint_perfect']}/50 at step {strict['step']}.")
    lines += ['', 'Face F1 >= 0.997 and strict 50/50 are separate outcomes. This compares whole architecture versions and cannot isolate Fourier, Graph order or FFN effects. Parameter counts differ (A 112,212,544; B 246,575,680).', '',
        'Third-card connectivity and worker were lost during training. Any later evaluation fallback uses only a naturally released, already allocated training GPU. Checkpoint evaluation may therefore be delayed; no training budget is added.', '',
        'This is an automatically generated evidence summary. Human scientific interpretation remains pending. Old fixed100 experiments and teacher diffusion/prior networks are outside this run. No follow-on training is launched.', '',
        '## Artifact layout', '', 'All experiment code, configs, logs, full evaluation metrics and prediction arrays are included in independent ZIP shards. Full model/Adam/RNG checkpoints remain on the server and are listed with sizes and save-time SHA256; final checkpoint hashes are independently rechecked. Linked source data remain at their original paths.']
    if incomplete_reason:
        lines += ['', 'INCOMPLETE: ' + incomplete_reason]
    (OUT / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    write(OUT / 'RESULTS.json', dict(branches=rows, latest_common_step=common_step,
        paired_log_audit_passed=audit.returncode == 0, incomplete_reason=incomplete_reason,
        optimizer_updates_added_by_postprocessing=0))
    manifest = []
    for variant in VARIANTS:
        branch = ROOT / variant
        completion = rows[variant]['completion']
        for p in sorted(branch.glob('checkpoint-*.json')):
            entry = read(p)
            model_path = Path(entry['path'])
            entry['identity'] = 'full native model + AdamW + RNG + progress/config'
            entry['hash_provenance'] = 'save-time SHA256 receipt'
            entry['present'] = model_path.exists()
            entry['size_matches'] = entry['present'] and model_path.stat().st_size == entry['bytes']
            if completion and entry['path'] == completion['final_checkpoint']['path']:
                entry['final_sha256_rechecked'] = sha(model_path)
                assert entry['final_sha256_rechecked'] == entry['sha256']
            manifest.append(entry)
    write(OUT / 'CHECKPOINT_MANIFEST.json', manifest)


def package():
    destination = ROOT / 'delivery'
    destination.mkdir(exist_ok=True)
    assert not list(destination.glob('*.zip')), 'Refuse overwriting a prior delivery'
    files = []
    for base in [ROOT, ROOT / 'reference', ROOT / 'frozen_code', ROOT / 'postprocess', OUT, ROOT / 'queue'] + [ROOT / v for v in VARIANTS]:
        iterator = base.iterdir() if base == ROOT else base.rglob('*')
        for path in iterator:
            if path.is_symlink() or not path.is_file() or path.suffix in ['.pt', '.pyc', '.tmp', '.lock']:
                continue
            if any(x in path.parts for x in ['.git', '__pycache__']):
                continue
            if path.name in ['unattended.log', 'UNATTENDED_STATUS.json', 'resource_account.lock']:
                continue
            files.append(path)
    files = sorted(set(files))
    file_manifest = [dict(path=str(p.relative_to(ROOT)), bytes=p.stat().st_size, sha256=sha(p)) for p in files]
    manifest_path = destination / 'FILE_MANIFEST.json'
    write(manifest_path, file_manifest)
    # Shard by uncompressed size so normal shards are <=64 MiB even for NPZ inputs.
    groups = []; current = []; size = 0
    for path in files:
        if current and size + path.stat().st_size > 64 * 1024 * 1024:
            groups.append(current); current = []; size = 0
        current.append(path); size += path.stat().st_size
    if current:
        groups.append(current)
    archives = []
    for i, group in enumerate(groups, 1):
        path = destination / f'CAD50_own512_V2_evidence_{i:03d}.zip'
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.write(manifest_path, 'FILE_MANIFEST.json')
            for source in group:
                archive.write(source, str(source.relative_to(ROOT)))
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None
        archives.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path), files=len(group), crc_verified=True))
    write(destination / 'DELIVERY_INDEX.json', dict(archives=archives, independent_zip_shards=True,
        weights_excluded='See repro_outputs/CHECKPOINT_MANIFEST.json; full weights stay on server',
        source_data_excluded='Read-only data/pools symlinks; identity recorded in branch configs',
        all_selected_files=len(files), source_commit=read(OUT / 'SOURCE_COMMIT.json')))


def main(wait):
    with (OUT / 'unattended.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.time() + 26 * 3600
        if wait:
            while True:
                maybe_fallback()
                ready = all(terminal(v) for v in VARIANTS) and not pending() and not worker_active()
                write(OUT / 'UNATTENDED_STATUS.json', dict(pid=os.getpid(), heartbeat=time.time(),
                    state='ready_to_package' if ready else 'waiting', training_terminal={v: terminal(v) for v in VARIANTS},
                    pending_evaluations=len(pending()), evaluation_worker_active=worker_active(),
                    training_launches_or_resumes_allowed=False))
                if ready or time.time() >= deadline:
                    break
                time.sleep(30)
        else:
            ready = all(terminal(v) for v in VARIANTS) and not pending() and not worker_active()
            assert ready, 'Branches/evaluations are not terminal; use --wait'
        try:
            summarize(None if ready else '26-hour completion wait expired; some training/evaluations missing')
            package()
            write(OUT / 'UNATTENDED_COMPLETE.json', dict(finished=time.time(), all_training_and_evaluation_terminal=ready,
                delivery=str(ROOT / 'delivery'), optimizer_updates_added=0))
        except BaseException as error:
            write(OUT / 'UNATTENDED_FAILURE.json', dict(error=str(error), traceback=traceback.format_exc()))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    main(parser.parse_args().wait)
