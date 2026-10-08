"""One-time source verification and bounded GPU calibration; no training steps."""
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'code'))
from _faces_reference import atomic_json, file_sha
from latent_data import LatentCache

GPU = 'GPU-208b1847-4ab0-f090-b8d4-f21ebd908b00'
PRIOR = Path('/guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_continue_20261006')
VAE = Path('/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_hard_continue1000_20260930/run/checkpoints/vae-1000.pt')


def main():
    if (ROOT/'campaign.json').exists(): raise RuntimeError('Campaign already exists; clock must not reset')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == GPU
    query = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader'],text=True).strip()
    processes = subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True).strip()
    assert query.startswith(GPU) and not processes, (query, processes)
    cache = LatentCache(PRIOR/'cache')
    for uid in cache.uids: cache.get(uid)
    assert cache.sha256 == 'f14f796c421cab6b9d5206047f927ea7c3490ff3c2435b2ebd38904a749c27e3'
    assert file_sha(VAE) == 'e89b8078b7abb0ca7b0c2c44f9f20382ec5b742f2aee948f209d642852714d15'
    baseline = json.loads((PRIOR/'vae_baseline'/'summary.json').read_text())
    assert baseline['complete']
    now = time.time()
    campaign = dict(schema_version=1,started_unix=now,deadline_unix=now+14400,
        max_gpu_seconds=14400,max_updates_per_group=8000,authorized_gpu_uuid=GPU,
        order=['c1_fourier','c2_teacher'],allocation='Approximately equal remaining wall time per group; includes save and evaluation',
        gpu_query=query,cache=str(cache.root),cache_sha256=cache.sha256,
        vae_checkpoint=str(VAE),vae_checkpoint_sha256=file_sha(VAE),baseline=str(PRIOR/'vae_baseline'),
        normalization_sha256=cache.manifest['normalization_sha256'],selection_sha256=cache.manifest['data']['selection_sha256'],
        source_code_sha256={p.name:file_sha(p) for p in sorted((ROOT/'code').glob('*.py'))},
        config_sha256={p.name:file_sha(p) for p in sorted((ROOT/'configs').glob('*.json'))},
        storage=dict(root=str(ROOT),persistent_mount='storage GPFS',quota='unknown',
            prior_destination_error='/ssdwork/guohaoran: errno122 Disk quota exceeded during code upload; no cleanup attempted',
            full_checkpoint_bytes_estimate=28008000000,retained_per_group_max_without_extra_best=3,
            peak_two_groups_checkpoints=7,peak_two_groups_bytes_estimate=196056000000,
            validation='Actual complete update1 model/AdamW/RNG/cursor is fsynced, SHA checked and atomically renamed; previous checkpoint kept until success; this proves one save only'))
    atomic_json(ROOT/'campaign.json',campaign)
    atomic_json(ROOT/'perf-budget.json',dict(schema_version=1,authorized_gpu_uuid=GPU,max_gpu_seconds=1200,max_optimizer_updates=1))
    cmd=[sys.executable,str(ROOT/'code/performance_preflight.py'),'--config',str(ROOT/'configs/c1_fourier.json'),
        '--cache',str(cache.root),'--output',str(ROOT/'perf'),'--budget-file',str(ROOT/'perf-budget.json')]
    with (ROOT/'perf.log').open('x') as log:
        child = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try: code=child.wait(timeout=1190)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
            atomic_json(ROOT/'perf-timeout.json',dict(pid=child.pid,ended_unix=time.time(),reason='1200-second preflight hard limit; no automatic retry'))
            raise
    if code: raise RuntimeError('Preflight failed; inspect perf.log and perf/report.json')
    result=json.loads((ROOT/'perf/report.json').read_text())
    assert result['complete'] and result['adam_steps']==[0]
    print(json.dumps(dict(preflight_complete=True,elapsed_since_campaign_start=time.time()-now,selected=result['selected_measurement']),indent=2),flush=True)


if __name__ == '__main__': main()
