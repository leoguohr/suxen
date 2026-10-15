# topology: T1 launch

- Frozen source commit: `72c993772a91482a3660faa7c83993bfd566cbfe`.
- Persistent output root: `/guohaoran/nexus_topology_flow_overfit50_20261015`.
- Runtime: original single-GPU train.py; Run A and Run B each use one A100. No four-card DDP throughput or 70 GiB occupancy is claimed for this launch.
- Fixed 50 objects, 52,820 GT vertex tokens; point-cloud/normal and GT-vertex conditions. This is Topology Flow, separate from Vertex Diffusion.
- Event: **launch**. This report does not establish final Flow reconstruction/generation success.

## Preflight

- Original CPU gate: passed. Cache: 50/50 objects, `d9_center_max_offset=0.0`.
- Mu decoding exactly matches edge TP/FP/FN=154973/125/5 and face TP/FP/FN=102878/870/12.
- Original single-GPU 20-update speed check: 8.5808 s/update, peak allocated 60.046 GiB; chunk_tokens=0, activation checkpointing enabled.
- Raw compute estimate for 10,000 updates: 23.84 hours per run, excluding startup, probes, checkpoint saves and evaluations.
- Speed-check Flash/SDPA comparison: `{"flash_available": true, "max_abs_diff": 0.0, "rel_rms_diff": 0.0}`. The fresh zero-output head can make this comparison trivial; the nonzero warm-start follow-up is included separately when available.

## Frozen VAE decoder noise tolerance

|alpha|edge micro-F1|face micro-F1|joint strict / 50|overflow objects|normalized latent RMSE|
|---:|---:|---:|---:|---:|---:|
|0.0|0.999580748|0.995731666|15/50|0|0.000000000|
|0.02|0.999596864|0.995755678|15/50|0|0.020000947|
|0.05|0.999561404|0.995707490|13/50|0|0.050002370|
|0.1|0.999506600|0.995659012|13/50|0|0.100004741|
|0.2|0.999148815|0.994886135|9/50|0|0.200009483|
|0.3|0.998282125|0.992934281|8/50|0|0.300014229|
|0.5|0.992797088|0.980079604|2/50|0|0.500023697|

The tested alpha grid brackets the face-F1<0.99 transition; it does not identify a precise tolerance threshold or prove Flow quality.

## Actual A/B training evidence

|run|latest complete logged update|reported loss first / last|max grad norm|median s/update|peak allocated GiB|
|---|---:|---|---:|---:|---:|
|A|19|2.024367570877075 / 2.020005464553833|0.1892530769109726|8.610232688020915|60.04902219772339|
|B|14|2.024367570877075 / 2.0205330848693848|0.3382273316383362|8.536099092103541|60.04902219772339|

For startup reports, loss, gradient and speed summaries use only updates1-500; latest logged update may already be higher.

### Run A: selected t-bin MSE and probes

- First500 startup evidence is incomplete; no first500 comparison is claimed.

### Run B: selected t-bin MSE and probes

- First500 startup evidence is incomplete; no first500 comparison is claimed.

The probe covers six objects, not the full 50-object topology evaluation. Loss/t-bin values are training diagnostics. Read raw files for checkpoint-save status; a log update is not itself a durable checkpoint.

## Launch / migration record

```json
{
  "host": "bl66k0ivsj6g2-0",
  "pid": 528,
  "source_commit": "72c993772a91482a3660faa7c83993bfd566cbfe",
  "execution": "Original single-GPU train.py, one independent experiment per GPU; no DDP adapter",
  "jobs": {
    "A_fresh": {
      "gpu": 0,
      "command": [
        "/guohaoran/envs/nexus-algo/bin/python",
        "-u",
        "/guohaoran/nexus_topology_flow_overfit50_20261015/preflight_20261015/source/topology/flow_overfit50_20261015/train.py",
        "--cache",
        "/guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_continue_20261006/cache",
        "--vae",
        "/guohaoran/nexus_topology_flow_overfit50_20261015/inputs/vae-1000.pt",
        "--output",
        "/guohaoran/nexus_topology_flow_overfit50_20261015/A_fresh"
      ],
      "log": "/guohaoran/nexus_topology_flow_overfit50_20261015/A_fresh.log",
      "max_updates": 10000,
      "stop_at_pass": 1,
      "pid": 531
    },
    "B_s0warm": {
      "gpu": 1,
      "command": [
        "/guohaoran/envs/nexus-algo/bin/python",
        "-u",
        "/guohaoran/nexus_topology_flow_overfit50_20261015/preflight_20261015/source/topology/flow_overfit50_20261015/train.py",
        "--cache",
        "/guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_continue_20261006/cache",
        "--vae",
        "/guohaoran/nexus_topology_flow_overfit50_20261015/inputs/vae-1000.pt",
        "--output",
        "/guohaoran/nexus_topology_flow_overfit50_20261015/B_s0warm",
        "--init-vertex",
        "/guohaoran/tmp/nexus_vertex_arch_sweep_recovery_20261007/runs/S0/checkpoint-034000.pt"
      ],
      "log": "/guohaoran/nexus_topology_flow_overfit50_20261015/B_s0warm.log",
      "max_updates": 10000,
      "stop_at_pass": 1,
      "pid": 532
    }
  },
  "gpu_inventory": "0, GPU-7292ff1a-ad35-7511-dd59-eec27d3ea414, 81920 MiB\n1, GPU-675508bf-7357-b900-1f47-6ebce2a27584, 81920 MiB\n",
  "client_expiry": "2026-10-15T22:22:01+08:00",
  "seconds_until_save_stop": 12240.0,
  "started_unix": 1792059813.3246024,
  "preflight_reused_after_current_hash_runtime_verification": true,
  "early_stop_note": "Original stop-at-pass may finish at a milestone before LR decay; not guaranteed 10000 actual updates"
}
```

## Evidence boundaries

- Earlier preflight evidence is reused with source hashes and original paths. Current-run metrics are drawn only from these formal run directories.
- Old experimental/checkpoint directories were read only. No checkpoint or prediction tensor is included in this publication.
- All original model/loss/sampler code remains frozen. Reporting scripts are external to the source snapshot.

[Compressed evidence](../evidence/topology-t1-launch-20261015/topology-t1-evidence.tgz) · [Evidence files](../evidence/topology-t1-launch-20261015/) · [Upload SHA256 manifest](../evidence/topology-t1-launch-20261015/UPLOAD_MANIFEST.json)
