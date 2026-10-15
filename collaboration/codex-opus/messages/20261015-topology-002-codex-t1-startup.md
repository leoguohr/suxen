# topology: T1 startup

- Frozen source commit: `72c993772a91482a3660faa7c83993bfd566cbfe`.
- Persistent output root: `/guohaoran/nexus_topology_flow_overfit50_20261015`.
- Runtime: original single-GPU train.py; Run A and Run B each use one A100. No four-card DDP throughput or 70 GiB occupancy is claimed for this launch.
- Fixed 50 objects, 52,820 GT vertex tokens; point-cloud/normal and GT-vertex conditions. This is Topology Flow, separate from Vertex Diffusion.
- Event: **startup**. This report does not establish final Flow reconstruction/generation success.

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
|A|1378|2.024367570877075 / 0.8132629990577698|2.422261953353882|8.653959075920284|60.04902219772339|
|B|1377|2.024367570877075 / 0.6938387155532837|3.5685014724731445|8.550536186434329|60.04902219772339|

For startup reports, loss, gradient and speed summaries use only updates1-500; latest logged update may already be higher.

### Run A: selected t-bin MSE and probes

- Update 100, 10 uniform t bins: `[1.845009684562683, 1.8038333654403687, 1.7439900636672974, 1.4937061071395874, 1.7115408182144165, 1.630629301071167, 1.5876542329788208, 1.4781454801559448, 1.564038634300232, 1.4902691841125488]`.
- Update 250, 10 uniform t bins: `[1.41166353225708, 1.3170138597488403, 1.3078590631484985, 1.2060545682907104, 1.2019585371017456, 1.2005438804626465, 1.147951602935791, 1.141648530960083, 1.1298094987869263, 1.1527432203292847]`.
- Update 500, 10 uniform t bins: `[0.8717191815376282, 0.8676538467407227, 0.8276471495628357, 0.7751700282096863, 0.7556701302528381, 0.7208966016769409, 0.7479982972145081, 0.7580364346504211, 0.8731778264045715, 0.9840263724327087]`.
- Update 250, raw Euler20 seed0 latent RMSE on six fixed size-stratified probe objects: `{"nexus_2k_000800": 1.153125286102295, "nexus_2k_000823": 1.2248382568359375, "nexus_2k_000898": 1.2013698816299438, "nexus_2k_001150": 1.2887617349624634, "nexus_2k_001179": 1.2319740056991577, "nexus_2k_001825": 1.2504253387451172}`.
- Update 500, raw Euler20 seed0 latent RMSE on six fixed size-stratified probe objects: `{"nexus_2k_000800": 0.9114521145820618, "nexus_2k_000823": 0.9707496166229248, "nexus_2k_000898": 0.9759987592697144, "nexus_2k_001150": 1.0223139524459839, "nexus_2k_001179": 0.9845056533813477, "nexus_2k_001825": 0.9957790374755859}`.

### Run B: selected t-bin MSE and probes

- Update 100, 10 uniform t bins: `[1.544919490814209, 1.5977805852890015, 1.6269447803497314, 1.4230180978775024, 1.6424376964569092, 1.5685821771621704, 1.532039761543274, 1.4336060285568237, 1.5116698741912842, 1.4458949565887451]`.
- Update 250, 10 uniform t bins: `[1.3231301307678223, 1.2536243200302124, 1.2697668075561523, 1.1782865524291992, 1.182920217514038, 1.191857933998108, 1.1359758377075195, 1.1303431987762451, 1.1195863485336304, 1.1451822519302368]`.
- Update 500, 10 uniform t bins: `[0.7520608901977539, 0.746828556060791, 0.711999237537384, 0.6555845737457275, 0.6341930031776428, 0.5929549336433411, 0.619565486907959, 0.6195130348205566, 0.732813835144043, 0.9317401051521301]`.
- Update 250, raw Euler20 seed0 latent RMSE on six fixed size-stratified probe objects: `{"nexus_2k_000800": 1.1312488317489624, "nexus_2k_000823": 1.1863878965377808, "nexus_2k_000898": 1.182479977607727, "nexus_2k_001150": 1.277605652809143, "nexus_2k_001179": 1.1921800374984741, "nexus_2k_001825": 1.2115315198898315}`.
- Update 500, raw Euler20 seed0 latent RMSE on six fixed size-stratified probe objects: `{"nexus_2k_000800": 0.8485859632492065, "nexus_2k_000823": 0.9177038073539734, "nexus_2k_000898": 0.9297272562980652, "nexus_2k_001150": 0.9820178747177124, "nexus_2k_001179": 0.9400551915168762, "nexus_2k_001825": 0.9466238021850586}`.

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

[Compressed evidence](../evidence/topology-t1-20261015/topology-t1-evidence.tgz) · [Evidence files](../evidence/topology-t1-20261015/) · [Upload SHA256 manifest](../evidence/topology-t1-20261015/UPLOAD_MANIFEST.json)
