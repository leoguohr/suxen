# D15 50-object update-7000 continuation

Use the verified source checkpoint `/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_overfit50_d15_20260929/run/checkpoint-019000.pt` (SHA-256 `c27503c8a7e4b52407685e4532608fbaf281bfb771d3f1917e94e172ae9dd8fc`, 27,990,489,354 bytes). The wrapper performs a CPU structure check, verifies both selected GPUs are present and idle and both training locks are free, then invokes the original `overfit50_train.py --resume`. It does not change model, data, optimizer, RNG, effective batch, sampling, or evaluation code.

After placing this directory beside the original server task directory, launch once on the verified two-A100 host:

```bash
CUDA_VISIBLE_DEVICES=GPU-6a420b8b-6596-aba3-657a-02729d35afa3,GPU-c3c69625-6ea3-026c-c731-5c7dd735e1b7 \
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
/guohaoran/envs/nexus-algo/bin/python -u \
/ssdwork/guohaoran/nexus_fast_track/diagnostics/nexus_overfit50_resume7000_20260930/resume_pipeline.py \
--gpu-uuids GPU-6a420b8b-6596-aba3-657a-02729d35afa3,GPU-c3c69625-6ea3-026c-c731-5c7dd735e1b7
```

The new `run/` and `/tmp/nexus_overfit50_resume7000_20260930/run/` must not already exist. The wrapper copies the complete old training log into the new run. The original trainer archives that copy and retains updates 1–7000 before writing 7001–10000; it saves at 7001 and every 200 updates. It prunes only checkpoints in the new run and new NVME directory. The source checkpoint stays in the original run.

After verified training completion at update 10000/step 22000, the wrapper hashes the final durable and NVME checkpoints, runs the original two-seed full-tree and GT-parent evaluation, and creates `NEXUS_Overfit50_D15_Resume7000_results.zip` with the runtime condition and label NPZ arrays but without large weights. `audit/pipeline_status.json` records progress or a failure with no automatic retry. Structural preflight alone does not prove that resume executed successfully; the saved update-7001 checkpoint is the first durable evidence of resumed execution.
