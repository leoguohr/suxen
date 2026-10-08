# Actual commands and durable logs

Working directory: `/guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923`.

```sh
CUDA_VISIBLE_DEVICES='' PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /opt/conda/bin/python test_core.py
CUDA_VISIBLE_DEVICES='' PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python /opt/conda/bin/python prepare_initialization.py
/opt/conda/bin/python launch_local.py --role A
/opt/conda/bin/python launch_local.py --role B
/opt/conda/bin/python launch_local.py --role eval
/opt/conda/bin/python postprocess/check_paired_logs.py
/opt/conda/bin/python -u postprocess/finish_unattended.py --wait
```

The launch records contain actual argv, PID, allocated GPU UUID and host. Training uses the installed RigorPilot run-train supervisor; `_runtime/` contains command specs, stdout, stderr, status and events. The orchestrator intake did not launch training. `release.json` records the passed initialization gate before optimizer updates began. The completion coordinator is independently detached; its launch receipt and log are `unattended_launch.json` and `unattended.log`. These commands are evidence, not instructions to rerun existing jobs.
