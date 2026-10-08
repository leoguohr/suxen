# 本轮CPU收尾命令

训练/采样启动入口保留于launch.json、runtime/scripts/phase3_pipeline.py；历史流水线仅供审计，不要为复算再次运行它。
本轮仅执行以下收尾命令（ROOT为解压后的Phase3根目录）：

```sh
export CUDA_VISIBLE_DEVICES=''
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONPATH="$ROOT/runtime/d15_code:$ROOT/runtime"
python "$ROOT/runtime/scripts/phase3_compare.py" --root "$ROOT" --baseline "$ROOT/baseline_phase2"
python "$ROOT/verify_final_checkpoint.py"
python "$ROOT/finalize_evidence.py"
```

服务器实际ROOT=/ssdwork/guohaoran/nexus_fast_track/diagnostics/phase3_coarse_weight_20260928，baseline是相邻的nexus_phase2_d15_10_20260926，解释器/guohaoran/envs/nexus-algo/bin/python。
比较脚本要求comparison输出目录尚不存在；重算时请先将已有comparison改名留存。权重校验脚本和打包脚本按服务器布局使用；没有服务器权重时不能执行权重校验。
本轮补充脚本只写审计/交付文件，未改动runtime内代码。
