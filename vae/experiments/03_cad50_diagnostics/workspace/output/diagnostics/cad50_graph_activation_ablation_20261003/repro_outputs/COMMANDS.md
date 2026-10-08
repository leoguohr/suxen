# 命令记录

`documented`、`adapted` 和 `inferred` 表示来源，不表示已经执行。`not_run` 表示未执行；运行状态和证据记录实际尝试，不等同于科研结果验收通过。

## 环境准备建议

以下是环境准备建议，未执行。使用前请确认其是否适用于选定目标；不代表必须新建环境或安装依赖。

```bash
# [inferred]
# execution_status: not_run
# platforms: windows, macos, linux
python -m venv .venv
# [inferred]
# execution_status: not_run
# platforms: windows
.\.venv\Scripts\Activate.ps1
# [inferred]
# execution_status: not_run
# platforms: macos, linux
source .venv/bin/activate
```

## 资源观察

以下仅为观察记录，未执行资源准备。缺少常见目录不等于缺少必需资源。完整清单见 artifacts/assets/asset_manifest.json。

```bash
# No command recorded.
```

## 主命令

记录的执行状态：`not_run`。具体结果请检查对应的运行状态文件与日志。

```bash
# [documented]
# execution_status: not_run
/opt/conda/bin/python -B -u launch_all.py --source /guohaoran/nexus_fast_track/diagnostics/own512_v2_recipe_pair_20260923_resume_01
```

## 验证

单独验证命令的执行状态：`not_run`。内置结果比较另见备注和 status.json。

```bash
# No command recorded.
```

## 环境提示（不自动构成阻塞）

保留环境规划器的原始观察。修改环境前应检查这些提示；它们不证明选定命令无法运行。

- No top-level environment specification file was found.

## 备注

- README 路径：/Users/luthier/Documents/sophomore/output/diagnostics/cad50_graph_activation_ablation_20261003/README.md
- 检测到的顶层条目：README.md, __pycache__, data_objective.py, evaluation.py, launch_all.py, native_models.py, package_results.py, prepare.py, repro_outputs, support.py, test_cpu.py, train.py
- Defaulted to a virtualenv fallback because no environment file was detected.
- 主运行标签：来自 README 的 documented（code_block），章节 `正式训练命令`
- Planned skill chain: repo-intake-and-plan, env-and-assets-bootstrap, minimal-run-and-audit
- 未执行单独的验证命令。内置指标比较状态为 `not_evaluated`；期望值与容差见 `status.json.result_match`。
