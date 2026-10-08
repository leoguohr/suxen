# Training Commands

## Setup

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

## Assets

```bash
# [documented]
# execution_status: not_run
# 来自 README.md 的资源线索：/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_lr03_pair_20260921/B_lr03/checkpoint-new0500-step2500.pt；准备前先确认是否与选定命令相关。
```

## Training run

```bash
# [documented]
# execution_status: not_run
/opt/conda/bin/python -u H_terminal_ffn/train.py --mode terminal_ffn
```

## Verification

```bash
# [inferred]
# execution_status: not_run
python - <<'PY'
import pathlib
print(pathlib.Path('train_outputs/status.json').exists())
PY
```

## Notes

- README 路径：/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_cad50_terminal_ffn_pair_20260923/README.md
- 检测到的顶层条目：.git, H_terminal_ffn, README.md, baseline_source
- Defaulted to a virtualenv fallback because no environment file was detected.
- 主运行标签：来自 README 的 documented（code_block），章节 `执行命令`
- Planned skill chain: repo-intake-and-plan, env-and-assets-bootstrap, run-train
