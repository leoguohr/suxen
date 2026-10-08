# Comparability Report

- Mode: `train`
- Target repo: `/Users/luthier/Documents/sophomore/nexus_fast_track/diagnostics/teacher_cad50_terminal_ffn_pair_20260923`
- Comparability status: `qualified`
- README-first: `True`
- Documented command: `/opt/conda/bin/python -u H_terminal_ffn/train.py --mode terminal_ffn`
- Command source: `code_block`
- Command section: `执行命令`

## Comparison Anchors

- README documented command
- repository files used to interpret the README
- paper or baseline references only when explicitly resolved

## Protocol Deviations

- None.

## Patch And Execution Effects

- patches_applied=False
- readme_fidelity=preserved
- highest_patch_risk=low
- run_mode=full_kickoff
- dataset=unknown
- checkpoint_source=/guohaoran/nexus_fast_track/diagnostics/teacher_cad50_lr03_pair_20260921/B_lr03/checkpoint-new0500-step2500.pt

## Assumptions And Gaps

- README remains the primary source of truth.
- Environment creation should prefer isolated setup before any semantic code changes.
- Model architecture should remain unchanged unless the researcher explicitly requests otherwise.

## Interpretation

Treat results as directly comparable only when the documented command, data, preprocessing, checkpoint, metric, and baseline conditions remain aligned.
