"""Report all four cells and factorial differences, without claiming a unique cause."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def main(old_code):
    root, old = Path(__file__).resolve().parent, Path(old_code)
    locations = dict(Fourier_LN_post=old/'runs/V2_control', Fourier_LN_pre=old/'runs/Graph_LN_pre',
                     XYZ_LN_post=root/'runs/XYZ_LN_post', XYZ_LN_pre=root/'runs/XYZ_LN_pre')
    rows = {}
    for label, folder in locations.items():
        assert read(folder/'complete.json')['completed_updates'] == 2000
        evaluation = read(folder/'evaluations/step-02000/evaluation.json')
        assert evaluation['complete'] and len(evaluation['meshes']) == 50
        rows[label] = dict(counts=evaluation['counts'], joint_strict=evaluation['joint_strict'],
                           strict_uids=evaluation['joint_strict_uids'], large16=evaluation['large16'],
                           checkpoint=evaluation['checkpoint'])
    differences = {}
    for task in ('edge', 'face'):
        f_post, f_pre, x_post, x_pre = [rows[k]['counts'][task]['micro_f1'] for k in locations]
        differences[task] = dict(Fourier_effect_post=f_post-x_post, Fourier_effect_pre=f_pre-x_pre,
             postLN_effect_Fourier=f_post-f_pre, postLN_effect_XYZ=x_post-x_pre,
             interaction=(f_post-x_post)-(f_pre-x_pre))
    result = dict(scope='CAD50, seed0, 2000 five-mesh updates per cell', rows=rows,
                  micro_F1_differences=differences,
                  limitation='Single seed and finite CAD50 budget; does not establish unique cause of fixed100 historical bottleneck.')
    (root/'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = ['# Fourier × Graph/LN 完整结果', '',
             '四格均为同一初始化、2000次五mesh更新。以下是同预算末尾真实网络完整验收。', '',
             '| 条件 | Edge F1 | Face F1 | 联合严格成功 | 困难16条严格成功 |',
             '|---|---:|---:|---:|---:|']
    for label, row in rows.items():
        lines.append(f"| {label} | {row['counts']['edge']['micro_f1']:.9f} | {row['counts']['face']['micro_f1']:.9f} | {row['joint_strict']}/50 | {row['large16']['joint_strict']}/16 |")
    lines += ['', '## 差值（F1绝对值，不是相对百分比）', '', json.dumps(differences, ensure_ascii=False, indent=2),
              '', '交互项是两个条件下干预收益之差。逐UID取舍与所有FP/FN见comparison.json。',
              '单种子、短预算CAD50结果不能指定原100条历史瓶颈的唯一根因；四格指标也可能存在取舍。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-code', required=True)
    main(parser.parse_args().old_code)
