"""Produce a compact, evidence-bounded report from the fixed-checkpoint gradient probe."""
import hashlib
import json
from pathlib import Path
import math

ROOT = Path(__file__).resolve().parent
r = json.loads((ROOT/'complete.json').read_text())
p = r['provenance']
assert p['forward_count'] == 1 and p['optimizer_steps'] == 0
assert p['parameters_and_buffers_unchanged'] and p['rng_unchanged']
assert p['script_sha256'] == hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
groups = ['encoder','decoder_body','edge_head']
for key in ['small_l2','large_l2']:
    assert math.isclose(sum(r['comparison'][g][key]**2 for g in groups),r['comparison']['all'][key]**2,rel_tol=1e-12)
assert math.isclose(sum(r['comparison'][g]['dot'] for g in groups),r['comparison']['all']['dot'],rel_tol=1e-12)
assert sum(r['parameter_counts'][g] for g in groups) == r['parameter_counts']['all']
for g,c in r['comparison'].items():
    assert math.isclose(c['small_over_large_l2'],c['small_l2']/c['large_l2'],rel_tol=1e-12)
    assert math.isclose(c['cosine'],c['dot']/(c['small_l2']*c['large_l2']),rel_tol=1e-12)
names = dict(encoder='Encoder',decoder_body='Decoder body',edge_head='edge head',all='全模型')
lines = ['# Step6000：Small / Large Soft4梯度比较', '',
         '同一个step6000 checkpoint，一次完整两mesh前向，共用同一计算图，分别对每条mesh的Soft4反传。无optimizer、无参数更新、无梯度裁剪，不乘联合loss中的1/2。τ=1、membership detach、FP32组归约；梯度点积与范数用FP64累加。', '',
         '|部分|cos(g_S,g_L)|‖g_S‖|‖g_L‖|‖g_S‖/‖g_L‖|比例百分数|',
         '|---|---:|---:|---:|---:|---:|']
for g in groups+['all']:
    c=r['comparison'][g]
    lines.append(f"|{names[g]}|{c['cosine']:.8f}|{c['small_l2']:.9g}|{c['large_l2']:.9g}|{c['small_over_large_l2']:.9g}|{100*c['small_over_large_l2']:.6f}%|")
lines += ['', 'Encoder、Decoder body、edge head三者互不重叠；Decoder body不包含edge head。全模型为三者之和，111,491,168个参与边重建的参数。冻结的logvar和face head在本目标下梯度为0，不影响全模型范数。', '',
          '四处余弦都接近0，而非明显接近−1。全模型small梯度只有large的0.282%，large梯度范数约为small的354.6倍。这个checkpoint上的原始联合梯度主要由large决定；本结果不支持“small以同等或更强梯度，强烈反向拖住large”这一解释。', '',
          '结论只针对这个后期checkpoint和原始梯度，不能排除早期训练干扰，也不能由此直接确定Adam实际更新方向或最终残留错误的根因。', '',
          '|本次前向|Soft4|TP / FP / FN|Edge F1|','|---|---:|---|---:|']
for name,x in zip(['Small','Large'],r['metrics']):
    lines.append(f"|{name}|{x['soft4_loss']:.9g}|{x['tp']} / {x['fp']} / {x['fn']}|{100*x['edge_f1']:.6f}%|")
lines += ['', '原训练step6000记录large为TP7709、FP2、FN10；本次同权重重新前向为TP7711、FP1、FN8。保留原Flash/CUDA后端，存在数值非逐位确定性；两条梯度始终来自本次同一个前向，未混用两个状态。', '',
          '|重复反向检查|全模型余弦|相对L2差异|','|---|---:|---:|']
for label in ['small','large']:
    c=r['backward_repeat'][label]['all']
    lines.append(f"|{label}同loss重复|{c['cosine']:.9f}|{100*c['difference_over_first_l2']:.6f}%|")
lines += ['', '重复反向的方向非常接近，范数比和“没有强烈反向冲突”的结论不依赖微小数值波动。所有参数、buffer、RNG在诊断前后不变，checkpoint文件哈希未变化。', '',
          f"Checkpoint：`{r['checkpoint']}`", f"SHA256：`{r['checkpoint_sha256']}`", '',
          '完整原始small/large参数梯度保存在服务器本目录的per_mesh_parameter_gradients.pt；本地complete.json包含全部数值、模块参数量、重复检查与来源记录。', '']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
(ROOT/'verification.json').write_text(json.dumps(dict(disjoint_partition_and_full_norms_verified=True,
    requested_ratio_small_over_large_verified=True,parameters_unchanged=True,optimizer_steps=0),indent=2)+'\n')
print('\n'.join(lines[:12]))
