"""Verify the two-mesh trajectory and report simultaneous exact edge reconstruction."""
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
UIDS = ['nexus_2k_000387', 'nexus_2k_001849']
NAMES = ['Small (386 vertices)', 'Large (2575 vertices)']
COLORS = ['#16887a', '#bd4149']


def read(path):
    return json.loads(path.read_text())


def window_stats(trace):
    flags = [x['I_t'] for x in trace]
    streak = longest = 0
    for flag in flags:
        streak = streak + 1 if flag else 0
        longest = max(longest, streak)
    meshes = {}
    for i, uid in enumerate(UIDS):
        rows = [x['rows'][i] for x in trace]
        meshes[uid] = {key: np.quantile([r[key] for r in rows], [0, .5, 1]).tolist()
                       for key in ['edge_f1', 'tp', 'fp', 'fn', 'soft4_loss', 'balanced_loss']}
        meshes[uid]['perfect_count'] = sum(r['is_perfect'] for r in rows)
    return dict(start_step=trace[0]['step'], end_step=trace[-1]['step'], evaluations=len(trace),
                first_both_step=next((x['step'] for x in trace if x['I_t']), None),
                both_count=sum(flags), longest_both_streak=longest, meshes=meshes)


traces, metas, completes = {}, {}, {}
script_sha = hashlib.sha256((ROOT/'run.py').read_bytes()).hexdigest()
for phase, start, end, rates in [('learning', 0, 3000, (1e-5, 1e-4)),
                                  ('refine', 3000, 4000, (1e-6, 1e-5))]:
    p = ROOT/phase
    t = [json.loads(line) for line in (p/'trace.jsonl').read_text().splitlines()]
    meta, done = read(p/'provenance.json'), read(p/'complete.json')
    assert [x['step'] for x in t] == list(range(start, end+1))
    assert meta['script_sha256'] == script_sha and meta['selected_uids'] == UIDS
    assert meta['loaded_weights_verified_tensor_equal']
    assert meta['optimizer_restored_exactly'] == (phase == 'refine')
    assert done['optimizer_final_step'] == end and done['final'] == t[-1]
    assert done['frozen_and_rng_checks_passed'] and done['original_sources_unchanged']
    for j, x in enumerate(t):
        assert x['phase'] == phase and len(x['rows']) == 2
        assert (x['encoder_lr'], x['decoder_lr']) == rates
        assert x['rng_unchanged'] and x['frozen_unchanged']
        for i, r in enumerate(x['rows']):
            assert r['uid'] == UIDS[i]
            assert r['tp']+r['fn'] == [1152, 7719][i]
            assert r['tn']+r['fp'] == [73153, 3306306][i]
            assert r['edge_f1'] == 2*r['tp']/(2*r['tp']+r['fp']+r['fn'])
            assert r['is_perfect'] == (r['fp'] == r['fn'] == 0)
            assert np.isclose(r['soft4_loss'], np.mean([g['mean'] for g in r['soft4_groups'].values()]))
            if j:
                assert np.isclose(r['mu_relative_delta'], r['mu_delta_l2']/(t[j-1]['rows'][i]['mu_l2']+1e-12))
        assert x['I_t'] == int(all(r['is_perfect'] for r in x['rows']))
        assert np.isclose(x['objective'], np.mean([r['soft4_loss'] for r in x['rows']]))
    traces[phase], metas[phase], completes[phase] = t, meta, done
for key in ['initial_sha256', 'source_sha256', 'loss_script_sha256', 'soft4_tau',
            'soft4_epsilon', 'soft4_membership', 'soft4_group_reduction', 'threshold', 'mode',
            'loss_reduction', 'clip_norm', 'weight_decay', 'hardware', 'torch_version', 'cuda_version']:
    assert metas['learning'][key] == metas['refine'][key], key
assert metas['learning']['initial_sha256'] == '0f917dfd2d30175b34abcd9c3ff3ec612b57d61660fb6e4c19c6bd23caebb0fd'
assert metas['learning']['start_checkpoint_sha256'] == metas['learning']['initial_sha256']
boundary = read(ROOT/'refine/resume_verification.json')
assert boundary['model_and_adam_restored']
assert boundary['learning_step3000']['rows'] == traces['learning'][-1]['rows']
assert boundary['refinement_step3000']['rows'] == traces['refine'][0]['rows']

# Count one observation per completed update; preserve both boundary forwards separately.
combined = traces['learning'][:-1] + traces['refine']
assert [x['step'] for x in combined] == list(range(4001))
first, count, streak, longest = None, 0, 0, 0
intervals = []
for x in combined:
    if x['I_t']:
        if first is None:
            first = x['step']
        if intervals and intervals[-1][1]+1 == x['step']:
            intervals[-1][1] = x['step']
        else:
            intervals.append([x['step'], x['step']])
    count += x['I_t']
    streak = streak+1 if x['I_t'] else 0
    longest = max(longest, streak)
    assert (x['first_both_step'], x['both_count'], x['both_streak'], x['longest_both_streak']) == (first, count, streak, longest)

summary = dict(overall=window_stats(combined), perfect_intervals=intervals,
               phases={p: window_stats(t) for p, t in traces.items()},
               tail_windows={str(n): window_stats(combined[-n:]) for n in [200, 500]},
               final=combined[-1], learning_step3000=traces['learning'][-1],
               refinement_step3000=traces['refine'][0],
               seconds=sum(d['seconds'] for d in completes.values()))
summary['tail200_mu_relative_delta_medians'] = {
    phase: {uid: float(np.median([x['rows'][i]['mu_relative_delta'] for x in t[-200:]]))
            for i, uid in enumerate(UIDS)} for phase, t in traces.items()}
verification = dict(initial_sha256=metas['learning']['initial_sha256'],
                    resume_sha256=metas['refine']['start_checkpoint_sha256'],
                    weights_and_adam_restored=True, all_4001_steps_verified=True,
                    every_update_two_full_meshes_equal_weight=True,
                    all_hard_counts_losses_lr_and_counters_verified=True,
                    boundary_I_t_learning=traces['learning'][-1]['I_t'],
                    boundary_I_t_refine=traces['refine'][0]['I_t'],
                    counting='learning steps0..2999 + refinement steps3000..4000; no duplicate boundary')
for name, data in [('summary.json', summary), ('verification.json', verification)]:
    (ROOT/name).write_text(json.dumps(data, indent=2)+'\n')
(ROOT/'combined_trace.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in combined))

fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
steps = [x['step'] for x in combined]
for i, (name, color) in enumerate(zip(NAMES, COLORS)):
    rows = [x['rows'][i] for x in combined]
    axes[0, 0].plot(steps, [100*r['edge_f1'] for r in rows], color=color, label=name, lw=.7)
    for key, style in [('fp', '-'), ('fn', '--')]:
        axes[0, 1].plot(steps, [r[key] for r in rows], color=color, ls=style, label=name+' '+key.upper(), lw=.7)
    axes[1, 0].plot(steps, [r['soft4_loss'] for r in rows], color=color, lw=.7)
    axes[1, 1].plot(steps, [r['balanced_loss'] for r in rows], color=color, lw=.7)
    axes[2, 0].plot(steps, [r['mu_relative_delta'] for r in rows], color=color, lw=.7)
axes[0, 0].set(ylabel='Hard Edge F1 (%)', ylim=(-1, 101)); axes[0, 0].legend()
axes[0, 1].set(ylabel='FP / FN', yscale='symlog'); axes[0, 1].legend(fontsize=8)
axes[1, 0].plot(steps, [x['objective'] for x in combined], color='black', lw=.5, alpha=.6, label='Mean objective')
axes[1, 0].set(ylabel='Soft4 loss', yscale='log'); axes[1, 0].legend()
axes[1, 1].set(ylabel='GT-balanced BCE', yscale='log')
axes[2, 0].set(ylabel='Relative change of mu', yscale='log')
flags = np.array([x['I_t'] for x in combined])
rolling = np.convolve(flags, np.ones(200), 'valid')
axes[2, 1].plot(steps[199:], rolling, color='#5c477d', label='Both-perfect count in trailing 200 steps')
perfect_steps = [x['step'] for x in combined if x['I_t']]
if perfect_steps:
    axes[2, 1].scatter(perfect_steps, [0]*len(perfect_steps), marker='|', s=10, color='#5c477d', label='Both perfect')
axes[2, 1].set(ylabel='Both-perfect evaluations / 200', ylim=(-3, 203)); axes[2, 1].legend(fontsize=8)
for ax in axes.flat:
    ax.axvline(3000, color='gray', ls='--', lw=1)
    ax.set_xlabel('Completed updates'); ax.grid(alpha=.2)
fig.suptitle('Two-mesh Soft4: same random initial.pt, both complete meshes every update\nE/D LR: 1e-5 / 1e-4 until 3000, then 1e-6 / 1e-5 with restored Adam to 4000')
fig.savefig(ROOT/'training_curve.png', dpi=150); plt.close(fig)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
t = traces['refine']; steps_refine = [x['step'] for x in t]
for i, (name, color) in enumerate(zip(NAMES, COLORS)):
    axes[0].plot(steps_refine, [x['rows'][i]['edge_f1']*100 for x in t], color=color, lw=.8, label=name)
    for key, style in [('fp', '-'), ('fn', '--')]:
        axes[1].plot(steps_refine, [x['rows'][i][key] for x in t], color=color, ls=style, lw=.8, label=name+' '+key.upper())
axes[0].set(ylabel='Hard Edge F1 (%)'); axes[0].legend()
axes[1].set(ylabel='FP / FN', yscale='symlog', xlabel='Completed updates'); axes[1].legend(ncol=2)
for ax in axes:
    ax.grid(alpha=.2)
fig.suptitle('Two-mesh Soft4: refinement phase (LR x0.1)')
fig.savefig(ROOT/'refinement_curve.png', dpi=150); plt.close(fig)

s = summary['overall']
lines = ['# Two-mesh + Soft4：3000步学习 + 1000步精修', '',
         '两条完整mesh每步同时参与，L=0.5×Soft4_small+0.5×Soft4_large。从同一个随机initial.pt开始，前3000步E/D LR=1e-5/1e-4；恢复step3000模型与Adam全部状态后，E/D LR=1e-6/1e-5继续至4000。', '',
         'τ=1、membership detach、FP32组归约、μ mode、Face=KL=wd=0、clip=1、固定数据及原Flash后端。验收使用同一次前向的两条mesh，threshold=0；同时FP=FN=0才记I_t=1。', '',
         f"首次同时100%：{s['first_both_step']}；总次数：{s['both_count']}；最长连续：{s['longest_both_streak']}。None表示未出现。", '',
         '|窗口|同时100%次数|small F1 min / median / max|large F1 min / median / max|',
         '|---|---:|---|---|']
for n, w in summary['tail_windows'].items():
    fs = [' / '.join(f'{100*v:.5f}%' for v in w['meshes'][uid]['edge_f1']) for uid in UIDS]
    lines.append(f"|最后{n}步|{w['both_count']}/{n}|{fs[0]}|{fs[1]}|")
lines += ['', '|阶段 / mesh|TP / FP / FN|Edge F1|Soft4|GT-balanced BCE|', '|---|---|---:|---:|---:|']
for label, x in [('学习阶段step3000', traces['learning'][-1]), ('精修结束step4000', combined[-1])]:
    for name, r in zip(NAMES, x['rows']):
        lines.append(f"|{label} / {name}|{r['tp']} / {r['fp']} / {r['fn']}|{100*r['edge_f1']:.6f}%|{r['soft4_loss']:.8g}|{r['balanced_loss']:.8g}|")
lines += ['', '两阶段step3000各进行了一次前向。权重及Adam状态逐张量核对相等；保留原CUDA/Flash后端，不声称数值逐位确定。统计轨迹仅计一次step3000，使用精修阶段重放；两次原始结果均保留在summary.json及resume_verification.json。', '',
          '本次未通过两条同时100%的验收。small最后500步全部100%；large最终还有17条错边和97条漏边。降低学习率后large从69.65%改善至99.26%，末段仍有改善；这不足以认定网络容量不够，也不能将large-only成功直接等同于two-mesh联合训练成功。', '',
          '每步μ相对变化在两个阶段各自最后200步的中位数：small从约1.817%降到0.145%，large从约1.462%降到0.144%。它说明精修阶段编码变化明显变小，但这一统计本身不证明唯一根因。', '',
          '是否出现过完全重建、是否持续保持完全重建，应分别判断。本实验仅验收这两条训练mesh的边重建，Face与KL未训练，不代表面重建或泛化成功。', '',
          '模型和Adam每200步保存；首次同时100%若出现则额外保存。checkpoint位于服务器同名diagnostics目录，完整日志及报告已保存在本地。', '',
          '![完整曲线](training_curve.png)', '', '![精修阶段](refinement_curve.png)', '']
(ROOT/'REPORT.md').write_text('\n'.join(lines))
print(json.dumps(dict(overall=s, tails=summary['tail_windows'], final=summary['final']), indent=2))
