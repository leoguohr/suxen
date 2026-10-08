"""Summarize completed free-embedding training without executing any updates."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent
run = root / 'run'
rows = [json.loads(line) for line in (run/'updates.jsonl').read_text().splitlines()]
done = json.loads((run/'complete.json').read_text())
assert len(rows) == 2001 and [x['step'] for x in rows] == list(range(2001))
assert all(x['tp']+x['fn'] == 2406 and x['tn']+x['fp'] == 320400 for x in rows)
assert done['perfect_updates'] == sum(x['perfect'] for x in rows[1:])
assert done['final'] == rows[-1]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout='constrained')
steps = [x['step'] for x in rows]
axes[0].plot(steps, [x['fp'] for x in rows], label='FP: extra edges', color='#ce5839')
axes[0].plot(steps, [x['fn'] for x in rows], label='FN: missing edges', color='#267ca8')
axes[0].set(ylabel='Full edge-graph errors', title='All 322,806 pairs; threshold = 0')
axes[0].legend()
axes[1].plot(steps, [x['f1']*100 for x in rows], color='#287654')
axes[1].set(ylabel='Edge F1 (%)', title='Strict success requires FP = FN = 0')
axes[2].plot(steps, [x['edge_soft4'] for x in rows], label='Edge Soft4')
axes[2].plot(steps, [x['objective'] for x in rows], label='Training objective = Edge Soft4 / 4')
axes[2].set(ylabel='Loss', title='Fully differentiable memberships')
axes[2].legend()
for ax in axes:
    ax.set_xlabel('Actual optimizer updates')
    ax.grid(alpha=.2)
    if done['first_perfect_step'] is not None:
        ax.axvline(done['first_perfect_step'], color='#555555', linestyle='--', alpha=.6)
fig.suptitle('804 vertices | free 32D Edge embeddings | Adam 1e-3 | clip 1', fontsize=14)
fig.savefig(root/'training_curves.png', dpi=180)
fig.savefig(root/'training_curves.pdf')

table = ['| Step | TP | FP | FN | F1 | Edge Soft4 |', '|---:|---:|---:|---:|---:|---:|']
for step in [0,50,100,200,500,1000,2000]:
    x=rows[step]
    table.append(f"| {step} | {x['tp']} | {x['fp']} | {x['fn']} | {100*x['f1']:.6f}% | {x['edge_soft4']:.9f} |")
conclusion = ('获得当前32维评分表示对804点GT边图的严格可行解；这不是完整AE/VAE成功。'
              if done['first_perfect_step'] is not None else
              '2000步内未获得完整GT边图的严格可行解；不能据此判定32维表达不可能。')
report = f'''# 804点自由Edge embedding拟合

{conclusion}

本次只训练804×32=25,728个表示参数，从only804_mu step1000导出的edge_head_raw初始化。
每次forward仅中心化一次；沿用16+16拆分、原spacetime评分和scale=0.9306077080970389。
全部322,806对参与fully-diff Soft4，外层除4；fresh Adam lr=1e-3、betas=(0.9,0.999)、eps=1e-8、wd=0、clip=1。
Encoder/Decoder/其他head没有载入；无sampling、KL、Face目标。执行2000个更新，无延长。

初始表示、全部logits、Edge loss及raw表示梯度与导出快照bitwise一致；重复forward/backward一致。

{chr(10).join(table)}

- 首次严格全对step：{done['first_perfect_step']}
- 2000个更新后状态中的严格全对次数：{done['perfect_updates']}
- 最长连续全对：{done['longest_consecutive_perfect']}
- 最小FP+FN：{done['minimum_total_errors']}，首次达到的step：{done['best_step']}
- 最后200步：{json.dumps(done['last_200'],ensure_ascii=False)}
- 最后500步：{json.dumps(done['last_500'],ensure_ascii=False)}
- 2000步均存在非零参数更新：{done['every_update_nonzero']}
- 发生裁剪的更新数：{done['clipped_updates']}
- 训练及过程保存耗时：{done['wall_seconds']:.1f}秒
- 原快照文件哈希未改变：{done['source_files_unchanged']}

## 结论边界

保持当前32维、评分、scale、中心化和fully-diff Soft4，在自由表示参数化下已严格拟合完整GT边图。
因此，这条804点边图不能再归因为“当前32维评分根本表示不了”或“Soft4必然无法拟合”。
截图中的局部梯度取舍确实存在，但没有阻止本次收敛。它不是不可行性的证明。
本次绕过了上游网络，并采用表示变量专属Adam和LR；没有单独证明Encoder或Decoder哪一个有错，
也没有验证Face、完整AE/VAE或多mesh共同重建。后续可以使用成功表示作为上游学习的明确参照。

从first-perfect及末尾checkpoint重新载入原始表示，按原评分重新forward；另从GT faces独立重建Edge标签。
核验结果见run/saved_checkpoint_verification.json，确认全部322,806对的FP/FN均为0。

文件：run/updates.jsonl含step0及全部2000步的更新后全量指标；各checkpoint带表示、Adam、全部pair标签和logits。
若出现严格成功，first-perfect.pt/npz保存首次成功状态。best.pt仅按FP+FN最小保存，不能自动视作成功。

最初启动在创建Adam时遇到服务器ONNX/protobuf导入兼容错误，尚无optimizer update。
只为本进程指定PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python后重新启动；没有安装或修改服务器依赖。
失败启动记录单独保存在setup_failed_before_update，不属于这2000步训练。
'''
(root/'REPORT.md').write_text(report)
print(report)
