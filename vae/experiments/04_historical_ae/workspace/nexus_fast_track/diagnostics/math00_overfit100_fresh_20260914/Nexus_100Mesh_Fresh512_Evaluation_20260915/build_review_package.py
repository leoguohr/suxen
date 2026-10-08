"""Package saved evidence only; no model execution or training."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import zipfile
import statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'Nexus_100Mesh_Fresh512_Evaluation_20260915'
RAW = OUT / 'evidence'
RAW.mkdir(parents=True, exist_ok=True)
for name in ['evaluation_evidence_20260915.tar.gz', 'review_runtime.tar.gz']:
    with tarfile.open(ROOT / name) as t:
        for m in t.getmembers():
            assert not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
            assert m.isfile() or m.isdir()
        t.extractall(RAW)
shutil.copy2(ROOT / 'checkpoint_inventory.json', OUT / 'checkpoint_inventory.json')

def readj(p):
    return json.loads(p.read_text())

def readlines(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]

def savecsv(name, rows):
    with (OUT / name).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

updates = readlines(RAW / 'run/updates.jsonl')
assert [r['update'] for r in updates] == list(range(1, 5001))
data = {r['uid']: r for r in csv.DictReader((RAW / 'overfit100_manifest.csv').open())}
assert len(data) == 100
manifest = readj(RAW / 'run/manifest.json')
checks = []
for path, expected in manifest['source_sha256'].items():
    local = RAW / 'source_archive' / path.split('/nexus_fast_track/')[1]
    actual = hashlib.sha256(local.read_bytes()).hexdigest()
    checks.append(dict(path=str(local.relative_to(OUT)), expected=expected, actual=actual, match=actual == expected))
for path, expected in manifest['entry_sha256'].items():
    actual = hashlib.sha256((RAW / path).read_bytes()).hexdigest()
    checks.append(dict(path='evidence/' + path, expected=expected, actual=actual, match=actual == expected))
assert all(c['match'] for c in checks)
for rec in readj(RAW / 'review_runtime/resolved_modules.json'):
    historical = manifest['source_sha256'].get(rec['remote_path'])
    if historical:
        assert historical == rec['sha256']

flat, summaries = [], []
for sf in sorted((RAW / 'run').glob('eval-summary-epoch*.json')):
    s = readj(sf)
    rows = readlines(RAW / f"run/eval-epoch{s['epoch']:03d}.jsonl")
    assert len(rows) == 100 and {r['uid'] for r in rows} == set(data)
    for r in rows:
        d = data[r['uid']]
        q = dict(epoch=r['epoch'], update=r['updates'], uid=r['uid'],
                 vertices=int(d['vertices']), gt_edges=int(d['gt_edges']), gt_faces=int(d['gt_faces']),
                 participations=r['participations'], edge_soft4=r['parts']['edge'], face_soft4=r['parts']['face'])
        for kind in ['edge', 'face']:
            for key in ['tp', 'fp', 'fn', 'f1']:
                q[kind + '_' + key] = r[kind].get(key)
        q.update(face_complete=r['face']['complete'],
                 missing_gt_face_candidates=r['missing_gt_face_candidates'],
                 covered_gt_faces=r['gt_face_candidates'],
                 face_fn_present_but_negative=r['face']['fn'] - r['missing_gt_face_candidates'],
                 face_fp_outside_training_pool=r['face']['actual_fp_outside_training_pool'],
                 face_fp_lower_bound=r['face']['fp_lower_bound'],
                 edge_perfect=r['edge_perfect'], face_perfect=r['face_perfect'], joint_perfect=r['joint_perfect'])
        for k, v in r['margins'].items():
            q['min_margin_' + k] = v
        flat.append(q)
    assert sum(r['joint_perfect'] for r in rows) == s['joint_perfect']
    for kind in ['edge', 'face']:
        for metric in ['fp', 'fn']:
            vals = [r[kind][metric] for r in rows]
            expected = sum(vals) if all(v is not None for v in vals) else None
            assert expected == s['total_' + kind + '_' + metric]
    summaries.append(dict(epoch=s['epoch'], update=s['updates'],
        edge_soft4_mean=statistics.mean(r['parts']['edge'] for r in rows),
        face_soft4_mean=statistics.mean(r['parts']['face'] for r in rows),
        edge_perfect=s['edge_perfect'], face_perfect=s['face_perfect'], joint_perfect=s['joint_perfect'],
        edge_fp=s['total_edge_fp'], edge_fn=s['total_edge_fn'], face_fp=s['total_face_fp'], face_fn=s['total_face_fn'],
        face_incomplete=len(s['face_incomplete']),
        missing_gt_face_candidates=sum(r['missing_gt_face_candidates'] for r in rows),
        face_fp_outside_training_pool=sum(r['face']['actual_fp_outside_training_pool'] for r in rows) if not s['face_incomplete'] else None))
savecsv('per_mesh_all_evaluations.csv', flat)
savecsv('per_mesh_final.csv', [r for r in flat if r['epoch'] == 200])
savecsv('evaluation_trend.csv', summaries)

participation = {u: 0 for u in data}
training_rows = []
for r in updates:
    assert len(r['meshes']) == 4
    for m in r['meshes']:
        participation[m['uid']] += 1
        assert m['participation'] == participation[m['uid']]
        training_rows.append(dict(update=r['update'], epoch=r['epoch'], uid=m['uid'],
            participation=m['participation'], edge_soft4=m['parts']['edge'], face_soft4=m['parts']['face'],
            weighted_loss=m['weighted_loss'], mu_rms=m['mu_rms'], hidden_rms=m['hidden_rms'],
            edge_rms=m['edge_rms'], face_rms=m['face_rms']))
assert set(participation.values()) == {200}
savecsv('per_mesh_training_trace.csv', training_rows)
assert readj(RAW / 'runner_exit.json')['train_exit'] == 0

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
xs = [s['update'] for s in summaries]
for key, label in [('edge_soft4_mean','Edge'),('face_soft4_mean','Face (training pool)')]:
    ax[0,0].plot(xs, [s[key] for s in summaries], 'o-', label=label)
ax[0,0].set(title='Same 100 meshes: mean evaluation Soft4', ylabel='Loss')
for metric, label in [('edge_fp','Edge FP'), ('edge_fn','Edge FN'), ('face_fp','Face FP'), ('face_fn','Face FN')]:
    ys=[s[metric] if s[metric] is not None else float('nan') for s in summaries]
    ax[0,1].plot(xs,ys,'o-',label=label)
ax[0,1].set(yscale='symlog', title='Actual reconstruction: total errors', ylabel='Count (symlog)')
ax[1,0].plot(xs,[s['joint_perfect'] for s in summaries],'o-',label='Edge + actual Face perfect')
ax[1,0].set(ylim=(-1,101), title='Strict pass at same checkpoint', ylabel='Meshes / 100')
bins=[]
for i in range(0,len(updates),100):
    w=updates[i:i+100]
    bins.append((w[-1]['update'], statistics.mean(x['objective'] for x in w)))
ax[1,1].plot(*zip(*bins),label='100-update mean')
ax[1,1].set(title='Training objective (changing mesh batches)', ylabel='Loss')
for a in ax.flat:
    a.set_xlabel('Optimizer updates');a.grid(alpha=.25);a.legend(fontsize=8)
fig.suptitle('100-mesh fresh512, mu-only | completed 5000 updates; strict pass 2/100')
fig.savefig(OUT/'training_evaluation.png',dpi=160)
plt.close(fig)

last=summaries[-1]
passed=[r['uid'] for r in flat if r['epoch']==200 and r['joint_perfect']]
report=f'''# 100-mesh fresh512：完整评估包

本包取自已完成实验，不新增训练或模型前向。优先阅读本文件、training_evaluation.png、evaluation_trend.csv 和 per_mesh_final.csv。

## 完成状态与结果

- 5000个实际optimizer更新，200epochs；每条完整mesh参与200次；退出码0，按预算停止。
- 末尾严格成功：2/100，UID：{', '.join(passed)}。
- 末尾 Edge FP/FN：{last['edge_fp']}/{last['edge_fn']}；实际 Face FP/FN：{last['face_fp']}/{last['face_fn']}。
- 最终100条Face枚举全部完成。epoch0部分枚举超时，FP/F1为null；图中不将其当作0。
- 后半程错误总体减少，但仍有大量多余边、面；严格成功的UID并非在各检查点完全相同。不能宣布100条overfit成功，也不能凭有限预算失败宣布容量不足。

## 执行口径

- 同一个随机初始化模型：latent512、decoder hidden1024、Edge/Face评分各32维。μ重建，sampling关闭，KL=0；logvar参数冻结，四组fresh Adam。
- math00：Encoder和Decoder显式FP32 MATH、确定性Graph正反向、TF32/autocast关闭。manifest.args.precision的历史字符串不是实际backend判据；以manifest.backend、runtime安装和review_runtime源码为准。
- 每微批1条完整mesh；累积4条，各自(Edge Soft4+Face Soft4)/4，统一clip=1，然后一次Adam更新。每epoch100条无重复遍历，因此25更新/epoch。
- fully-differentiable Soft4：membership参与反传；每类软权重加权BCE/对应权重总量（epsilon=1e-8），四组固定平均。Edge全pair；Face固定positive+mixed池。公式及chunk归约见effective_functions.py.txt、loss_changes_exact_runtime.txt和common.py。
- E/μ LR第1步1e-6，第100步1e-5；D/head第1步1e-5，第100步1e-4，之后保持。无旧权重或Adam迁移；旧模型只用于离线Face负例池生成。
- 全量验收暂停更新，在同一checkpoint依次前向100条，不是packed100。Face来自当次预测Edge图；缺边导致未进入候选的GT Face仍计FN。阈值0。

## 材料索引

| 路径 | 内容 |
|---|---|
| training_evaluation.png | loss、实际错误、严格成功数曲线 |
| evaluation_trend.csv | 7轮全量验收汇总，分别记录训练pool loss与实际重建错误 |
| per_mesh_all_evaluations.csv | 700条逐mesh评估：规模、参与次数、loss、TP/FP/FN/F1、候选覆盖、margin与完整性 |
| per_mesh_final.csv | 末尾100条的完整汇总 |
| per_mesh_training_trace.csv | 20000条训练参与记录；变化batch上的训练loss不可当成固定集合的同口径趋势 |
| evidence/run/updates.jsonl | 全部5000条原始更新日志：四组梯度、clip、LR、实际位移、表示尺度 |
| evidence/run/eval-*.jsonl、eval-summary-*.json | 原始全量验收，不改变null或下界定义 |
| evidence/run/epoch-order-*.json | 每epoch固定保存的实际UID顺序 |
| evidence/run/manifest.json、complete.json、status.json | 运行配置、组定义、源文件哈希、预算完成证据 |
| evidence/runner_exit.json、train-console.log | 正常退出证据与完整控制台日志 |
| evidence/overfit100_manifest.csv、data_manifest.csv | 100条数据路径、规模、GT数目、哈希 |
| evidence/selection.json、excluded.csv、data_validation.json、pool_provenance.json | 预先选样规则、排除原因、完整数据/候选核验与池来源 |
| evidence/preflight/、preflight-console.log | 100条零更新资源预检、梯度累积核验 |
| evidence/train.py、runtime.py、evaluate.py、prepare.py | 本轮实际训练、数值路径、评估与pool准备入口 |
| evidence/source_archive/ | 与本次manifest记录逐项SHA核对通过的历史源文件快照 |
| evidence/review_runtime/ | 仅import运行时后解析出的依赖源码与有效函数；补充动态加载的代码依赖 |
| evidence/loss_changes_exact_runtime.txt、sampling_forward.py、graph_backward.py、C_graph_only/ | 实际动态替换公式与计算实现 |
| evidence/USER_PROTOCOL.md、README.md | 原协议与启动阶段说明；README是历史启动说明，当前完成状态看本文件和complete.json |
| checkpoint_inventory.json | 服务器checkpoint路径、大小及已记录的验收SHA |
| verification.json、SHA256SUMS.txt | 日志/计数/源文件一致性核验，包内逐文件哈希 |

## 范围与证据边界

这是用于审阅训练结果、目标函数及实际执行方式的评估材料包，不是离线重新运行推理的模型分发包。
大checkpoint（末尾含Adam约1.3GB）、原mesh二进制与177MB训练pool保留服务器，本包提供路径、已有数据哈希、生成规则和checkpoint索引；没有假装打入权重。
本轮原日志未保存所有候选逐条logit/ID、每顶点embedding或完整梯度张量，本包不凭空补造。margin为原日志中的各组最小值。
额外读取runtime只做import，不实例化模型、前向、反向或optimizer更新。训练时已有source/entry哈希全部重新核对；额外依赖的当前快照有独立SHA，但不能据此假装它们都有训练启动时的哈希。
原始记录保持原样，汇总CSV由build_review_package.py生成。CSV中空单元格对应原始null，不能当0。

服务器实验根目录：/guohaoran/nexus_fast_track/diagnostics/math00_overfit100_fresh_20260914
'''
(OUT/'START_HERE_评估说明.md').write_text(report)
verification=dict(updates=5000, training_participations=20000, each_mesh_participations=200,
    full_evaluation_rows=len(flat), evaluations=len(summaries), recorded_source_hash_checks=checks,
    evaluation_aggregate_checks_passed=True, train_exit=0, final_face_complete=100,
    final_joint_perfect=last['joint_perfect'], generated_from_saved_logs_only=True)
(OUT/'verification.json').write_text(json.dumps(verification,indent=2))
shutil.copy2(__file__,OUT/'build_review_package.py')
files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.name!='SHA256SUMS.txt')
(OUT/'SHA256SUMS.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(OUT))+'\n' for p in files))
dest=Path('/Users/luthier/Downloads')/(OUT.name+'.zip')
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():z.write(p,arcname=str(p.relative_to(OUT.parent)))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
print(json.dumps(dict(zip=str(dest), bytes=dest.stat().st_size, files=len(files)+1,
    sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2))
