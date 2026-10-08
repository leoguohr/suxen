"""Review archive plus optional complete saved tensors; never includes credentials."""
import hashlib,json,zipfile
from pathlib import Path
R=Path(__file__).resolve().parent
S=R.parent/'decoder_last2_joint_fixed100_20260920'
P=R.parent/'decoder_last2_gradient_groups_20260920'
d=json.loads((R/'result.json').read_text())
audit=json.loads((R/'independent_audit.json').read_text())
assert len(audit['points'])==11 and audit['complete_face_enumerations']==1100
points=sorted(d['points'],key=lambda p:p['lambda'])
labels=dict(success_edge='原成功71条 Edge',success_face='原成功71条 Face',failed_edge='原失败29条 Edge',failed_face='原失败29条 Face')
lines=['# 保存的Adam候选方向：FP32有限位移验证','',
 '11个独立参数点与1100条完整mesh验收已完成。主线optimizer更新0次，本轮没有新建optimizer、没有backward、没有训练。', '',
 '起点为两块联合分支末尾（Joint累计1000次；原末端Adam step1500，第14块Adam step500）。'
 '复用上一诊断保存的实际FP32 Adam候选位移，未重新执行Adam。'
 '固定共同父的成功71条/失败29条分组，不按照本轮72条成功重新分组。', '',
 'θλ=FP32(θ0+λΔθA)，λ=0, ±1/16, ±1/8, ±1/4, ±1/2, ±1。'
 '每一点均由原参数独立构造，保存FP32 state并重载后再评估。'
 '预测使用重载参数与原参数之差（FP64相减与点积归约），不是名义λ直接乘预测值。', '',
 '## 结论', '',
 '本轮确认了上一轮候选方向的局部实际取舍：五个正向点全部提高困难Edge目标、降低其余三部分；五个反向点全部反转。'
 '在最小相邻正向点1/16、1/8，困难Edge实测ΔL/P分别约1.0100、1.0166；反向对应约0.9901、0.9845。'
 '±1/16的困难Edge中心割线相对残差约0.0060%，四部分的中心割线相对残差均不超过0.314%。'
 '这把当前点的梯度预测与可解析的实际位移联系起来，并未证明所有连续步长或其他工作点都具有同一关系。', '',
 '完整正向步长λ=1时，总训练loss从0.068589706降到0.068047561，但困难Edge的ΔL=+1.07944e-5。'
 'Edge FP增加1274（102302→103576），Edge FN保持1；实际Face FP增加1135（5635→6770），FN减少158（338→180）。'
 '联合严格成功仍为同一批72/100。实际Face总FP+FN从5973增到6950，所以不能把Face训练loss下降称为实际Face整体重建改善。', '',
 '完整步长的Face一阶外推明显失真：成功Face的实测ΔL/P约0.190，困难Face约0.294。'
 '半步的困难Face目标反而比完整步更低，这显示该方向的大步长非线性；本轮并不据此选择新LR。'
 '现有证据支持“当前合成方向存在可观测的局部目标取舍”，不支持直接给出新权重、判定容量不足或把反向位移当作新训练方案。', '',
 '## 四部分目标的实际变化', '',
 '下表每项都是相对原点的Δloss；成功/失败组内各mesh仍乘1/100，不按71或29重新归一化。'
 'Edge为全pair目标；Face为原固定扩充pool上的fully-diff Soft4，不能解释为实际Face候选全集的loss。', '',
 '| λ | 成功Edge ΔL | 成功Face ΔL | 困难Edge ΔL | 困难Face ΔL | 总ΔL |',
 '|---:|---:|---:|---:|---:|---:|']
zero=next(p for p in points if p['lambda']==0)
for p in points:
    x=p['comparisons']
    lines.append('| '+str(p['lambda'])+' | '+' | '.join(f"{x[k]['delta_loss']:+.9g}" for k in d['manifest']['parts'])+f" | {p['total_loss']-zero['total_loss']:+.9g} |")
lines+=['','## 预测与实际：困难Edge','',
 '| λ | 实际相对参数位移 | g·实际Δθ | 实际ΔL | ΔL / P |',
 '|---:|---:|---:|---:|---:|']
for p in points:
    x=p['comparisons']['failed_edge']
    lines.append(f"| {p['lambda']} | {p['geometry']['relative_norm']:.9g} | {x['prediction']:+.9g} | {x['delta_loss']:+.9g} | {x['ratio'] if x['ratio'] is not None else 'NA'} |")
lines+=['','## 正负位移的中心割线核验','',
 '使用g·(实际Δθ+−实际Δθ−)作为预测分母；不是假定FP32舍入后的两侧严格对称。', '',
 '| λ绝对值 | 分量 | 实际割线 / 预测割线 | 相对残差 |', '|---:|---|---:|---:|']
for s in d['centered_secants']:
    lines.append(f"| {s['size']} | {labels[s['part']]} | {s['ratio']:.8g} | {s['relative_error']:.8g} |")
lines+=['','## 完整实际重建','',
 '| λ | Edge FP/FN | 实际Face FP/FN | 联合成功 | 原72条丢失 | 新增成功 |',
 '|---:|---:|---:|---:|---:|---:|']
for p in points:
    a=p['summary']['all']; e=a['edge']; f=a['face']
    lines.append(f"| {p['lambda']} | {e['fp']}/{e['fn']} | {f['fp']}/{f['fn']} | {a['joint_perfect']}/100 | {len(p['lost_source72'])} | {len(p['gained_over_source72'])} |")
lines+=['','实际Face来自各参数点自己的预测Edge图，GT未进入该候选集仍计FN。'
 '1100次Face枚举均完整，没有使用训练pool代替实际验收。'
 '每条的候选覆盖、池外Face FP、margin及身份保留见actual_per_mesh.csv和points/*/meshes.jsonl。','',
 '## 核验与边界','',
 '- 全部100条、全部11点：重复forward的hidden、训练logits和scalar bitwise一致。',
 '- 原点全部loss、Edge/Face计数、margin与保存的末尾记录逐项一致。',
 '- 每点另外核验最小和最大mesh的真实完整网络与冻结缓存路径，hidden/Edge/Face embedding逐位一致。其余冻结缓存沿用已验证的100条边界，输入与pool哈希逐项核对。',
 '- CPU独立用FP64表达式再舍入FP32，复核11份落盘state，重新计算实际位移、四项点积、1100条计数和组归约。没有声称独立重跑全部网络。',
 '- 源checkpoint、完整模型、候选向量文件SHA不变；实验内存模型最后还原θ0，全部grad=None，RNG不变。',
 '- 启动预检曾遇到state_dict中非tensor元数据不能做tensor_hash的问题；仅修正哈希过滤，尚未评估任何位移点或执行optimizer。预检日志完整保留。',
 '- 这些点只验证一个工作点、一个保存方向。离散点的趋势不能证明整个连续区间或所有后续Adam更新；每个mesh、每个候选也不必与组目标同向。',
 '- 本轮不据此改loss权重、冻结成功样本、重置Adam、扩展架构或启动续训。','',
 '## 文件索引','',
 '- result.json：11点汇总、实际位移/预测、逐组计数、成功UID变化、中心割线。',
 '- loss_displacement.csv / centered_secants.csv：四目标数值比较。',
 '- actual_counts.csv / actual_per_mesh.csv：全部结构计数与margin。',
 '- points/*/result.json、meshes.jsonl：原始逐点与逐mesh记录。',
 '- manifest.json / partition.json / independent_audit.json / complete.json：协议与核验。',
 '- run.py：本次实际无更新入口；runtime/core/effective_code与runtime_dependencies：实际公式和运行时替换来源。历史core含训练函数，但本次未调用。',
 '- source/：原点记录、cache/pool哈希、前一轮四梯度诊断与边界核验。',
 '- tensors ZIP：11份实际落盘FP32 tail state、四份梯度与保存Adam位移、原开放模块/Adam checkpoint；主线大模型、缓存和数据未打入，路径及哈希见EXCLUDED_FILES.json。','']
(R/'REPORT.md').write_text('\n'.join(lines))
# Standard static research plots, linked to the exact tabulated measurements.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for ax,part in zip(axes.flat,d['manifest']['parts']):
    xx=[p['lambda'] for p in points]
    ax.plot(xx,[p['comparisons'][part]['delta_loss'] for p in points],'o-',label='measured delta loss')
    ax.plot(xx,[p['comparisons'][part]['prediction'] for p in points],'x--',label='g dot actual delta')
    ax.axhline(0,color='gray',lw=.6); ax.axvline(0,color='gray',lw=.6)
    ax.set(title=part,xlabel='lambda',ylabel='loss change'); ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0)); ax.legend(fontsize=8)
fig.savefig(R/'finite_displacement.png',dpi=180);plt.close(fig)
excluded=dict(full_model=dict(path=d['manifest']['source_full_model'],sha256=d['manifest']['source_full_sha256']),
    frozen_cache=dict(path=str(S/'cache'),manifest='source/cache_manifest.json'),
    face_pools=dict(path=str(S/'augmented_pools'),hashes='source/source_manifest.json'),
    note='Review excludes large tensor files. Companion tensors ZIP includes all 11 saved FP32 states, complete saved gradients/direction and source trainable checkpoint. Full upstream model, dataset and cached tensors remain on server.')
(R/'EXCLUDED_FILES.json').write_text(json.dumps(excluded,indent=2)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()
entries=[]
for p in R.iterdir():
    if p.is_file() and not p.is_symlink() and p.suffix in ['.py','.json','.md','.csv','.png','.log','.txt'] and p.name!='package_manifest.json': entries.append((p,p.name))
for folder in ['points','effective_code','preflight0']:
    for p in (R/folder).rglob('*'):
        if p.is_file() and p.suffix!='.pt' and '__pycache__' not in p.parts: entries.append((p,str(p.relative_to(R))))
for p in (S/'runtime_dependencies').rglob('*'):
    if p.is_file() and '__pycache__' not in p.parts: entries.append((p,str(p.relative_to(S))))
for src,target in [
 ('run/checkpoint-new0500-tail1500.json','current_checkpoint_metrics.json'),
 ('run/final_real_network.json','current_actual100.json'),
 ('parent_actual_baseline.json','partition_parent_actual.json'),
 ('cache/manifest.json','cache_manifest.json'),
 ('source_manifest.json','source_manifest.json'),
 ('cache_gradient_verification.json','all100_cache_gradient_verification.json')]: entries.append((S/src,'source/'+target))
for name in ['result.json','partition.json','independent_audit.json']: entries.append((P/name,'source/previous_diagnostic_'+name))
for name in ['READY.json','overfit100_manifest.csv','data_manifest.csv','selection.json','pool_provenance.json','construction_args.json']:
    entries.append((R/name,'source/'+name))
tensor_entries=[(p,'points/'+str(p.relative_to(R/'points'))) for p in (R/'points').glob('*/tail-fp32.pt')]
tensor_entries += [(P/'gradient_vectors.pt','gradient_vectors.pt'),
    (S/'run/checkpoint-new0500-tail1500.pt','current_trainable_checkpoint.pt'),
    (R/'manifest.json','manifest.json'),(R/'independent_audit.json','independent_audit.json')]
def package(name,files):
    dest=R/name; hashes=[]
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=4) as z:
        for p,n in sorted(files,key=lambda x:x[1]):
            digest=sha(p);z.write(p,n);hashes.append(dict(path=n,bytes=p.stat().st_size,sha256=digest))
        z.writestr('FILE_MANIFEST.json',json.dumps(hashes,indent=2)+'\n')
    return dict(path=str(dest),bytes=dest.stat().st_size,sha256=sha(dest),files=len(hashes))
review=package('Nexus_Adam_FiniteDisplacement_Review_20260921.zip',entries)
print('REVIEW',json.dumps(review),flush=True)
tensors=package('Nexus_Adam_FiniteDisplacement_Tensors_20260921.zip',tensor_entries)
(R/'package_manifest.json').write_text(json.dumps(dict(review=review,tensors=tensors),indent=2)+'\n')
print('PACKAGED',json.dumps(dict(review=review,tensors=tensors)),flush=True)
