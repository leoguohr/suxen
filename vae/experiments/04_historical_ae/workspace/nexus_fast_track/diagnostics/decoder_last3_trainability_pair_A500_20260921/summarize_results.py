"""Read-only CPU acceptance and evidence bundle after both fixed budgets end."""
import csv,hashlib,json,math,zipfile
from pathlib import Path
import numpy as np
import torch as T
T.set_num_threads(1)
R=Path(__file__).resolve().parent;O=R/'repro_outputs';cfg=json.loads((R/'config.json').read_text())
def read(p):return json.loads(p.read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(name,d): (O/name).write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def csvout(name,rows):
    with (O/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
complete=read(R/'complete.json');assert complete['state']=='complete' and complete['total_experiment_updates']==1000
assert not T.cuda.is_initialized()
step_rows=[];full_rows=[];per_uid=[];endpoints={};manifest=[]
for branch,treatment in cfg['branches'].items():
    run=R/branch/'run';trace=[json.loads(s) for s in (run/'step_records.jsonl').read_text().splitlines()];updates=[json.loads(s) for s in (run/'updates.jsonl').read_text().splitlines()]
    assert [r['after_new_updates'] for r in trace]==list(range(501)) and [r['update'] for r in updates]==list(range(1,501))
    original=set(trace[0]['edge_success_uids']);previous=original;ever_lost=set();loss_steps=[]
    for step,row in enumerate(trace):
        got={x['uid'] for x in row['meshes'] if x['fp']==x['fn']==0};assert got==set(row['edge_success_uids']) and len(got)==row['edge_success_count']
        assert row['edge_fp']==sum(x['fp'] for x in row['meshes']) and row['edge_fn']==sum(x['fn'] for x in row['meshes'])
        assert set(row['lost_vs_A500'])==original-got and set(row['lost_vs_previous_update'])==previous-got
        assert len(row['meshes'])==100
        if step:
            assert row['update_that_produced_this_state']==updates[step-1]
            assert updates[step-1]['mesh_count']==100
            expected=dict(decoder_tail=[2000+step],edge_head=[2000+step],face_head=[2000+step],decoder14=[1000+step])
            if treatment:expected['decoder13']=[step]
            assert updates[step-1]['adam_steps']==expected
            assert math.isclose(updates[step-1]['clip_coefficient'],min(1.,1./(updates[step-1]['global_gradient_norm']+1e-6)),rel_tol=1e-12)
        ever_lost.update(original-got)
        if original-got:loss_steps.append(step)
        step_rows.append(dict(branch=branch,new_updates=step,edge_success=len(got),edge_fp=row['edge_fp'],edge_fn=row['edge_fn'],lost_vs_A500_count=len(original-got),lost_vs_previous_count=len(previous-got),actual_delta_norm=updates[step-1]['actual_delta_norm'] if step else 0,actual_relative_delta=updates[step-1]['actual_relative_delta'] if step else 0,global_gradient_norm=updates[step-1]['global_gradient_norm'] if step else 0,clip_coefficient=updates[step-1]['clip_coefficient'] if step else 1,**row['components']))
        previous=got
    for step in cfg['checkpoints']:
        e=read(run/f'actual-new{step:04d}.json');p=run/f'checkpoint-new{step:04d}.pt';digest=sha(p);assert digest==e['checkpoint_sha256']
        cp=T.load(p,map_location='cpu',mmap=True,weights_only=False);assert cp['new_updates']==step and set(cp['per_uid_new_training_participations'].values())=={step}
        for g in cp['optimizer']['param_groups']:
            base=0 if g['name']=='decoder13' else 1000 if g['name']=='decoder14' else 2000
            if step==0 and g['name']=='decoder13':assert not any(i in cp['optimizer']['state'] for i in g['params'])
            else:assert all(int(cp['optimizer']['state'][i]['step'])==base+step for i in g['params'])
        manifest.append(dict(path=str(p),sha256=digest,bytes=p.stat().st_size,kind='partial_weights_optimizer_all_four_rng',branch=branch,new_updates=step))
        successes=[]
        for m in e['meshes']:
            pred=R/m['prediction_path'];assert sha(pred)==m['prediction_sha256']
            with np.load(pred,allow_pickle=False) as q:
                n=len(q['vertices']);pe=q['predicted_edge_ids'];ge=q['gt_edges'];gf=q['gt_faces'];fi=q['actual_face_candidate_ids'];fl=q['actual_face_candidate_logits']
                def ek(x):return x[:,0].astype(np.int64)*n+x[:,1]
                def fk(x):return (x[:,0].astype(np.int64)*n+x[:,1])*n+x[:,2]
                pk=ek(pe);gk=ek(ge);ky=fk(fi);gy=fk(gf);ep=np.isin(pk,gk);fy=np.isin(ky,gy);fpred=fl>0
                etp=int(ep.sum());efp=len(pe)-etp;efn=len(ge)-etp;ftp=int((fy&fpred).sum());ffp=int((~fy&fpred).sum());ffn=len(gf)-ftp
                assert [etp,efp,efn,ftp,ffp,ffn]==[m[k][v] for k in ['edge','face'] for v in ['tp','fp','fn']]
                assert len(np.unique(pk))==len(pk) and len(np.unique(ky))==len(ky) and (q['predicted_edge_logits']>0).all() and np.isfinite(fl).all() and np.isfinite(q['predicted_edge_logits']).all()
                assert (pe[:,0]<pe[:,1]).all() and (fi[:,0]<fi[:,1]).all() and (fi[:,1]<fi[:,2]).all()
                nb=[set() for _ in range(n)]
                for a,b in pe:nb[int(a)].add(int(b))
                count=sum(len(nb[a]&nb[b]) for a,js in enumerate(nb) for b in js)
                assert count==len(fi)==m['face']['scored_candidates']
                for a,b in [(0,1),(0,2),(1,2)]:assert np.isin(fi[:,a].astype(np.int64)*n+fi[:,b],pk).all()
                covered=np.ones(len(gf),bool)
                for a,b in [(0,1),(0,2),(1,2)]:covered &=np.isin(gf[:,a].astype(np.int64)*n+gf[:,b],pk)
                assert int((~covered).sum())==m['missing_gt_face_candidates'] and np.array_equal(covered,q['gt_face_covered'])
            ok=efp==efn==ffp==ffn==0
            if ok:successes.append(m['uid'])
            per_uid.append(dict(branch=branch,new_updates=step,uid=m['uid'],edge_fp=efp,edge_fn=efn,face_fp=ffp,face_fn=ffn,joint_perfect=ok,missing_gt_face_candidates=m['missing_gt_face_candidates']))
        assert set(successes)==set(e['joint_perfect_uids']) and len(successes)==e['summary']['joint_perfect']
        full_rows.append(dict(branch=branch,new_updates=step,edge_success=e['summary']['edge_perfect'],joint_success=len(successes),edge_fp=e['summary']['edge']['fp'],edge_fn=e['summary']['edge']['fn'],face_fp=e['summary']['face']['fp'],face_fn=e['summary']['face']['fn'],lost_A500=len(e['lost_A500']),new_over_A500=len(e['new_over_A500'])))
    final=T.load(run/'model-new0500-inference.pt',map_location='cpu',mmap=True,weights_only=False);cp=T.load(run/'checkpoint-new0500.pt',map_location='cpu',mmap=True,weights_only=False)
    mapping=dict(block='autoencoder.decoder_blocks.15',penultimate='autoencoder.decoder_blocks.14',third='autoencoder.decoder_blocks.13',final_norm='autoencoder.decoder_output_norm',head='autoencoder.edge_embedding',face='autoencoder.face_embedding')
    for k,t in cp['tail'].items():a,b=k.split('.',1);assert T.equal(t,final['model'][mapping[a]+'.'+b])
    parent=T.load(cfg['source_full'],map_location='cpu',mmap=True,weights_only=False)
    for k,t in parent['model'].items():
        if T.is_tensor(t) and k not in cp['trainable_names']:assert T.equal(t,final['model'][k]),k
        elif not T.is_tensor(t):assert t==final['model'][k],k
    del parent
    p=run/'checkpoint-new0001.pt';manifest.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,kind='partial_weights_optimizer_all_four_rng',branch=branch,new_updates=1,full_evaluation=False))
    p=run/'model-new0500-inference.pt';manifest.append(dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size,kind='full_model_inference_weights',branch=branch,new_updates=500))
    e=read(run/'actual-new0500.json');endpoints[branch]={'joint_success':e['summary']['joint_perfect'],'success_uids':e['joint_perfect_uids'],'lost_A500':e['lost_A500'],'new_over_A500':e['new_over_A500'],'summary':e['summary'],'edge_ever_lost_A500_uids':sorted(ever_lost),'edge_loss_steps':loss_steps,'minimum_edge_success':min(x['edge_success_count'] for x in trace),'edge_full_step_coverage':'all states 0..500','face_coverage':'only full checkpoints 0/100/200/300/400/500; no all-step Face claim'}
    del final,cp
assert not T.cuda.is_initialized()
recovery=None
rec=O/'recovery_204'
if (rec/'gate.json').exists():
    gate=read(rec/'gate.json');assert gate['status']=='passed' and gate['optimizer_updates']==0
    assert sha(rec/'latest.pt')==gate['resume_checkpoint_sha256']
    for name in ['updates.jsonl','step_records.jsonl']:
        assert (R/'Control_last2/run'/name).read_bytes().startswith((rec/name).read_bytes())
    records=[read(p) for p in (O/'_runtime/train').glob('*/state.json')]
    old=next(x for x in records if x['run_id']=='20260921T142436Z-4bf71f12')
    new=next(x for x in records if x.get('retry_of')==old['run_id'])
    assert old['status']=='interrupted' and new['status']=='success' and new['attempt']==2
    recovery={'interrupted_control_step':204,'remaining_control_updates':296,'remaining_treatment_updates':500,'replayed_optimizer_updates':0,'prefix_logs_unchanged':True,'zero_update_gate_passed':True,'resume_checkpoint_sha256':gate['resume_checkpoint_sha256'],'old_runtime':old['run_id'],'resumed_runtime':new['run_id']}
    manifest.append(dict(path=str(rec/'latest.pt'),sha256=sha(rec/'latest.pt'),bytes=(rec/'latest.pt').stat().st_size,kind='immutable_interruption_snapshot_partial_optimizer_all_four_rng',branch='Control_last2',new_updates=204))
control=endpoints['Control_last2'];treatment=endpoints['Treatment_last3'];primary=treatment['joint_success']>max(72,control['joint_success']) and not treatment['lost_A500']
acceptance={'cpu_audit_passed':True,'per_branch_optimizer_updates':500,'total_optimizer_updates':1000,'primary_treatment_success':primary,'endpoints':endpoints,'main_goal_100':any(x['joint_success']==100 for x in endpoints.values()),'project_record_over74':any(x['joint_success']>74 for x in endpoints.values()),'new_branch_progress_over72':any(x['joint_success']>72 for x in endpoints.values()),'no_cross_checkpoint_union':True,'new_layer_added':False,'optimizer_steps_in_cpu_audit':0,'recovery':recovery}
save('ACCEPTANCE.json',acceptance);save('CHECKPOINT_MANIFEST.json',{'models':manifest,'source_full':cfg['source_full'],'source_full_sha256':cfg['source_full_sha256'],'source_checkpoint':cfg['source_checkpoint'],'source_checkpoint_sha256':cfg['source_checkpoint_sha256'],'weights_included_in_zip':False})
csvout('edge_every_step.csv',step_rows);csvout('actual_full_checkpoints.csv',full_rows);csvout('actual_per_uid.csv',per_uid)
lines=['# A500最后两块/三块可训练性对照结果','','两支各新增500次全100条累积更新，预算已关闭。CPU逐文件/状态/预测计数验收通过；没有后继训练。','', '| 分支 | 最终联合严格成功 | Edge FP/FN | 实际Face FP/FN | 丢失A500成功UID数 |','|---|---:|---:|---:|---:|']
for name,x in endpoints.items():
    s=x['summary'];lines.append(f"| {name} | {x['joint_success']}/100 | {s['edge']['fp']}/{s['edge']['fn']} | {s['face']['fp']}/{s['face']['fn']} | {len(x['lost_A500'])} |")
lines.extend(['',f'预注册Treatment首要判据：{primary}。分支进展（>72）：{acceptance["new_branch_progress_over72"]}；刷新历史纪录（>74）：{acceptance["project_record_over74"]}；主目标（同一checkpoint100/100）：{acceptance["main_goal_100"]}。','','本实验只检验额外开放已有block13的效用；无论结果如何，都不能单独证明整个架构容量上限或冻结是唯一根因。','', '## 全程Edge证据与Face采样边界',''])
for name,x in endpoints.items():lines.append(f"- {name}：完整覆盖更新后状态0..500；最低Edge成功{x['minimum_edge_success']}/100；曾丢失A500 Edge成功UID：{x['edge_ever_lost_A500_uids']}；发生步骤：{x['edge_loss_steps']}。")
lines.extend(['','完整Face仅在0/100/200/300/400/500验收；没有声称检查点之间Face全程保持。每步参数实际位移/clip/逐UID Edge见step_records.jsonl、updates.jsonl与edge_every_step.csv。','', '## 执行边界','', '新网络层数始终16；Control开放14/15，Treatment多开放13；旧Adam组/四类RNG继承，新增组单独空state；原数据/pool/目标/后端/评分不变。0步对两支全部100条验证完整/缓存前向及梯度逐位一致；父A500预测数组也逐项一致。','', '预检曾因新缓存导出包装中的no_grad与现有math00日志上下文不兼容失败；当时更新0，失败日志保留。仅使新包装恢复原grad模式后重试，未改模型/后端/损失。','', '所有结果以独立checkpoint为单位，不拼UID、不延长预算；源权重和数据只读。精确来源、argv、环境、GPU身份与运行日志保留在repro_outputs和源码manifest中。'])
if recovery:
    lines.extend(['','## 已审计的容器中断与恢复','', '原容器在Control完成204次更新后消失；原runtime的进程已不存在，状态按skill恢复规则标为interrupted，原状态和日志已另存不可变快照。GPU UUID、源码与Python/torch/CUDA/cuDNN/依赖版本保持一致。', '', '只读CPU检查确认latest.pt为204更新，Adam旧组2204/block14组1204；跨容器第200点全部100条预测数组逐项一致，第204点全100条完整/缓存前向及梯度逐位一致。两项恢复校验optimizer更新0。', '', '显式从204接续Control剩余296次，Treatment仍独立从共同A500执行500次。旧日志字节前缀保持不变，最终每支0..500状态/1..500更新连续；没有重跑204次或叠加预算。恢复命令和retry_of谱系见recovery_204与_runtime/train。'])
(O/'SUMMARY.md').write_text('\n'.join(lines)+'\n');(O/'RUN_REPORT.md').write_text('\n'.join(lines)+'\n')
try:
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for name in cfg['branches']:
        rs=[r for r in step_rows if r['branch']==name];es=[r for r in full_rows if r['branch']==name];xs=[r['new_updates'] for r in rs]
        axes[0,0].plot(xs,[r['edge_success'] for r in rs],label=name)
        axes[0,1].plot([r['new_updates'] for r in es],[r['joint_success'] for r in es],marker='o',label=name)
        axes[1,0].plot(xs,[r['edge_fp'] for r in rs],label=name+' FP');axes[1,0].plot(xs,[r['edge_fn'] for r in rs],linestyle='--',label=name+' FN')
        axes[1,1].plot(xs[1:],[r['actual_delta_norm'] for r in rs][1:],label=name)
    for ax in axes[0]:
        ax.axhline(72,color='gray',linestyle=':');ax.axhline(74,color='gray',linestyle='--');ax.set_ylim(0,102)
    axes[0,0].set_title('Edge strict count: every update');axes[0,1].set_title('Joint strict count: scheduled full evaluations')
    axes[1,0].set_title('Edge FP / FN');axes[1,0].set_yscale('symlog',linthresh=1);axes[1,1].set_title('Actual parameter displacement L2')
    for ax in axes.flat:ax.set_xlabel('New optimizer updates');ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(O/'paired_trends.png',dpi=160);plt.close(fig)
except ImportError as e:(O/'plot_unavailable.txt').write_text(str(e))
save('status.json',{'state':'complete','cpu_acceptance':'passed','per_branch_new_updates':500,'total_new_updates':1000,'primary_treatment_success':primary,'main_goal_100':acceptance['main_goal_100'],'follow_on_training_started':False})
p=O/'COMMANDS.md';p.write_text(p.read_text().replace('尚未执行','已执行通过；结果见ACCEPTANCE.json'))
p=O/'COMPARABILITY_REPORT.md';p.write_text(p.read_text().replace('目前完整对照结果未验收。','两支预算与最终CPU验收已完成；结果见SUMMARY.md和ACCEPTANCE.json。'))
with (O/'LOG.md').open('a') as f:f.write('\n- 最终CPU验收通过：两支各500次更新，所有501个状态和6个完整检查点逐文件通过；预算关闭，没有后继训练。\n')
review=R/'Nexus_Last2_vs_Last3_A500_Review.zip';predzip=R/'Nexus_Last2_vs_Last3_A500_Predictions.zip';entries=[]
with zipfile.ZipFile(review,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(R.rglob('*')):
        if not p.is_file() or p.is_symlink() or any(x in p.parts for x in ['.git','cache','preflight_predictions']):continue
        if p.suffix in ['.pt','.npz','.zip','.pyc'] or 'predictions-new' in str(p):continue
        rel=str(p.relative_to(R));z.write(p,rel);entries.append({'path':rel,'bytes':p.stat().st_size,'sha256':sha(p)})
    z.writestr('FILE_MANIFEST.json',json.dumps(entries,indent=2)+'\n')
with zipfile.ZipFile(predzip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in cfg['branches']:
        for p in sorted((R/name/'run').glob('predictions-new*/*.npz')):z.write(p,str(p.relative_to(R)))
for p in [review,predzip]:
    with zipfile.ZipFile(p) as z:assert z.testzip() is None
save('DELIVERY.json',{'archives':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in [review,predzip]],'excluded':'weights, raw datasets, pools, hidden caches; identities retained in CHECKPOINT_MANIFEST/source records'})
print('FINAL ACCEPTANCE',json.dumps({k:v for k,v in acceptance.items() if k!='endpoints'}),flush=True)
