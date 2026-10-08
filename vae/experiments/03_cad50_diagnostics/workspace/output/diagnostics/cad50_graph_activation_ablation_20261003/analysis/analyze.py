"""Read-only saved-evidence analysis; no model forward or optimizer updates."""
import csv
import io
import json
from pathlib import Path
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ARMS = ['V2_control','Graph_LN_pre','Graph_SiLU','Encoder_ReLU']
LABELS = ['完整V2','Graph LN前移','Graph SiLU','Encoder ReLU']
COLORS = ['#2563eb','#d97706','#059669','#9333ea']

with zipfile.ZipFile(ROOT/'cad50_results_and_speed_compact.zip') as archive:
    comparison = json.loads(archive.read('repro_outputs/comparison.json'))
    rows = list(csv.DictReader(io.StringIO(archive.read('repro_outputs/per_mesh.csv').decode())))
    config = json.loads(archive.read('runs/V2_control/config.json'))
gt = {r['uid']:r for r in config['data']['records']}
result = dict(scope='CAD50 mu AE; one common random state; four arms x 2000 five-mesh updates', arms={})
sets = {a:{r['uid'] for r in rows if r['arm']==a and r['step']=='2000' and r['joint_strict']=='True'} for a in ARMS}
for arm in ARMS:
    final = [r for r in rows if r['arm']==arm and r['step']=='2000']
    groups = {}
    for label, predicate in [('8_vertices',lambda n:n==8),('12_to_16',lambda n:12<=n<=16),('66_to_274',lambda n:66<=n<=274)]:
        chosen = [r for r in final if predicate(int(r['vertices']))]
        metrics = {}
        for task, truthfield in [('edge','edges'),('face','faces')]:
            truth = sum(gt[r['uid']][truthfield] for r in chosen)
            fp = sum(int(r[task+'_fp']) for r in chosen)
            fn = sum(int(r[task+'_fn']) for r in chosen)
            tp = truth-fn
            metrics[task] = dict(gt=truth,tp=tp,fp=fp,fn=fn,precision=tp/max(tp+fp,1),
                                recall=tp/max(truth,1),f1=2*tp/max(2*tp+fp+fn,1))
        groups[label] = dict(meshes=len(chosen),joint_strict=sum(r['joint_strict']=='True' for r in chosen),**metrics)
    missing = sum(int(r['face_fn_missing']) for r in final)
    present = sum(int(r['face_fn_present']) for r in final)
    result['arms'][arm] = dict(groups=groups,missing_candidate_fraction=missing/(missing+present),
        retained_vs_control=sorted(sets[arm]&sets['V2_control']),
        added_vs_control=sorted(sets[arm]-sets['V2_control']),lost_vs_control=sorted(sets['V2_control']-sets[arm]))
(OUT/'group_analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')

font = Path('/System/Library/Fonts/STHeiti Medium.ttc')
fontManager.addfont(font)
plt.rcParams.update({'font.family':FontProperties(fname=font).get_name(),'axes.unicode_minus':False,
                     'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig, axes = plt.subplots(2,2,figsize=(12,8))
for arm,label,color in zip(ARMS,LABELS,COLORS):
    evals = comparison['arms'][arm]['evaluations']
    steps = [e['step'] for e in evals]
    axes[0,0].plot(steps,[100*e['counts']['edge']['micro_f1'] for e in evals],marker='o',color=color,label=label)
    axes[0,1].plot(steps,[100*e['counts']['face']['micro_f1'] for e in evals],marker='o',color=color)
    axes[1,0].plot(steps,[e['joint_strict'] for e in evals],marker='o',color=color)
    recall = []
    for step in steps:
        selected=[r for r in rows if r['arm']==arm and int(r['step'])==step and int(r['vertices'])>=66]
        truth=sum(gt[r['uid']]['edges'] for r in selected)
        recall.append(100*(truth-sum(int(r['edge_fn']) for r in selected))/truth)
    axes[1,1].plot(steps,recall,marker='o',color=color)
titles = ['全部50条：Edge micro-F1','全部50条：实际Face micro-F1',
          '同checkpoint联合严格成功数','16条较大CAD：真实Edge召回率']
ylabels=['F1 (%)','F1 (%)','成功mesh数 / 50','Recall (%)']
for ax,title,ylabel in zip(axes.flat,titles,ylabels):
    ax.set(title=title,xlabel='有效optimizer更新',ylabel=ylabel,xticks=[0,500,1000,1500,2000])
    ax.grid(alpha=.2);ax.set_ylim(bottom=0)
axes[1,0].axhline(24,color='#64748b',ls='--',lw=1)
axes[1,0].text(25,24.7,'8顶点样本共24条',fontsize=10,color='#475569')
axes[1,0].set_ylim(0,29)
fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,.95),ncol=4,frameon=False)
fig.suptitle('CAD50同起点对照：小mesh严格成功与大mesh真边恢复',fontsize=14,y=.99)
fig.tight_layout(rect=[0,0,1,.90])
fig.savefig(OUT/'training_comparison.png',dpi=170)
plt.close(fig)
print('SAVED',OUT/'training_comparison.png')
