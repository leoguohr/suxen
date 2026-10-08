"""Package all small experiment evidence; retain large complete states on server."""
import csv,hashlib,json,zipfile
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;O=R/'repro_outputs'
assert json.loads((O/'ACCEPTANCE.json').read_text())['passed']
assert (O/'ANNOTATED_README.md').exists()
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
rows=list(csv.DictReader((O/'actual_trend.csv').open()))
fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
for branch,label,color in [('S_soft4_full_control','S: full Soft4 (reused)','#2563eb'),('H_paper_hard4','H: hard four-group BCE','#e76f00')]:
    data=[x for x in rows if x['branch']==branch];steps=[int(x['new_step']) for x in data]
    for ax,key,title in [(axes[0,0],'face_f1','Actual Face micro-F1 (all 50)'),(axes[0,1],'joint','Strict Edge + Face success (of 50)'),(axes[1,0],'edge_f1','Edge micro-F1 (all pairs)'),(axes[1,1],'large_face_f1','Actual Face micro-F1 (16 larger CAD)')]:
        ax.plot(steps,[float(x[key]) for x in data],marker='o',label=label,color=color)
        ax.set_title(title);ax.set_xlabel('New optimizer updates from B2500');ax.set_xticks([0,25,50,75,100]);ax.grid(alpha=.25)
axes[0,0].legend(fontsize=8);axes[0,1].set_yticks([30,31,32,33,34,35])
fig.suptitle('CAD50: identical B2500 model / Adam / RNG; 100-update loss comparison')
fig.savefig(O/'actual_reconstruction_trends.png',dpi=170);plt.close(fig)
paths=[]
for p in R.rglob('*'):
    if not p.is_file() or p.is_symlink():continue
    rel=p.relative_to(R)
    if any(x in ['.git','__pycache__'] for x in rel.parts):continue
    if p.suffix in ['.pt','.zip','.pyc','.tmp','.lock'] or p.name in ['FILE_MANIFEST.json','DELIVERY.json']:continue
    paths.append(p)
manifest=[dict(path=str(p.relative_to(R)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
(O/'FILE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
output=R/'CAD50_Soft4_vs_PaperHard4_B2500_100steps_Evidence.zip'
with zipfile.ZipFile(output,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in paths:z.write(p,p.relative_to(R))
    z.write(O/'FILE_MANIFEST.json','repro_outputs/FILE_MANIFEST.json')
with zipfile.ZipFile(output) as z:
    assert z.testzip() is None
    for row in manifest:
        content=z.read(row['path'])
        assert len(content)==row['bytes'] and hashlib.sha256(content).hexdigest()==row['sha256']
receipt=dict(path=str(output),bytes=output.stat().st_size,sha256=sha(output),
    archive_crc_and_all_entry_sha256_verified=True,files=len(manifest)+1,
    full_model_adam_rng_weights_included=False,weight_manifest='repro_outputs/CHECKPOINT_MANIFEST.json',
    source_control_reused=True,optimizer_updates_in_packaging=0)
(O/'DELIVERY.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
