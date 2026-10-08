"""Read-only source collection; writes only the task-specific /tmp directory."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import socket
import zipfile
from build_local import inspect

ROOT=Path('/tmp/nexus_vae_archive_20261008')
BASE=Path('/ssdwork/guohaoran/nexus_fast_track/diagnostics')
NAMES=['ownv2_fixed100_vae_small_noise_20260929',
       'ownv2_fixed100_vae_negative_pair_20260930',
       'ownv2_fixed100_vae_hard_continue1000_20260930',
       'cad50_fourier_graph_continue10000_20261005']


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()


def save(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def main():
    ROOT.mkdir(exist_ok=True)
    all_files=[];excluded=[];assets=[]
    for name in NAMES:
        source=BASE/name
        assert source.is_dir(),source
        rows=[]
        target=ROOT/(name+'.zip')
        assert not target.exists(),target
        with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
            for p in sorted(source.rglob('*')):
                if not p.is_file():continue
                rel=str(p.relative_to(source));size=p.stat().st_size
                if p.is_symlink() or '__pycache__' in p.parts or p.suffix in ['.pt','.pyc','.lock'] or p.name=='network-output.npz':
                    excluded.append({'source':str(p),'bytes':size,'reason':'historical_checkpoint_or_redundant_hidden_or_cache',
                                     'sha256':None,'hash_status':'not_rehashed_in_this_collection'})
                    continue
                data=p.read_bytes();findings,_=inspect(data,str(p))
                if findings:
                    excluded.append({'source':str(p),'bytes':size,'reason':'credential_pattern_quarantine','findings':findings});continue
                z.writestr(name+'/'+rel,data)
                rows.append({'source':str(p),'path':name+'/'+rel,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
            z.writestr('FILE_MANIFEST.json',json.dumps(rows,indent=2)+'\n')
        assets.append({'asset':target.name,'bytes':target.stat().st_size,'sha256':digest(target),'files':len(rows)})
        all_files.extend(rows)
        print('ARCHIVE',assets[-1],flush=True)
    source=BASE/NAMES[2]
    data=json.loads((source/'run/data_manifest.json').read_text())
    target=ROOT/'fixed100_original_data.zip';rows=[]
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for row in data['records']:
            for kind in ['mesh','topology']:
                p=Path(row[kind+'_path'])
                if not p.exists():p=Path('/ssdwork'+str(p))
                assert digest(p)==row[kind+'_sha256'],p
                rel='data/'+str(p).split('/nexus_fast_track/',1)[1]
                z.write(p,rel)
                rows.append({'source':str(p),'original_runtime_path':row[kind+'_path'],'path':rel,
                             'bytes':p.stat().st_size,'sha256':row[kind+'_sha256'],'uid':row['uid'],'kind':kind})
        selection=Path(data['source'])
        if not selection.exists():selection=Path('/ssdwork'+str(selection))
        for filename,expected in [('selection.json',data['selection_sha256']),('overfit100_manifest.csv',data['manifest_sha256'])]:
            p=selection/filename;assert digest(p)==expected
            z.write(p,'selection/'+filename)
            rows.append({'source':str(p),'path':'selection/'+filename,'bytes':p.stat().st_size,'sha256':expected})
        z.writestr('data_manifest.json',json.dumps(data,indent=2)+'\n')
        z.writestr('FILE_MANIFEST.json',json.dumps(rows,indent=2)+'\n')
    assets.append({'asset':target.name,'bytes':target.stat().st_size,'sha256':digest(target),'files':len(rows)})
    all_files.extend(rows)
    checkpoint=source/'run/checkpoints/vae-1000.pt'
    h=digest(checkpoint)
    assert h=='e89b8078b7abb0ca7b0c2c44f9f20382ec5b742f2aee948f209d642852714d15'
    environment={}
    for name in ['torch','numpy','scipy','transformers']:
        try:environment[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    result={'host':socket.gethostname(),'environment':environment,'assets':assets,
            'final_checkpoint':{'path':str(checkpoint),'bytes':checkpoint.stat().st_size,'sha256':h},
            'included_source_files':len(all_files),'excluded':excluded,
            'operations':'read source files, hash and archive on CPU; no training or GPU evaluation',
            'scope':'all current VAE code/results plus latest CAD50 factorial; final VAE checkpoint transferred separately'}
    save(ROOT/'SERVER_COLLECTION.json',result)
    save(ROOT/'SERVER_FILE_MANIFEST.json',all_files)
    print('COMPLETE',len(all_files),len(excluded),flush=True)


if __name__=='__main__':main()
