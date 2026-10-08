"""Read-only experiment snapshot; writes only a new publication directory, no GPU."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tarfile
import time

DEST = Path('/guohaoran/nexus_fast_track/diagnostics/topology_flow_export_20261008')
ROOTS = {
    'candidates': Path('/guohaoran/nexus_fast_track/diagnostics/topology_flow_candidates_20261007'),
    'c0': Path('/guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_continue_20261006'),
    'initial': Path('/ssdwork/guohaoran/nexus_fast_track/diagnostics/topology_flow_user50_20261002'),
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for b in iter(lambda: stream.read(16 << 20), b''): h.update(b)
    return h.hexdigest()


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')


def main():
    os.nice(10)
    DEST.mkdir(exist_ok=False)
    inventory = dict(snapshot_unix=time.time(), hostname=socket.gethostname(),
        collection='Read-only source collection; no training, no evaluation, no GPU allocation',
        gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name','--format=csv,noheader'],text=True),
        sources={k:str(v) for k,v in ROOTS.items()}, files=[], links=[], exclusions=[], checkpoints=[])
    expected = {}
    for root in ROOTS.values():
        for f in root.glob('**/checkpoint-manifest.json'):
            for e in json.loads(f.read_text()).get('checkpoints',[]):
                if e.get('retained',True): expected[str(Path(e['path']).resolve())]=e
    groups = {k:[] for k in ('candidates','c0','fixed50','initial_history')}
    checkpoints=[]
    for label,root in ROOTS.items():
        for p in sorted(root.rglob('*')):
            rel=p.relative_to(root)
            if '__pycache__' in rel.parts: continue
            if p.is_symlink():
                inventory['links'].append(dict(source=str(p),target=os.readlink(p),resolved=str(p.resolve())))
                continue
            if not p.is_file(): continue
            if p.suffix in ('.pt','.pth','.ckpt'):
                checkpoints.append(p); continue
            if p.suffix in ('.lock','.pid','.tmp','.uploadtmp','.pyc'):
                inventory['exclusions'].append(dict(source=str(p),reason='transient operational file',bytes=p.stat().st_size));continue
            group=label if label!='initial' else ('fixed50' if rel.parts[0] in ('cache','inputs_verified','vae_baseline') else 'initial_history')
            before=p.stat()
            entry=dict(source=str(p),archive_path=group+'/'+str(rel),bytes=before.st_size,sha256=sha(p),mtime_ns=before.st_mtime_ns)
            after=p.stat()
            if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Source changed during snapshot: '+str(p))
            inventory['files'].append(entry);groups[group].append((p,entry))
    save(DEST/'SOURCE_MANIFEST.json',inventory)
    for group,files in groups.items():
        out=DEST/('TopologyFlow_'+group+'_20261008.tar.gz')
        with tarfile.open(out,'w:gz',compresslevel=1) as tar:
            for p,e in files:
                if p.stat().st_size!=e['bytes'] or p.stat().st_mtime_ns!=e['mtime_ns']:raise RuntimeError('Source changed before archive: '+str(p))
                tar.add(p,arcname=e['archive_path'],recursive=False)
        print('ARCHIVE',group,out.stat().st_size,flush=True)
    for p in checkpoints:
        print('HASH_CHECKPOINT',str(p),flush=True)
        before=p.stat();digest=sha(p);after=p.stat();e=expected.get(str(p.resolve()))
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):raise RuntimeError('Checkpoint changed')
        if e and (e['sha256']!=digest or e['bytes']!=before.st_size):raise RuntimeError('Checkpoint manifest mismatch: '+str(p))
        inventory['checkpoints'].append(dict(path=str(p),bytes=before.st_size,sha256=digest,
            recorded_manifest_matches=bool(e),completed_updates=e.get('completed_updates') if e else None,
            protected=e.get('protected') if e else None,uploaded=False,reason='Large full training states retained on persistent server under prior user authorization',verified_unix=time.time()))
        save(DEST/'SOURCE_MANIFEST.json',inventory)
    save(DEST/'SERVER_WEIGHT_MANIFEST.json',dict(checkpoints=inventory['checkpoints']))
    assets=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(DEST.glob('*.tar.gz'))]
    save(DEST/'REMOTE_ASSET_MANIFEST.json',dict(assets=assets,completed_unix=time.time()))
    print('COLLECTION_COMPLETE',flush=True)


if __name__=='__main__': main()
