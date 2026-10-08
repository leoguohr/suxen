"""Split the frozen checkpoint without changing the source file."""
from pathlib import Path
import hashlib
import json

source=Path('/ssdwork/guohaoran/nexus_fast_track/diagnostics/ownv2_fixed100_vae_hard_continue1000_20260930/run/checkpoints/vae-1000.pt')
root=Path('/tmp/nexus_vae_archive_20261008')
limit=850*1024**2
expected='e89b8078b7abb0ca7b0c2c44f9f20382ec5b742f2aee948f209d642852714d15'
rows=[];total=hashlib.sha256()
with source.open('rb') as f:
    index=0
    while f.tell()<source.stat().st_size:
        index+=1
        p=root/f'vae_step36220.pt.part{index:03d}'
        assert not p.exists(),p
        n=0;h=hashlib.sha256()
        with p.open('wb') as out:
            while n<limit:
                block=f.read(min(8<<20,limit-n))
                if not block:break
                out.write(block);h.update(block);total.update(block);n+=len(block)
        rows.append({'asset':p.name,'bytes':n,'sha256':h.hexdigest()})
        print('PART',p.name,n,flush=True)
assert total.hexdigest()==expected
(root/'CHECKPOINT_PARTS.json').write_text(json.dumps({'source':str(source),'bytes':source.stat().st_size,
    'sha256':expected,'parts':rows,'format':'raw byte split; concatenate in listed order'},indent=2)+'\n')
print('CHECKPOINT_VERIFIED',expected,flush=True)
