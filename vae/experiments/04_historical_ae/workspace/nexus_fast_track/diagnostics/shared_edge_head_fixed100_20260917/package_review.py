"""Compact review archive; omitted large tensors have paths and hashes in the full archive."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent
full=json.loads((ROOT/'package_verification.json').read_text())
dest=ROOT/'Nexus_SharedEdgeHead_Fixed100_2000Updates_Review_20260917.zip'
with zipfile.ZipFile(full['path']) as src:
    old={name:h for h,name in (line.split('  ',1) for line in src.read('SHA256SUMS.txt').decode().splitlines())}
    omitted={n:dict(sha256=h,server_path=str(ROOT/n),full_archive_entry=n) for n,h in old.items()
        if (n.startswith('cache/nexus_') and n.endswith('.npz')) or n.startswith('run/final_outputs/')}
    keep={n:h for n,h in old.items() if n not in omitted}
    external=json.dumps(dict(full_archive=full,omitted_large_tensors=omitted,
        note='Review contains all 2000 update logs and 2001 all-100 evaluations, shared-head checkpoints, source and verification. Full archive also contains every fixed hidden and all final pair logits.'),indent=2).encode()+b'\n'
    new=dict(keep);new['EXTERNAL_TENSORS.json']=hashlib.sha256(external).hexdigest()
    new['package_review.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for n,h in keep.items():
            b=src.read(n);assert hashlib.sha256(b).hexdigest()==h;z.writestr(n,b)
        z.writestr('EXTERNAL_TENSORS.json',external);z.write(__file__,'package_review.py')
        z.writestr('SHA256SUMS.txt',''.join(f'{h}  {n}\n' for n,h in sorted(new.items())))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for n,h in new.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
record=dict(path=str(dest),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),files_verified=len(new),excluded_binary_files=len(omitted))
(ROOT/'review_package_verification.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True)
