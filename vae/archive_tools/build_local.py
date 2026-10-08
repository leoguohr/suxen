"""Collect scoped VAE artifacts with source hashes and recursive archive inspection."""
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parents[1]
STAGE = ROOT / 'stage'
TEXT = {'.py','.sh','.md','.txt','.json','.jsonl','.csv','.yaml','.yml','.toml','.log','.diff','.patch','.html','.tex','.expect','.exp'}
PATTERNS = {
 'token': re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{25,}|glpat-[A-Za-z0-9_-]{16,}|sk-[A-Za-z0-9_-]{24,})\b'),
 'private_key': re.compile(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----'),
 'credential_assignment': re.compile(r'''(?i)\b(?:password|passwd|access[_-]?token|api[_-]?key|client[_-]?secret)\b\s*[=:]\s*["'][^"'\r\n]{6,}["']'''),
 'sshpass': re.compile(r'''sshpass\s+-p\s+[^\s]+'''),
 'url_credential': re.compile(r'(?:https?|ssh)://[^\s/:]+:[^\s/@]+@'),
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def inspect(data, name, depth=0):
    findings = []
    members = []
    suffix = Path(name).suffix.lower()
    if suffix in ('.zip', '.npz'):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            assert z.testzip() is None, name
            for item in z.infolist():
                if item.is_dir(): continue
                parts = PurePosixPath(item.filename)
                assert not parts.is_absolute() and '..' not in parts.parts, name
                body = z.read(item)
                inner = name + '!/' + item.filename
                members.append({'member': inner, 'bytes':len(body), 'sha256':sha(body)})
                if Path(item.filename).suffix.lower() in TEXT | {'.zip'}:
                    assert depth < 5
                    f, m = inspect(body, inner, depth+1)
                    findings.extend(f); members.extend(m)
    elif suffix in TEXT or Path(name).name in ('Dockerfile','requirements.txt','SHA256SUMS'):
        try: text = data.decode('utf-8')
        except UnicodeDecodeError: return findings, members
        for kind, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append({'file':name, 'kind':kind, 'line':text.count('\n',0,match.start())+1})
        # Random-looking bare password literals contain punctuation not used in paths/hashes.
        for number,line in enumerate(text.splitlines(),1):
            for atom in re.findall(r'''[^\s"'`<>]{28,64}''',line):
                if (re.fullmatch(r'[A-Za-z0-9!@#$%^&*?+=-]{28,64}',atom)
                    and sum(c in '!@#$%^&*?' for c in atom)>=3
                    and any(c.isupper() for c in atom) and any(c.islower() for c in atom)
                    and sum(c.isdigit() for c in atom)>=2):
                    findings.append({'file':name,'kind':'password_like_literal','line':number})
    return findings,members


def group_for(path):
    n=path.name
    if n.startswith('ownv2_fixed100_vae_'):return '01_frozen_vae'
    if n.startswith(('own512_','ownv2_')):return '02_ownae_v2'
    if n.startswith(('teacher_cad50_','cad50_','graph_activation_')):return '03_cad50_diagnostics'
    return '04_historical_ae'


def main():
    assert not STAGE.exists(), 'Do not overwrite a prior snapshot'
    manifest=[]; exclusions=[]; nested=[]; selected=[]
    inventory=json.loads((ROOT/'evidence/local_inventory.json').read_text())
    for row in inventory:
        source=Path(row['path']); group=group_for(source)
        for base,dirs,files in os.walk(source):
            dirs[:]=[x for x in dirs if x not in ['.git','__pycache__','.pytest_cache','.ruff_cache']]
            for name in sorted(files):
                p=Path(base)/name
                selected.append((p,group,Path('workspace')/p.relative_to(WORK)))
    for p in [Path('/Users/luthier/Desktop/Nexus_OwnAE_VAE_定稿代码_20261006.zip'),
              WORK/'output/pdf/nexus_vae_code_lecture_20261002/Nexus_VAE_代码精读与PyTorch实践.pdf',
              WORK/'output/pdf/nexus_vae_code_lecture_20261002/Nexus_VAE_注释源码与练习.zip']:
        selected.append((p,'01_frozen_vae',Path('reading')/p.name))
    for p in (WORK/'output/handoffs').glob('Own*.zip'):
        selected.append((p,'02_ownae_v2',Path('handoffs')/p.name))
    for name in ['OwnAE_V2_all100_step34220_visual_results.pdf','OwnAE_V2_all100_step21280_vs24920.pdf']:
        p=WORK/'output/pdf'/name
        selected.append((p,'02_ownae_v2',Path('visualizations')/name))
    for number,(source,group,relative) in enumerate(selected):
        name=source.name
        if source.is_symlink() or name in {'.DS_Store','.env','.netrc','credentials.json'} or source.suffix in {'.pyc','.pem','.key'}:
            exclusions.append({'source':str(source),'reason':'cache_symlink_or_sensitive_filename'});continue
        data=source.read_bytes(); digest=sha(data)
        findings,members=inspect(data,str(source))
        if findings:
            exclusions.append({'source':str(source),'bytes':len(data),'sha256':digest,
                               'reason':'credential_pattern_quarantine','findings':findings})
            continue
        dest=STAGE/group/relative
        dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
        manifest.append({'group':group,'path':str(relative),'source':str(source),
                         'bytes':len(data),'sha256':digest,'operation':'byte_identical_copy'})
        nested.extend(members)
        if number % 500 == 0:print('COLLECT',number,len(manifest),flush=True)
    write(ROOT/'evidence/local_file_manifest.json',manifest)
    write(ROOT/'evidence/local_exclusions.json',exclusions)
    write(ROOT/'evidence/archive_member_manifest.json',nested)
    write(ROOT/'evidence/local_security_scan.json',{'selected':len(selected),'included':len(manifest),
      'nested_members_hashed':len(nested),'credential_quarantined':sum(e['reason']=='credential_pattern_quarantine' for e in exclusions),
      'patterns':list(PATTERNS)+['password_like_literal'],'secrets_or_matched_values_written':False,
      'scope':'UTF-8 text, recursively nested ZIP text; binary data hashed, not executed'})
    print('COLLECTED',len(manifest),'members',len(nested),'excluded',len(exclusions),flush=True)


if __name__=='__main__':main()
