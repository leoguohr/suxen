"""Prepare only the dedicated Topology Flow subtree and reproducible release inventory."""
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PUB=ROOT/'repo/topology_flow/2026-10-08'
WORKSPACE=ROOT.parents[2]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    PUB.mkdir(parents=True,exist_ok=False)
    for name in ('snapshot','reports'):
        shutil.copytree(ROOT/name,PUB/name,ignore=shutil.ignore_patterns('__pycache__','*.partial-download'))
    (PUB/'audit').mkdir();(PUB/'tools').mkdir()
    for name in ('SOURCE_MANIFEST.json','SERVER_WEIGHT_MANIFEST.json','REMOTE_ASSET_MANIFEST.json','VISUAL_ASSET.json',
            'VISUAL_MANIFEST.json','LOCAL_SOURCE_MANIFEST.json','local_inventory.json','local_inventory.md',
            'server_runtime_snapshot.json','remote_security_scan.json','frozen_vae_dependency.json','github_publish_tests.json'):
        shutil.copy2(ROOT/'audit'/name,PUB/'audit'/name)
    for p in (ROOT/'tools').glob('*.py'):shutil.copy2(p,PUB/'tools'/p.name)
    for source,label in [('output/pdf/nexus_topology_structure_20261006','20261006'),('output/pdf/topology_candidates_20261007','20261007')]:
        target=PUB/'research'/label;target.mkdir(parents=True)
        for p in (WORKSPACE/source).glob('*.pdf'):shutil.copy2(p,target/p.name)
    shutil.copy2(ROOT/'README_publication.md',PUB/'README.md')
    files=[]
    for p in sorted(PUB.rglob('*')):
        if p.is_file():files.append(dict(path=str(p.relative_to(PUB)),bytes=p.stat().st_size,sha256=sha(p)))
    (PUB/'FILE_MANIFEST.json').write_text(json.dumps(dict(files=files,scope='Snapshot content; this manifest and outer Release asset manifests excluded to avoid self-reference'),ensure_ascii=False,indent=2)+'\n')
    report=ROOT/'assets/TopologyFlow_report_bundle_20261008.zip'
    with zipfile.ZipFile(report,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(PUB.rglob('*')):
            if p.is_file():z.write(p,'topology_flow/2026-10-08/'+str(p.relative_to(PUB)))
    assets=json.loads((ROOT/'audit/REMOTE_ASSET_MANIFEST.json').read_text())['assets']
    assets.append(json.loads((ROOT/'audit/VISUAL_ASSET.json').read_text()))
    for p in (ROOT/'assets/TopologyFlow_local_research_and_history_20261008.zip',report):
        assets.append(dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)))
    record=dict(repository='leoguohr/nexus',private=True,branch='archive/topology-flow-20261008',
        release_tag='topology-flow-evidence-2026-10-08',assets=assets,
        exclusions='Full model/Adam states stay on persistent server; eight exact file hashes plus frozen VAE dependency are in audit. No independent Vertex runs or teacher source bundles.',
        extraction='c0/candidates: extract into snapshot/. fixed50/initial_history: strip first component into snapshot/initial/. visuals: extract at this report root. local research/history: optional historical supplement.')
    for directory in (PUB,ROOT/'assets'):
        (directory/'ASSET_MANIFEST.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        (directory/'RELEASE_SHA256SUMS.txt').write_text(''.join(e['sha256']+'  '+e['name']+'\n' for e in assets))
    print(json.dumps(dict(publication=str(PUB),files=len(files),archive_bytes=report.stat().st_size,release_payload_bytes=sum(e['bytes'] for e in assets)),indent=2))


if __name__=='__main__':main()
