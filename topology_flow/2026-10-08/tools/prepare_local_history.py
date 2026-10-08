"""Package Topology-only local history/research; duplicate raw server data excluded."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]
WORKSPACE=ROOT.parents[2]


def main():
    inventory=json.loads((ROOT/'audit/local_inventory.json').read_text())
    included=[];excluded=[]
    archive=ROOT/'assets/TopologyFlow_local_research_and_history_20261008.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for entry in inventory['files']:
            rel=Path(entry['relative_path']); source=WORKSPACE/entry['path']
            reason=None
            if entry['recommendation'].startswith('exclude'):reason=entry['recommendation']
            if rel.parts[:2] in (('actual_run','cache'),('actual_run','inputs_verified'),('actual_run','vae_baseline')):
                reason='Identical role already included from authoritative server fixed50 archive; retain original inventory hash'
            if reason:
                excluded.append(dict(source=entry['path'],sha256=entry['sha256'],reason=reason));continue
            name='local_history/'+entry['root_id']+'/'+str(rel)
            if entry['root_id'].startswith('research_'):name='research/'+entry['root_id'].removeprefix('research_')+'/'+str(rel)
            data=source.read_bytes()
            assert hashlib.sha256(data).hexdigest()==entry['sha256'], str(source)
            z.writestr(name,data)
            included.append(dict(source=entry['path'],archive_path=name,bytes=len(data),sha256=entry['sha256']))
        record=dict(included=included,excluded=excluded,note='Historical states explicitly retained as history; stale placeholder charts excluded; no independent Vertex/teacher-source archives.')
        z.writestr('LOCAL_SOURCE_MANIFEST.json',json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'audit/LOCAL_SOURCE_MANIFEST.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print('LOCAL_ARCHIVE',archive.stat().st_size,len(included),'files')


if __name__=='__main__':main()
