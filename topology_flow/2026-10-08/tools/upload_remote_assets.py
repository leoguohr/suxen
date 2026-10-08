"""Upload only this task's prehashed assets. Credential is one stdin JSON in memory."""
import json
from pathlib import Path
import sys
import time
import urllib.parse
import github_publish as g

ROOT=Path('/guohaoran/nexus_fast_track/diagnostics/topology_flow_export_20261008')


def write(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


class ProgressReader:
    def __init__(self,stream,name,total):
        self.stream,self.name,self.total=stream,name,total
        self.bytes=0;self.last=0
    def read(self,size):
        block=self.stream.read(size);self.bytes+=len(block)
        if time.monotonic()-self.last>10 or self.bytes==self.total:
            write(ROOT/'upload_progress.json',dict(name=self.name,bytes_read=self.bytes,total_bytes=self.total,unix=time.time()))
            self.last=time.monotonic()
        return block


def main():
    request=json.loads(sys.stdin.readline())
    credential=request.pop('credential')
    g.private_repository(credential)
    release=g.api(g.PREFIX+'/releases/tags/'+g.TAG,credential)
    assert release['id']==request['release_id'] and g.remote_tag(credential)==request['commit']
    assert request['release_marker'] in release['body'] and not release['draft']
    entries=json.loads((ROOT/'REMOTE_ASSET_MANIFEST.json').read_text())['assets']
    entries.append(json.loads((ROOT/'VISUAL_ASSET.json').read_text()))
    receipts=ROOT/'upload_receipts';receipts.mkdir(exist_ok=True)
    for entry in entries:
        name=entry['name'];path=(ROOT/name).resolve()
        assert path.parent==ROOT and path.stat().st_size==entry['bytes'] and entry['bytes']<2**31
        assert g.sha(path)==entry['sha256']
        print('UPLOAD_BEGIN',name,entry['bytes'],flush=True)
        for attempt in range(3):
            asset=g.matching_asset(credential,release['id'],name,entry['bytes'],entry['sha256'])
            if asset:break
            url=release['upload_url'].split('{',1)[0]+'?name='+urllib.parse.quote(name)
            try:
                with path.open('rb') as stream:
                    g.request_json(url,credential,ProgressReader(stream,name,entry['bytes']),method='POST',size=entry['bytes'],timeout=600)
            except g.RequestError:
                print('UPLOAD_RESPONSE_UNKNOWN',name,'attempt',attempt+1,flush=True)
            asset=g.matching_asset(credential,release['id'],name,entry['bytes'],entry['sha256'])
            if asset:break
            time.sleep(3)
        if not asset:raise RuntimeError('Asset not verified after bounded attempts: '+name)
        record=dict(name=name,bytes=entry['bytes'],sha256=entry['sha256'],github_asset_id=asset['id'],
            github_digest=asset['digest'],github_state=asset['state'],release_id=release['id'],
            release_tag=g.TAG,url=asset['browser_download_url'],verified_unix=time.time())
        write(receipts/(name+'.json'),record)
        print('UPLOAD_VERIFIED',name,entry['sha256'],flush=True)
    print('REMOTE_UPLOADS_COMPLETE',flush=True)


if __name__=='__main__':main()
