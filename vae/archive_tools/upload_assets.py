"""Upload only the independent VAE Release; take authentication over stdin."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

REPO='leoguohr/nexus'
TAG='vae-evidence-2026-10-08'


def main():
    config=json.load(sys.stdin)
    token=config.pop('credential')
    release_id=config['release_id']
    receipts=Path(config['receipts']);receipts.mkdir(parents=True,exist_ok=True)
    headers={'Authorization':'Bearer '+token,'Accept':'application/vnd.github+json','User-Agent':'nexus-vae-archive'}
    def api(path,method=None):
        with urllib.request.urlopen(urllib.request.Request('https://api.github.com'+path,headers=headers,method=method),timeout=120) as r:return json.load(r) if method!='DELETE' else None
    info=api('/repos/'+REPO)
    release=api('/repos/'+REPO+'/releases/'+str(release_id))
    assert info['private'] and release['tag_name']==TAG
    upload_base=release['upload_url'].split('{',1)[0]
    assert upload_base.startswith('https://uploads.github.com/repos/'+REPO+'/')
    def upload(entry):
        path=Path(entry['path']);name=entry['asset']
        assert path.stat().st_size==entry['bytes'] and path.stat().st_size<900*1024**2
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
        assert h.hexdigest()==entry['sha256'],name
        result=None
        for attempt in range(3):
            assets=api('/repos/'+REPO+'/releases/'+str(release_id)+'/assets?per_page=100')
            found=[a for a in assets if a['name']==name]
            if found:
                a=found[0]
                if a['state']=='uploaded' and a['size']==entry['bytes'] and a.get('digest')=='sha256:'+entry['sha256']:
                    result=a;break
                # Only our own new release's incomplete upload may be cleaned up.
                assert a['state']=='starter' and a['size']==0,(name,a['state'],a['size'])
                api('/repos/'+REPO+'/releases/assets/'+str(a['id']),method='DELETE')
            response=receipts/(name+'.response.json')
            url=upload_base+'?'+urllib.parse.urlencode({'name':name})
            curl_config='header = '+json.dumps('Authorization: Bearer '+token)+'\n'
            cmd=['curl','--silent','--show-error','--fail-with-body','--http1.1',
                 '--connect-timeout','30','--max-time','1800','--config','-',
                 '-H','Content-Type: application/octet-stream','-X','POST',
                 '--data-binary','@'+str(path),'--output',str(response),url]
            print('UPLOAD_START',name,entry['bytes'],'attempt',attempt+1,flush=True)
            run=subprocess.run(cmd,input=curl_config,text=True,capture_output=True)
            if run.returncode==0:
                a=json.loads(response.read_text())
                assert a['size']==entry['bytes'] and a.get('digest')=='sha256:'+entry['sha256'],name
                result=a;break
            print('UPLOAD_RETRY',name,'curl_code',run.returncode,flush=True)
            time.sleep(2)
        assert result is not None,'Upload failed: '+name
        receipt={'asset':name,'bytes':entry['bytes'],'local_sha256':entry['sha256'],
                 'github_digest':result['digest'],'asset_id':result['id'],
                 'release_id':release_id,'release_tag':TAG,'verified':True,
                 'verification':'source SHA256 equals GitHub upload asset SHA256 digest',
                 'url':result['browser_download_url']}
        (receipts/(name+'.verified.json')).write_text(json.dumps(receipt,indent=2)+'\n')
        print('UPLOAD_VERIFIED',name,result['digest'],flush=True)
        return receipt
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.get('workers',2)) as pool:
        results=list(pool.map(upload,config['assets']))
    (receipts/'ALL_VERIFIED.json').write_text(json.dumps(results,indent=2)+'\n')
    print('ALL_UPLOADS_VERIFIED',len(results),flush=True)


if __name__=='__main__':main()
