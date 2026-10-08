"""Send credentials through stdin; never serialize them in the archive."""
import json
import subprocess
import sys
from github_api import ROOT,credential

kind=sys.argv[1]
assert kind in ('local','server')
manifest=json.loads((ROOT/'evidence/ASSET_MANIFEST.json').read_text())
release=json.loads((ROOT/'evidence/release_created.json').read_text())
data={'credential':credential(),'release_id':release['id'],'workers':2,
      'assets':[r for r in manifest['assets'] if r['upload_source']==kind]}
if kind=='local':
    data['receipts']=str(ROOT/'upload_receipts')
    command=[sys.executable,str(ROOT/'tools/upload_assets.py')]
else:
    data['receipts']='/tmp/nexus_vae_archive_20261008/upload_receipts'
    command=['ssh','-S','/tmp/nexus_vae_upload2_20261008.sock','-p','36910','root@172.16.78.10',
             'python /tmp/nexus_vae_archive_20261008/tools/upload_assets.py']
result=subprocess.run(command,input=json.dumps(data),text=True)
sys.exit(result.returncode)
