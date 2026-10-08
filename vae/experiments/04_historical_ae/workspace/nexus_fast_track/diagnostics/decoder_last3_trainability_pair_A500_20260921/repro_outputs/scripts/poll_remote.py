"""Read-only progress snapshot; no training launch, resume, or checkpoint load."""
import json,subprocess
from pathlib import Path
from datetime import datetime,timezone
REMOTE=r'''
from pathlib import Path
import json,datetime
n=Path('/guohaoran/nexus_fast_track/diagnostics/decoder_last3_trainability_pair_A500_20260921')
d={'observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':json.loads((n/'status.json').read_text()),'branches':{},'failures':{}}
for branch in ['Control_last2','Treatment_last3']:
 r=n/branch/'run'
 if not r.exists():continue
 rows=[json.loads(x) for x in (r/'step_records.jsonl').read_text().splitlines()];u=[json.loads(x) for x in (r/'updates.jsonl').read_text().splitlines()]
 last=rows[-1];actual=[]
 for p in sorted(r.glob('actual-new*.json')):
  e=json.loads(p.read_text());actual.append({'step':e['new_updates'],'summary':e['summary'],'lost_A500':e['lost_A500'],'new_over_A500':e['new_over_A500']})
 d['branches'][branch]={'updates':len(u),'last_edge_state':last['after_new_updates'],'edge_success':last['edge_success_count'],'edge_fp':last['edge_fp'],'edge_fn':last['edge_fn'],'minimum_edge_success':min(x['edge_success_count'] for x in rows),'maximum_edge_success':max(x['edge_success_count'] for x in rows),'ever_lost_A500':sorted({uid for x in rows for uid in x['lost_vs_A500']}),'latest_actual_delta_norm':u[-1]['actual_delta_norm'] if u else None,'full_evaluations':actual,'complete':(n/branch/'complete.json').exists()}
for p in n.glob('*failure.txt'):d['failures'][p.name]=p.read_text()[-4000:]
records=sorted([(json.loads(p.read_text()),p) for p in (n/'repro_outputs/_runtime/train').glob('*/state.json')],key=lambda x:x[0]['created_at'])
d['runtime_history']=[{k:x.get(k) for k in ['run_id','status','retry_of','attempt']} for x,p in records]
if records:
 s,p=records[-1];d['runtime']={k:s.get(k) for k in ['run_id','status','pid','returncode','last_heartbeat']}
 if s['status'] not in ['running','success']:d['failures']['runtime_stderr']=(p.parent/'stderr.log').read_text()[-4000:]
for p in (n/'repro_outputs/recovery_204').glob('*failure.txt'):d['failures'][p.name]=p.read_text()[-4000:]
for name in ['ACCEPTANCE.json','DELIVERY.json']:
 p=n/'repro_outputs'/name
 if p.exists():d[name]=json.loads(p.read_text())
sp=n/'repro_outputs/recovery_204/resume_train_supervisor.log'
d['supervisor_log_tail']=(sp if sp.exists() else n/'train_supervisor.log').read_text()[-2000:]
print(json.dumps(d))
'''
cmd=['/usr/bin/ssh','-S','/tmp/nexus-last3-pair-20260921.sock','-o','BatchMode=yes','-p','36910','root@172.16.78.10','/opt/conda/bin/python -B -']
d=json.loads(subprocess.check_output(cmd,input=REMOTE,text=True,timeout=30));(Path(__file__).resolve().parents[1]/'LIVE_SNAPSHOT.json').write_text(json.dumps(d,indent=2)+'\n')
compact={'at':d['observed_at'],'runtime':d.get('runtime',{}).get('status'),'branches':{},'failures':d['failures']}
for b,v in d['branches'].items():
 e=v['full_evaluations'][-1] if v['full_evaluations'] else None
 compact['branches'][b]={'updates':v['updates'],'edge_state':v['last_edge_state'],'edge':[v['edge_success'],v['edge_fp'],v['edge_fn']],'edge_range':[v['minimum_edge_success'],v['maximum_edge_success']],'ever_lost_count':len(v['ever_lost_A500']),'last_full':[e['step'],e['summary']['joint_perfect'],e['summary']['edge']['fp'],e['summary']['edge']['fn'],e['summary']['face']['fp'],e['summary']['face']['fn']] if e else None,'complete':v['complete']}
compact['cpu_acceptance']=d.get('ACCEPTANCE.json',{}).get('cpu_audit_passed');compact['delivery_ready']='DELIVERY.json' in d
if d['supervisor_log_tail']:compact['supervisor_log_tail']=d['supervisor_log_tail']
print(json.dumps(compact,ensure_ascii=False))
