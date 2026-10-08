"""Read-only Control checkpoint replay for fixed mined-negative logit identities."""
import json
from runtime import ROOT,T,np,c,b,setup,write,sha,tensor_hash
from train_low_lr import PARENT,restore_rng
from hardneg_tracking import record

def main():
    model,opt,groups,uids,old_pools,forward,capture,args=setup()
    assert not opt.state
    mining=json.loads((ROOT/'mining_complete.json').read_text())
    records={r['uid']:r for r in mining['records']};pools={}
    for u in uids:
        p=ROOT/'augmented_pools'/f'{u}_pool.npz';assert sha(p)==records[u]['new_pool_sha256']
        with np.load(p) as d:pools[u]={k:d[k] for k in ['vertices','edges','positive','mixed']}
    scales=model.scoring_contract();out=ROOT/'control_hardneg';out.mkdir(exist_ok=True)
    (ROOT/'control_evaluations').mkdir(exist_ok=True)
    results=[]
    for epoch,step in [(650,16250),(675,16875),(700,17500)]:
        summary=json.loads((PARENT/'run'/f'eval-summary-epoch{epoch}.json').read_text())
        p=PARENT/'run'/f'checkpoint-update{step}.pt';assert sha(p)==summary['checkpoint_sha256']
        cp=T.load(p,map_location='cpu',mmap=True,weights_only=False)
        model.load_state_dict(cp['model'],strict=True)
        restore_rng(cp,np.random.default_rng());del cp
        before=tensor_hash(model.named_parameters())
        original=(PARENT/'run'/f'eval-epoch{epoch}.jsonl').read_bytes()
        old={r['uid']:r for r in map(json.loads,original.splitlines())};assert list(old)==uids
        (ROOT/'control_evaluations'/f'eval-epoch{epoch}.jsonl').write_bytes(original)
        diagnostics=[]
        with (out/f'epoch{epoch}.jsonl').open('x',buffering=1) as f:
            for i,u in enumerate(uids):
                rows=forward(u);loss,parts,saved=b.full_objective(rows,[pools[u]],scales)
                assert parts[0]==old[u]['parts'],(epoch,u,'training-pool losses')
                y=np.r_[np.ones(len(pools[u]['positive']),dtype=bool),np.zeros(len(pools[u]['mixed']),dtype=bool)]
                assert c.metrics(y,saved[0]['face_train_logits'])==old[u]['face_training_pool'],(epoch,u,'pool counts')
                d=dict(uid=u,epoch=epoch,**record(out,epoch,u,pools[u],records[u],saved[0]))
                diagnostics.append(d);f.write(json.dumps(d,allow_nan=False)+'\n');del rows,loss,saved
                write(ROOT/'control_forward_status.json',dict(state='forward_only',epoch=epoch,meshes=i+1,optimizer_updates=0))
        assert tensor_hash(model.named_parameters())==before and not opt.state
        result=dict(epoch=epoch,checkpoint=str(p),checkpoint_sha256=summary['checkpoint_sha256'],
            all100_loss_and_pool_counts_exact=True,model_unchanged=True,optimizer_updates=0,
            fixed_mined_negatives=sum(d['count'] for d in diagnostics),mined_pool_fp=sum(d['pool_fp'] for d in diagnostics))
        results.append(result);print(json.dumps(result),flush=True)
    write(ROOT/'control_forward_verification.json',dict(completed=True,optimizer_updates=0,results=results,
        scope='Only forward replay for fixed mined-negative logits; actual reconstruction counts reused from completed Control'))

if __name__=='__main__':main()
