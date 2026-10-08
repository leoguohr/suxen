"""Fixed20 complete-mesh training. --preflight never performs optimizer.step."""
import argparse,fcntl,hashlib,inspect,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REF=ROOT.parent/'math00_four_mesh_lr10x_20260913'
sys.path.insert(0,str(REF))
import backend00 as b
T=b.torch;np=b.np;c=b.c
from mini_nexus.data_2k import Nexus2KManifestDataset
from mini_nexus.packed_topology import collate_packed_topology

def write(p,x):
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,indent=2,allow_nan=False,default=str)+'\n');tmp.replace(p)
def th(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def perfect(r):return all(r[k]['fp']==r[k]['fn']==0 for k in ['edge','face'])
def klparts(rows):
    mu,lv=rows[0][0],rows[1][0]
    km=.5*mu.square().mean();ks=.5*(lv.exp()-1-lv).mean()
    return km+ks,dict(mu=float(km.detach()),sigma=float(ks.detach()),total=float((km+ks).detach()))
def movement(group,before):
    ds=ts=0.;changed=0
    for n,p in group.items():
        d=p.detach().double()-before[n].double();ds+=float(d.square().sum());ts+=float(before[n].double().square().sum());changed+=int(d.count_nonzero())
    return dict(delta_l2=ds**.5,relative_l2=(ds/(ts+1e-30))**.5,changed_elements=changed)

def main(args):
    cfg=json.loads((ROOT/'recommended_config.json').read_text());sel=json.loads((ROOT/'selection.json').read_text())
    ready=json.loads((ROOT/'pools_ready.json').read_text());uids=sel['uids'];assert len(uids)==len(set(uids))==20
    lock=(ROOT/'execution.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    T.set_num_threads(1);T.cuda.set_device(0);T.backends.cuda.matmul.allow_tf32=False;T.backends.cudnn.allow_tf32=False;T.set_float32_matmul_precision('highest')
    start=Path(args.resume or cfg['start_checkpoint'])
    if not args.resume:assert c.m.probe.digest(start)==cfg['start_checkpoint_sha256']
    if not args.preflight:
        pre=json.loads((ROOT/'preflight/result.json').read_text());assert pre['passed'] and pre['optimizer_updates']==0
        assert pre['entry_sha256']==c.m.probe.digest(Path(__file__))
        assert pre['config_sha256']==c.m.probe.digest(ROOT/'recommended_config.json')
        for name,digest in pre['input_file_sha256'].items():assert c.m.probe.digest(ROOT/name)==digest,name
    c.m.probe.UIDS=uids
    cp,model,unused=c.m.probe.setup_model(start);del unused
    offset=int(cp.get('twenty_mesh_updates',0));assert 0<=offset<=10000
    segment=f'from_{offset:05d}'+(f'_resume_{time.time_ns()}' if args.resume else '')
    out=ROOT/'preflight' if args.preflight else ROOT/'run'/segment
    out.mkdir(parents=True,exist_ok=True)
    if not args.preflight:assert not (out/'updates.jsonl').exists(),'Output already exists; resume from a saved checkpoint instead.'
    b.ROOT=out;b.backend.ROOT=out;(out/'C_graph_only').mkdir(exist_ok=True)
    provenance=cp['diagnostic_manifest']
    assert c.m.probe.digest(REF/'backend00.py')==provenance['backend_sha']
    for p,h in provenance['source_sha256'].items():assert c.m.probe.digest(p)==h,p
    runtime=c.m.install_clamp(model,-20.)
    old='noise = torch.randn_like(sample_mu)';assert runtime.count(old)==1
    runtime=runtime.replace(old,"noise = torch.randn(sample_mu.shape, dtype=sample_mu.dtype, device=sample_mu.device, generator=autoencoder.diagnostic_train_rng)\n                autoencoder.diagnostic_train_draws += 1")
    exec(compile(runtime,str(out/'sampling_forward.py'),'exec'),b.backend.b.flash.__dict__);(out/'sampling_forward.py').write_text(runtime)
    rng=T.Generator(device='cuda');rng.set_state(cp['training_rng_state']);model.autoencoder.diagnostic_train_rng=rng
    model.autoencoder.diagnostic_train_draws=cp['training_noise_draws']
    b.install_deterministic(model)
    for section,blocks in [('encoder',model.autoencoder.encoder_blocks),('decoder',model.autoencoder.decoder_blocks)]:
        for i,block in enumerate(blocks):
            for name,module in block.named_modules():
                if isinstance(module,T.nn.MultiheadAttention):b.ATTENTION_NAMES[id(module)]=f'{section}_{i:02d}/{name}'
    assert len(b.ATTENTION_NAMES)==28
    model.requires_grad_(True);named=dict(model.autoencoder.named_parameters())
    groups={name:{n:named[n] for n in names} for name,names in provenance['groups'].items()}
    opt=T.optim.Adam([dict(params=list(g.values()),name=name,lr=cfg['lr'][name]) for name,g in groups.items()],weight_decay=0.)
    opt.load_state_dict(cp['optimizer'])
    for g in opt.param_groups:
        assert abs(g['lr']-cfg['lr'][g['name']])<1e-20 and g['weight_decay']==0
    assert all(T.equal(p.detach().cpu(),cp['model'][n]) for n,p in model.named_parameters())
    for g,oldg in zip(opt.param_groups,cp['optimizer']['param_groups']):
        for p,idx in zip(g['params'],oldg['params']):
            for k,v in cp['optimizer']['state'][idx].items():assert T.equal(opt.state[p][k].cpu(),v)
    active=[p for g in groups.values() for p in g.values()]
    assert len(active)==len(set(map(id,active)))==len(list(model.parameters()))
    ds=Nexus2KManifestDataset(Path(cp['args']['manifest']),'train')
    samples=[ds[ds.index_for_uid(u)] for u in uids]
    batches=[collate_packed_topology([s]).to('cuda') for s in samples]
    pools=[]
    for i,u in enumerate(uids):
        path=ROOT/'pools'/(u+'_pool.npz');assert c.m.probe.digest(path)==ready['records'][i]['pool_sha256']
        d=np.load(path);pools.append(d)
        assert np.array_equal(d['vertices'],batches[i].vertices[0,:len(d['vertices'])].cpu().numpy())
        assert np.array_equal(d['positive'],batches[i].face_set[0].cpu().numpy())
    scales=model.scoring_contract();c.PAIR_CHUNK=cp['args']['pair_chunk_size']
    # Same actual reconstruction routine; suppress huge duplicate NPZ output only.
    capsrc=inspect.getsource(c.capture).replace("np.savez_compressed(out/f'{uid}.npz',**table);tables.append(table)","tables.append(table)")
    ns=dict(c.capture.__wrapped__.__globals__);exec(capsrc,ns);capture=ns['capture']
    seeds=json.loads((ROOT/'evaluation_seeds.json').read_text())
    manifest=dict(parent_checkpoint=str(start),parent_sha256=c.m.probe.digest(start),uids=uids,groups=provenance['groups'],source_sha256=provenance['source_sha256'],backend_sha=provenance['backend_sha'],entry_sha256=c.m.probe.digest(Path(__file__)),config_sha256=c.m.probe.digest(ROOT/'recommended_config.json'),pools=ready['records'],microbatch_uid_order=[[u] for u in uids],logical_batch_size=20,objective='sum_mesh(EdgeSoft4+FaceSoft4+1e-4*KL)/20',lr={g['name']:g['lr'] for g in opt.param_groups},budget=10000,parent_offset=offset)
    write(out/'manifest.json',manifest)

    def forward(i,mode,seed=None):
        rows,stats,zs=c.m.forward(model,batches[i],mode,None if seed is None else [seed])
        for mu,z,st in zip(rows[0],zs,stats):
            d=z.detach()-mu.detach();st.update(actual_noise_rms=float(d.double().square().mean().sqrt()),actual_noise_max_abs=float(d.abs().max()),actual_noise_nonzero_elements=int(d.count_nonzero()))
        return rows,stats

    def eval_set(step,mode,noise_seeds,label):
        before=rng.get_state().clone();results=[]
        for i,u in enumerate(uids):
            rows,stats=forward(i,mode,None if noise_seeds is None else noise_seeds[i])
            detached=tuple(tuple(x.detach() for x in r) for r in rows);del rows
            with T.no_grad():
                loss,parts,_=b.full_objective(detached,[pools[i]],scales);_,ki=klparts(detached)
                capture.__wrapped__.__globals__['UIDS']=[u]
                rec,_=capture(detached,[pools[i]],scales,out)
            r=rec[0];r.pop('near',None);r.pop('crossings',None)
            results.append(dict(uid=u,rec=r,parts=parts[0],kl=ki,posterior=stats[0],perfect=perfect(r)))
            del detached
        assert T.equal(before,rng.get_state())
        v=dict(step=step,mode=mode,seeds=noise_seeds,meshes=results,all20_perfect=all(x['perfect'] for x in results),evaluation_layout='20 sequential complete-mesh forwards; one checkpoint and one prescribed set of noises')
        write(out/f'{label}_step{step:05d}.json',v);return v

    def noise_eval(step,final=False):
        tag='final_noise' if final else 'monitoring';collection=seeds[tag];values=[]
        with (out/f'{tag}_step{step:05d}.jsonl').open('w',buffering=1) as f:
            for j,noise in enumerate(collection):
                result=eval_set(step,'sample',noise,f'{tag}_{j:02d}');values.append(result);f.write(json.dumps(result,allow_nan=False)+'\n')
                if (j+1)%10==0:print('NOISE',tag,step,j+1,flush=True)
        summary=dict(step=step,total=50,all20_perfect=sum(r['all20_perfect'] for r in values),per_mesh=[dict(uid=u,perfect=sum(r['meshes'][i]['perfect'] for r in values),mean_parts={k:float(np.mean([r['meshes'][i]['parts'][k] for r in values])) for k in ['edge','face']}) for i,u in enumerate(uids)])
        write(out/f'{tag}_summary_step{step:05d}.json',summary)

    def accumulated_backward():
        opt.zero_grad(set_to_none=True);records=[];before=th(rng.get_state())
        for i,u in enumerate(uids):
            rows,stats=forward(i,'sample');after_noise=rng.get_state().clone();eh=th(model.autoencoder._diagnostic_eps[0])
            rec,parts,_=b.full_objective(rows,[pools[i]],scales);kl,ki=klparts(rows)
            v=(rec+1e-4*kl)/20
            records.append(dict(uid=u,parts=parts[0],kl=ki,posterior=stats[0],weighted_objective=float(v.detach()),epsilon_sha256=eh))
            v.backward();assert T.equal(after_noise,rng.get_state())
            del v,kl,rec,rows
        return dict(meshes=records,objective=sum(x['weighted_objective'] for x in records),rng_before_sha256=before,rng_after_sha256=th(rng.get_state()))

    if args.preflight:
        start_rng=rng.get_state().clone();start_draw=model.autoencoder.diagnostic_train_draws
        # Compare two complete meshes packed vs per-mesh accumulation at fixed noise.
        opt.zero_grad(set_to_none=True)
        packed=collate_packed_topology(samples[:2]).to('cuda')
        rr,_,_=c.m.forward(model,packed,'sample',seeds['fixed'][:2])
        packed_rec,_,_=b.full_objective(rr,pools[:2],scales)
        packed_kl=sum(klparts(tuple((x[i],) for x in rr))[0] for i in range(2))
        packed_value=(2*packed_rec+1e-4*packed_kl)/20
        packed_scalar=float(packed_value.detach());packed_value.backward()
        reference={n:p.grad.detach().clone() for n,p in named.items() if p.grad is not None}
        del packed,rr,packed_rec,packed_kl,packed_value
        opt.zero_grad(set_to_none=True);single_scalar=0.
        for i in range(2):
            rr,_=forward(i,'sample',seeds['fixed'][i]);re,_,_=b.full_objective(rr,[pools[i]],scales);kk,_=klparts(rr)
            vv=(re+1e-4*kk)/20;single_scalar+=float(vv.detach());vv.backward();del vv,rr,re,kk
        diff=refnorm=0.
        for n,g in reference.items():
            diff+=float((named[n].grad.double()-g.double()).square().sum());refnorm+=float(g.double().square().sum())
        accumulation_check=dict(packed_loss=packed_scalar,accumulated_loss=single_scalar,gradient_relative_l2=(diff/(refnorm+1e-30))**.5)
        assert abs(single_scalar-packed_scalar)<1e-5 and accumulation_check['gradient_relative_l2']<.005,accumulation_check
        del reference;opt.zero_grad(set_to_none=True)
        T.cuda.reset_peak_memory_stats();t=time.monotonic()
        record=accumulated_backward();T.cuda.synchronize();seconds=time.monotonic()-t
        grads={name:c.m.gradnorm(g.values()) for name,g in groups.items()}
        assert all(np.isfinite(x) and x>0 for x in grads.values())
        assert all(r['posterior']['actual_noise_nonzero_elements']>0 for r in record['meshes'])
        assert all(T.equal(p.detach().cpu(),cp['model'][n]) for n,p in model.named_parameters())
        opt.zero_grad(set_to_none=True);rng.set_state(start_rng);model.autoencoder.diagnostic_train_draws=start_draw
        # Complete hard reconstruction on all20; no optimizer step.
        mu=eval_set(0,'mu',None,'mu');sample=eval_set(0,'sample',seeds['fixed'],'sample')
        write(out/'result.json',dict(passed=True,input_file_sha256={name:c.m.probe.digest(ROOT/name) for name in ['selection.json','pools_ready.json','evaluation_seeds.json']},accumulation_check=accumulation_check,optimizer_updates=0,entry_sha256=manifest['entry_sha256'],config_sha256=manifest['config_sha256'],parent_sha256=manifest['parent_sha256'],peak_gib=T.cuda.max_memory_allocated()/2**30,one_logical_backward_seconds=seconds,gradient_norms=grads,all_parameters_unchanged=True,training_rng_restored=T.equal(start_rng,rng.get_state()),all20_backward=record,mu_all20_perfect=mu['all20_perfect'],sample_all20_perfect=sample['all20_perfect']))
        print('PREFLIGHT_PASSED_NO_UPDATES',seconds,flush=True);return

    def save(step):
        p=out/f'checkpoint-update{step:05d}.pt';tmp=p.with_suffix('.pt.tmp')
        T.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),args=cp['args'],completed_updates=4400+step,twenty_mesh_updates=step,diagnostic_manifest=manifest,training_rng_state=rng.get_state(),training_noise_draws=model.autoencoder.diagnostic_train_draws),tmp);tmp.replace(p)
        write(ROOT/'run/latest.json',dict(checkpoint=str(p),step=step))
    save(offset)
    eval_set(offset,'mu',None,'mu');eval_set(offset,'sample',seeds['fixed'],'sample');noise_eval(offset)
    with (out/'updates.jsonl').open('w',buffering=1) as log:
        for step in range(offset+1,10001):
            t=time.monotonic();record=accumulated_backward()
            grads={name:c.m.gradnorm(g.values()) for name,g in groups.items()}
            norm=float(T.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True))
            before={n:p.detach().clone() for n,p in named.items()}
            opt.step();updates={name:movement(g,before) for name,g in groups.items()};del before
            record.update(update=step,gradient_norms_preclip=grads,global_norm_preclip=norm,clip_coefficient=min(1.,1./(norm+1e-6)),actual_updates=updates,seconds=time.monotonic()-t)
            log.write(json.dumps(record,allow_nan=False)+'\n')
            if step%10==0:print('UPDATE',step,'objective',record['objective'],'seconds',record['seconds'],flush=True)
            if step%200==0:
                save(step);eval_set(step,'mu',None,'mu');eval_set(step,'sample',seeds['fixed'],'sample')
            if step in [1000,2000,4000,6000,8000,10000]:noise_eval(step)
    noise_eval(10000,True);write(out/'complete.json',dict(updates=10000,stopped_at_budget=True));print('COMPLETE_10000',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',type=Path);args=parser.parse_args()
    main(args)
