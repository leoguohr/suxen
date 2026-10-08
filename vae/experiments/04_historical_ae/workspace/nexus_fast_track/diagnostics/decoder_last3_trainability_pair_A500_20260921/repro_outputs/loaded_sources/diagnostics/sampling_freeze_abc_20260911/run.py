"""Beta-zero A/B/C freezing from identical B16100 and matching Adam states."""
import os
os.environ.setdefault('PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION','python')
import importlib.util
import json
from pathlib import Path
import time
import argparse
import inspect
import hashlib
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'face_freeze_ab_20260910'
SOURCE=ROOT.parent/'logvar_only_lr_ab_20260911'
spec=importlib.util.spec_from_file_location('frozen_ab_helpers',BASE/'run.py')
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
probe=h.probe;teacher=h.teacher
import mini_nexus.topology as topology
START=16100;END=16600;PERIOD=50
START_SHA='09f56d810fb9be8f7de3cac0c58df4b7d7bde72b49098b72d728be696bdf7c56'
UIDS=['nexus_2k_000387','nexus_2k_001849']
STAGES=[('A_decoder',0.),('B_encoder',0.),('C_logvar',0.)]
ADAM_DONOR_SHA='fa2a34bc51ef3b27a6c8cca5b1538a6cd58ecee2304c31e138a78253c8559af3'


def install_clamp(model,minimum):
    import mini_nexus.flash_varlen_topology as flash
    source=getattr(flash,'_diagnostic_original_forward_source',None)
    if source is None:
        source=inspect.getsource(flash._flash_varlen_autoencoder_forward)
        flash._diagnostic_original_forward_source=source
    assert source.count('.clamp(-10.0, 10.0)') == 1
    patched=source.replace('.clamp(-10.0, 10.0)', '.clamp(autoencoder.diagnostic_logvar_min, 10.0)')
    assert patched.count('            latent_rows.append(')==1
    patched=patched.replace('            latent_rows.append(', '            autoencoder._diagnostic_eps.append(noise.detach())\n            latent_rows.append(')
    model.autoencoder.diagnostic_logvar_min=float(minimum)
    exec(compile(patched,str(ROOT/'clamp_forward.py'),'exec'),flash.__dict__)
    return patched



def write(path,obj):probe.write(path,obj)
def norm(t):return float(t.detach().double().norm())
def quantiles(t):
    q=torch.quantile(t.detach().float().flatten(),t.new_tensor([0.,.5,.95,1.]).float())
    return dict(zip(['min','median','p95','max'],map(float,q)))
def gradnorm(params):return float(sum((p.grad.detach().double().square().sum() for p in params if p.grad is not None),torch.zeros((),device='cuda',dtype=torch.float64)).sqrt())


def forward(model,batch,mode,seeds=None):
    # Only the AE parent's flag selects reparameterization; all child dropout stays in eval mode.
    model.eval();a=model.autoencoder;a.training=(mode!='mu');a._diagnostic_eps=[]
    raw=[]
    raw_hook=a.log_variance.register_forward_hook(lambda module,args,output:raw.append(output))
    cache=[]
    hook=a.latent_input.register_forward_pre_hook(lambda module,args:cache.append(args[0]))
    try:rows=model.topology_embedding_rows(batch,sample_seeds=seeds)
    finally:
        hook.remove();raw_hook.remove()
    z=cache[0];mu=torch.cat(rows[0]);lv=torch.cat(rows[1]);std=(.5*lv).exp()
    if mode=='mu':assert torch.equal(z,mu)
    if seeds is not None:
        eps=[]
        for m,seed in zip(rows[0],seeds):
            gen=torch.Generator(device=m.device).manual_seed(seed)
            eps.append(torch.randn(m.shape,device=m.device,dtype=m.dtype,generator=gen))
        expected=mu+std*torch.cat(eps)
        torch.testing.assert_close(z,expected,rtol=0,atol=0)
    assert all(not child.training for name,child in a.named_modules() if name)
    zs=z.split([len(x) for x in rows[0]])
    raw_rows=raw[0].split([len(x) for x in rows[0]])
    torch.testing.assert_close(lv,raw[0].clamp(a.diagnostic_logvar_min,10.),rtol=0,atol=0)
    stats=[]
    eps_rows=a._diagnostic_eps if mode!='mu' else [torch.zeros_like(m) for m in rows[0]]
    assert len(eps_rows)==len(rows[0])
    for m,l,zz,raw_l,eps in zip(rows[0],rows[1],zs,raw_rows,eps_rows):
        sigma=(l.detach()*.5).exp();noise=sigma*eps
        torch.testing.assert_close(zz.detach(),m.detach()+noise,rtol=0,atol=0)
        stats.append(dict(std=quantiles(sigma),std_mean=float(sigma.mean()),logvar=quantiles(l),raw_logvar=quantiles(raw_l),
            raw_below_lower_fraction=float((raw_l.detach()<a.diagnostic_logvar_min).float().mean()),
            lower_clamp_fraction=float((l.detach()<=a.diagnostic_logvar_min).float().mean()),upper_clamp_fraction=float((l.detach()>=10).float().mean()),
            mu_l2=norm(m),noise_l2=norm(noise),noise_relative_l2=norm(noise)/(norm(m)+1e-12),R=norm(noise)/(norm(m)+1e-12),R_effective_after_add=norm(zz.detach()-m.detach())/(norm(m)+1e-12),R_expected_rms=norm(sigma)/(norm(m)+1e-12),kl_mu=float(.5*m.detach().float().square().mean()),kl_sigma=float(.5*(l.detach().float().exp()-1-l.detach().float()).mean()),
            kl_diagnostic_only=float(.5*(m.detach().square()+l.detach().exp()-1-l.detach()).mean())))
    return rows,stats,zs


def main(stage_index,gpu,lr_logvar):
    global START,END
    phase,beta=STAGES[stage_index];START=16100;END=16600
    branch=dict(logvar_min=-20.,lr=lr_logvar,beta=beta)
    out=ROOT/phase;out.mkdir(exist_ok=True);assert not (out/'trace.jsonl').exists()
    torch.set_num_threads(1);torch.cuda.set_device(gpu);torch.manual_seed(20260910)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start_path=SOURCE/'B/checkpoint-16100.pt'
    parent_sha=probe.digest(start_path)
    assert parent_sha==START_SHA
    probe.UIDS=UIDS;cp,model,batch=probe.setup_model(start_path);a=model.autoencoder
    assert cp['diagnostic_run']['completed_steps']==START
    assert cp['diagnostic_run']['logvar_clamp']==[-20.,10]
    for path,sha in cp['diagnostic_run']['source_sha256'].items():assert probe.digest(path)==sha
    for name,p in model.named_parameters():assert torch.equal(p.cpu(),cp['model'][name]),name
    runtime=install_clamp(model,-20.);(out/'clamp_forward.py').write_text(runtime)
    model.requires_grad_(True)
    initial_logvar={n:p.detach().cpu().clone() for n,p in a.log_variance.named_parameters()}
    enc_prefix=('vertex_input.','face_input.','encoder_blocks.','encoder_output_norm.','mu.')
    named=list(a.named_parameters())
    enc=[p for n,p in named if n.startswith(enc_prefix)]
    dec=[p for n,p in named if not n.startswith(enc_prefix+('face_embedding.','log_variance.'))]
    face=list(a.face_embedding.parameters());lvparams=list(a.log_variance.parameters());params=enc+dec+face+lvparams
    assert set(map(id,params))=={id(p) for p in model.parameters()} and len(params)==len(list(model.parameters()))
    assert all(p.requires_grad for p in params)
    groups=[dict(params=enc,lr=1e-8),dict(params=dec,lr=1e-7),dict(params=face,lr=1e-7)]
    donor_path=BASE/'B/checkpoint-15100.pt'
    assert probe.digest(donor_path)==ADAM_DONOR_SHA
    donor=torch.load(donor_path,map_location='cpu',mmap=True)
    assert all(torch.equal(p.cpu(),donor['model'][n]) for n,p in model.named_parameters() if not n.startswith('autoencoder.log_variance.'))
    opt=torch.optim.Adam(groups,weight_decay=0.);opt.load_state_dict(donor['optimizer'])
    for group,saved_group in zip(opt.param_groups,donor['optimizer']['param_groups']):
        assert {k:v for k,v in group.items() if k!='params'}=={k:v for k,v in saved_group.items() if k!='params'}
        for p,idx in zip(group['params'],saved_group['params']):
            assert all(torch.equal(opt.state[p][k].cpu(),v) for k,v in donor['optimizer']['state'][idx].items())
    saved_group=cp['optimizer']['param_groups'][0]
    opt.add_param_group(dict({k:v for k,v in saved_group.items() if k!='params'},params=lvparams))
    for p,idx in zip(lvparams,saved_group['params']):
        opt.state[p]={k:v.clone() if k=='step' else v.clone().to(p.device) for k,v in cp['optimizer']['state'][idx].items()}
        assert all(torch.equal(opt.state[p][k].cpu(),v) for k,v in cp['optimizer']['state'][idx].items())
    del donor
    assert [g['lr'] for g in opt.param_groups[:3]]==[1e-8,1e-7,1e-7]
    opt.param_groups[3]['lr']=lr_logvar
    names_by_id={id(p):'autoencoder.'+n for n,p in named}
    initial_steps={names_by_id[id(p)]:float(opt.state[p]['step']) for p in params}
    optimizer_names=[[names_by_id[id(p)] for p in g['params']] for g in opt.param_groups]
    active_groups={0:[1,2],1:[0],2:[3]}[stage_index]
    for gi,group in enumerate(opt.param_groups):
        for p in group['params']:p.requires_grad_(gi in active_groups)
    active=[p for p in params if p.requires_grad]
    frozen=[p for p in params if not p.requires_grad]
    frozen_versions={id(p):p._version for p in frozen}
    frozen_adam={names_by_id[id(p)]:{k:v.detach().cpu().clone() for k,v in opt.state[p].items()} for p in frozen}
    def check_frozen():
        for p in frozen:
            name=names_by_id[id(p)]
            assert p.grad is None and p._version==frozen_versions[id(p)],name
            assert torch.equal(p.detach().cpu(),cp['model'][name]),name
            assert all(torch.equal(v.cpu(),frozen_adam[name][k]) for k,v in opt.state[p].items()),name
    data=[];tris=[];labels=[];positive_keys=[]
    prep=json.loads((h.OLD/'preparation.json').read_text())
    for i,uid in enumerate(UIDS):
        path=teacher.modules.ab.PREVIOUS/(uid+'_pool.npz')
        assert probe.digest(path)==prep['previous_configuration']['pool_sha256'][uid]
        d=np.load(path);data.append(d)
        assert np.array_equal(d['vertices'],batch.vertices[i,:len(d['vertices'])].cpu().numpy())
        assert np.array_equal(d['positive'],batch.face_set[i].cpu().numpy())
        tris.append(torch.as_tensor(np.concatenate([d['positive'],d['mixed']]),device='cuda',dtype=torch.long))
        labels.append(torch.cat([torch.ones(len(d['positive']),device='cuda'),torch.zeros(len(d['mixed']),device='cuda')]))
        positive_keys.append(teacher._canonical_positive_edge_keys(batch.edge_index[i],len(d['vertices']),'cuda'))
    scales=model.scoring_contract();chunk=cp['args']['pair_chunk_size']
    meta=dict(phase=phase,runtime_patch_sha256=hashlib.sha256(runtime.encode()).hexdigest(),
        runtime_patch='Flash clamp minimum restored and sampled epsilon captured for exact R; scoring/architecture unchanged',start_step=START,end_step=END,updates=500,start_checkpoint=str(start_path),start_sha256=parent_sha,root_start_sha256=START_SHA,stage_index=stage_index,adam_donor=str(donor_path),adam_donor_sha256=ADAM_DONOR_SHA,optimizer_parameter_names=optimizer_names,adam_initial_steps=initial_steps,
        script_sha256=probe.digest(Path(__file__)),helper_sha256=probe.digest(BASE/'run.py'),source_sha256=cp['diagnostic_run']['source_sha256'],
        selected_uids=UIDS,lr_encoder=1e-8,lr_logvar=branch['lr'],lr_decoder=1e-7,lr_face_head=1e-7,
        trainable=phase,active_optimizer_groups=active_groups,trainable_parameter_names=[names_by_id[id(p)] for p in active],trainable_parameter_count=sum(p.numel() for p in active),
        optimizer='restore matching E/D/heads Adam from B15100 plus B16100 logvar Adam; all branches restore identical four groups; frozen groups retain unchanged Adam states',
        kl_weight=branch['beta'],edge_weight=1.,face_weight=1.,objective='mean_mesh(EdgeSoft4 + FaceSoft4) + beta * mean_mesh(mean_vertex_channel(0.5*(mu^2+exp(logvar)-1-logvar)))',kl_reduction='FP32 mean over each mesh N*64, then equal mean of two meshes; same vae_kl_loss as original',
        soft4=dict(tau=1,epsilon=1e-8,membership='detach(sigmoid)',reduction='FP32 fixed divisor4'),
        logvar_clamp=[branch['logvar_min'],10],logvar_weights_reinitialized=False,logvar_initialization_method='restore learned posterior from B16100',clip=1.,weight_decay=0.,threshold=0.,
        training_noise='fresh paired torch.randn each step; gradient only to active parameters; B retains differentiation through frozen logvar head into shared encoder; exact sigma*epsilon captured for R',
        fixed_eval_seeds={'fixed0':[720000,720001],'fixed1':[730000,730001]},full_eval_period=PERIOD,logvar_delta_definition='median(logvar_t)-median(logvar_t-50), per mesh, both clamped and raw; measured before update at state t',
        evaluation_modes=['training_sample','mu','fixed0','fixed1'],dropout=False,data_augmentation=False,negative_resampling=False,
        checkpoint_period=200,extra_checkpoints=[END],full_face_cap=5000000,numerical_backend='unchanged FP32 weights/BF16 Flash; no TF32',
        gpu=gpu,device=torch.cuda.get_device_name(),torch_version=torch.__version__,
        acceptance='actual edge-gated face reconstruction at threshold0; candidate-cap overflow is incomplete, never perfect')
    write(out/'provenance.json',meta)
    torch.set_rng_state(cp['rng_state']['cpu']);torch.cuda.set_rng_state(cp['rng_state']['cuda'])
    evals=[];previous_mu=None;interval_previous=None;interval_initial=None;started=time.monotonic()

    def face_scores(rows):
        return [topology.face_interval_logits(*(rows[3][i][t[:,j]] for j in range(3)),logit_scale=scales['face_logit_scale'],area_factor=scales['face_interval_factor']) for i,t in enumerate(tris)]

    def evaluate(step,train_rows,train_stats,train_z,train_logits):
        rng=torch.cuda.get_rng_state();cpu_rng=torch.get_rng_state();result=dict(step=step,modes={})
        with torch.no_grad():
            for mode in meta['evaluation_modes']:
                if mode=='training_sample':rows,stats,zs,logits=train_rows,train_stats,train_z,train_logits
                else:
                    rows,stats,zs=forward(model,batch,mode,meta['fixed_eval_seeds'].get(mode))
                    logits=face_scores(rows)
                target=out/'evaluations'/mode;target.mkdir(parents=True,exist_ok=True)
                for i,uid in enumerate(UIDS):
                    np.savez_compressed(target/f'posterior_step{step}_{uid}.npz',mu=rows[0][i].detach().cpu().numpy(),
                        logvar=rows[1][i].detach().cpu().numpy(),z=zs[i].detach().cpu().numpy(),edge=rows[2][i].detach().cpu().numpy(),face=rows[3][i].detach().cpu().numpy(),
                        training_logits=logits[i].detach().float().cpu().numpy())
                full=h.full_reconstruction(rows,data,scales,step,target,[x.detach().float().cpu().numpy() for x in logits])
                full['posterior']=stats;result['modes'][mode]=full
        assert torch.equal(rng,torch.cuda.get_rng_state()) and torch.equal(cpu_rng,torch.get_rng_state())
        evals.append(result);write(out/'full_reconstruction.json',evals)
        return result

    with (out/'trace.jsonl').open('w',buffering=1) as log:
        for step in range(START,END+1):
            tick=time.monotonic();opt.zero_grad(set_to_none=True)
            before_rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state())
            with torch.set_grad_enabled(step<END):
                rows,posterior,zs=forward(model,batch,'sample')
                assert not torch.equal(before_rng['cuda'],torch.cuda.get_rng_state())
                if step<END:
                    assert all(m.requires_grad==(stage_index==1) for m in rows[0])
                    assert all(lv.requires_grad==(stage_index in [1,2]) for lv in rows[1])
                    assert all(z.requires_grad==(stage_index in [1,2]) for z in zs)
                logits=face_scores(rows);losses=[];kl_losses=[];metrics=[]
                for i,uid in enumerate(UIDS):
                    edge,edge_groups=h.soft4_loss(rows[2][i],positive_keys[i],chunk,scales['edge_logit_scale'])
                    ns,ms=h.archived.soft4_sums(logits[i],labels[i]);assert ns.dtype==ms.dtype==torch.float32 and not ms.requires_grad
                    face_loss=(ns/(ms+1e-8)).mean();losses.append(edge+face_loss)
                    mu=rows[0][i].float();lv=rows[1][i].float()
                    kl=topology.vae_kl_loss(mu,lv);kl_losses.append(kl)
                    kl_mu=.5*mu.square().mean();kl_variance=.5*(lv.exp()-1-lv).mean()
                    torch.testing.assert_close(kl,kl_mu+kl_variance)

                    with torch.no_grad():
                        _,counts=teacher.paper_edge_loss_all_pairs(rows[2][i],batch.edge_index[i],pair_chunk_size=chunk,positive_keys=positive_keys[i],counts_on_device=True,logit_scale=scales['edge_logit_scale'])
                        counts={k:int(v) for k,v in counts.items()};counts['f1']=2*counts['tp']/max(2*counts['tp']+counts['fp']+counts['fn'],1)
                        y=labels[i]>0;s=logits[i]>0
                        fc=dict(tp=int((y&s).sum()),tn=int((~y&~s).sum()),fp=int((~y&s).sum()),fn=int((y&~s).sum()))
                        fc['f1']=2*fc['tp']/max(2*fc['tp']+fc['fp']+fc['fn'],1)
                        bce=torch.nn.functional.binary_cross_entropy_with_logits(logits[i],labels[i],reduction='none')
                        fc.update(soft4=float(face_loss),balanced_bce=float(.5*(bce[y].mean()+bce[~y].mean())),soft_group_masses=ms.tolist())
                        counts.update(soft4=float(edge),balanced_bce=float(h.balanced_loss(rows[2][i],positive_keys[i],chunk,scales['edge_logit_scale'])))
                    metrics.append(dict(uid=uid,edge=counts,face_training=fc,posterior=posterior[i],kl=float(kl.detach()),kl_mu=float(kl_mu.detach()),kl_variance=float(kl_variance.detach()),
                        mu_relative_delta=None if previous_mu is None else norm(rows[0][i]-previous_mu[i])/(norm(previous_mu[i])+1e-12)))
                reco_loss=torch.stack(losses).mean();kl_loss=torch.stack(kl_losses).mean()
                loss=reco_loss+branch['beta']*kl_loss
            record=dict(phase=phase,step=step,objective=float(loss.detach()),reconstruction_loss=float(reco_loss.detach()),kl_loss=float(kl_loss.detach()),weighted_kl=float((branch['beta']*kl_loss).detach()),kl_weight=branch['beta'],rows=metrics,
                posterior_noise_rng_sha256=hashlib.sha256(before_rng['cuda'].numpy().tobytes()).hexdigest())
            if (step-START)%PERIOD==0:
                medians=[dict(uid=uid,logvar=stats['logvar']['median'],raw_logvar=stats['raw_logvar']['median']) for uid,stats in zip(UIDS,posterior)]
                if interval_initial is None:interval_initial=medians
                record['logvar_change_50']=[dict(uid=now['uid'],step=step,previous_step=None if interval_previous is None else step-PERIOD,
                    logvar_median=now['logvar'],raw_logvar_median=now['raw_logvar'],
                    delta_logvar_median=None if interval_previous is None else now['logvar']-interval_previous[i]['logvar'],
                    delta_raw_logvar_median=None if interval_previous is None else now['raw_logvar']-interval_previous[i]['raw_logvar'],
                    delta_logvar_median_since_start=now['logvar']-interval_initial[i]['logvar']) for i,now in enumerate(medians)]
                interval_previous=medians
                record['full_reconstruction']=evaluate(step,rows,posterior,zs,logits)
            if (step-START)%200==0 or step==END:
                check_frozen()
                saved=dict(cp,model=model.state_dict(),optimizer=opt.state_dict(),rng_state=before_rng,
                    diagnostic_run=dict(meta,completed_steps=step),diagnostic_metrics=record)
                torch.save(saved,out/f'checkpoint-{step}.pt')
            previous_mu=tuple(x.detach().clone() for x in rows[0])
            if step<END:
                if stage_index==2 and (step-START)%PERIOD==0:
                    gr=torch.autograd.grad(reco_loss,lvparams,retain_graph=True)
                    gk=torch.autograd.grad(kl_loss,lvparams,retain_graph=True)
                    nr=norm(torch.cat([g.flatten() for g in gr]));nk=norm(torch.cat([g.flatten() for g in gk]))
                    dot=float(sum((x.detach().double()*y.detach().double()).sum() for x,y in zip(gr,gk)))
                    record['gradient_components']=dict(reconstruction_l2=nr,kl_l2=nk,weighted_kl_l2=branch['beta']*nk,weighted_kl_to_reco=branch['beta']*nk/(nr+1e-30),cosine=dot/(nr*nk+1e-30))
                loss.backward()
                record['gradient_norms']=dict(encoder_mu=gradnorm(enc),decoder_edge=gradnorm(dec),face_head=gradnorm(face),logvar=gradnorm(lvparams))
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in active)
                assert all(p.grad is None and p._version==frozen_versions[id(p)] for p in frozen)
                record['logvar_grad_all_zero']=record['gradient_norms']['logvar']==0
                record['logvar_bias_gradient']=None if a.log_variance.bias.grad is None else quantiles(a.log_variance.bias.grad)
                record['preclip_grad_l2']=float(torch.nn.utils.clip_grad_norm_(active,1.,error_if_nonfinite=True))
                gn=record['gradient_norms'];total=record['preclip_grad_l2']
                record['gradient_norms_aggregated']=dict(encoder=gn['encoder_mu'],decoder_heads=(gn['decoder_edge']**2+gn['face_head']**2)**.5,logvar=gn['logvar'])
                record['gradient_square_share']={k:v*v/max(total*total,1e-30) for k,v in record['gradient_norms_aggregated'].items()}
                record['global_clipping_coefficient']=min(1.,1./max(total,1e-30))
                record['global_clipping_coefficient_torch_eps']=min(1.,1./(total+1e-6))
                record['gradient_norms_postclip']=dict(encoder_mu=gradnorm(enc),decoder_edge=gradnorm(dec),face_head=gradnorm(face),logvar=gradnorm(lvparams))
                for k,v in gn.items():
                    assert np.isclose(record['gradient_norms_postclip'][k],v*record['global_clipping_coefficient_torch_eps'],rtol=2e-5,atol=1e-10)
                record['gradient_computed']={k:any(p.requires_grad for p in ps) for k,ps in [('encoder_mu',enc),('decoder_edge',dec),('face_head',face),('logvar',lvparams)]}
                opt.step()
            record['all_parameters_trainable']=all(p.requires_grad for p in params)
            record['logvar_weight_delta_from_init_l2']=norm(a.log_variance.weight-initial_logvar['weight'].to(a.log_variance.weight.device))
            record['logvar_bias_delta_from_init']=quantiles(a.log_variance.bias-initial_logvar['bias'].to(a.log_variance.bias.device))
            record.update(seconds=time.monotonic()-started,iteration_seconds=time.monotonic()-tick)
            log.write(json.dumps(record,allow_nan=False)+'\n')
            if step<=START+2 or (step-START)%100==0:
                print(json.dumps(dict(event='train',step=step,objective=record['objective'],rows=metrics,seconds=record['seconds'])),flush=True)
    check_frozen()
    for p in params:assert float(opt.state[p]['step'])==initial_steps[names_by_id[id(p)]]+(500 if p.requires_grad else 0)
    assert probe.digest(start_path)==parent_sha
    for path,sha in meta['source_sha256'].items():assert probe.digest(path)==sha
    write(out/'complete.json',dict(steps=END,updates=500,seconds=time.monotonic()-started,final=record,
        optimizer_steps_verified=True,frozen_parameters_and_adam_bitwise_verified=True,active_optimizer_groups=active_groups,parameter_delta_l2={group:float(sum((p.detach().double()-cp['model'][names_by_id[id(p)]].to(p.device).double()).square().sum() for p in ps).sqrt()) for group,ps in [('encoder_mu',enc),('decoder_edge',dec),('face_head',face),('logvar',lvparams)]},original_sources_unchanged=True,logvar_weight_delta_from_init_l2=norm(a.log_variance.weight.cpu()-initial_logvar['weight']),
        logvar_bias_delta_from_init=quantiles(a.log_variance.bias.cpu()-initial_logvar['bias']),
        logvar_min=branch['logvar_min'],beta=branch['beta']))
    print(json.dumps(dict(event='complete',steps=END)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage-index',type=int,choices=range(len(STAGES)),required=True);p.add_argument('--gpu',type=int,default=0);p.add_argument('--lr-logvar',type=float,choices=[1e-4],default=1e-4)
    args=p.parse_args();main(args.stage_index,args.gpu,args.lr_logvar)
