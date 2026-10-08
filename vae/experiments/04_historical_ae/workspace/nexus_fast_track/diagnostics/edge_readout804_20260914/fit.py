"""Offline least-squares readout from fixed decoder hidden; no optimizer."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def norm(x):
    return float(x.double().norm())


def residual(x, target):
    d=x.double()-target.double()
    return dict(relative_frobenius=norm(d)/norm(target), mse=float(d.square().mean()),
                rmse=float(d.square().mean().sqrt()), max_abs=float(d.abs().max()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--snapshot',type=Path,required=True)
    p.add_argument('--target',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'result.json').exists()
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    torch.use_deterministic_algorithms(True)
    tensors=np.load(args.snapshot/'representations_and_gradients.npz')
    head=np.load(args.snapshot/'input_posterior_output_head_weights.npz')
    source_edges=np.load(args.snapshot/'edge_all_pairs.npz')
    target_file=np.load(args.target)
    contract=json.loads((args.snapshot/'effective_code/runtime_contract.json').read_text())
    spec=importlib.util.spec_from_file_location('original_scoring',args.snapshot/'effective_code/effective_loss_and_scoring.py')
    scoring=importlib.util.module_from_spec(spec);spec.loader.exec_module(scoring)
    h=torch.from_numpy(tensors['decoder_hidden_after_final_ln'].copy())
    target=torch.from_numpy(target_file['edge_embedding_scoring'].copy())
    old_w=torch.from_numpy(head['edge_embedding__weight'].copy())
    bias=torch.from_numpy(head['edge_embedding__bias'].copy())
    assert h.shape==(804,1024) and target.shape==(804,32) and old_w.shape==(32,1024)
    assert np.array_equal(target_file['pairs'],source_edges['pairs'])
    assert np.array_equal(target_file['labels'],source_edges['labels'])
    paths=[args.snapshot/'representations_and_gradients.npz',args.snapshot/'input_posterior_output_head_weights.npz',
           args.snapshot/'edge_all_pairs.npz',args.snapshot/'effective_code/effective_loss_and_scoring.py',args.target]
    hashes={str(path):sha(path) for path in paths}
    hd=h.double();ed=target.double();hc=hd-hd.mean(0,keepdim=True)
    # Moore-Penrose solution at the conventional FP64 rank tolerance. No ridge or sweep.
    u,s,vh=torch.linalg.svd(hc,full_matrices=False)
    rcond=max(hc.shape)*torch.finfo(torch.float64).eps
    keep=s>s[0]*rcond
    w64=((vh[keep].T/s[keep])@(u[:,keep].T@ed)).T.contiguous()
    w32=w64.float()
    raw64=hd@w64.T+bias.double()
    fit64=raw64-raw64.mean(0,keepdim=True)
    spectrum_h=torch.linalg.svdvals(hd)
    spectra=dict(centered_h=s.numpy(),h=spectrum_h.numpy())
    rank64=int(keep.sum())
    rank32=int((s>s[0]*max(hc.shape)*torch.finfo(torch.float32).eps).sum())
    spectrum=dict(shape=list(h.shape),centered_rank_fp64=rank64,
                  fp64_relative_cutoff=rcond,fp64_absolute_cutoff=float(s[0]*rcond),
                  effective_rank_fp32_tolerance=rank32,
                  fp32_relative_cutoff=max(hc.shape)*torch.finfo(torch.float32).eps,
                  largest_singular=float(s[0]),smallest_retained_singular=float(s[keep][-1]),
                  last_singular=float(s[-1]),condition_retained=float(s[0]/s[keep][-1]),
                  singular_top10=s[:10].tolist(),singular_bottom10=s[-10:].tolist(),
                  target_mean_l2=norm(ed.mean(0)),h_centered_frobenius=norm(hc))
    hg=h.cuda();bg=bias.cuda();wg=w32.cuda()
    pairs=torch.from_numpy(source_edges['pairs']).cuda()
    y=torch.from_numpy(source_edges['labels']).cuda()
    scale=contract['scales']['edge_logit_scale']

    def score(e):
        logits=scoring.first_order_interval(e[pairs[:,0]],e[pairs[:,1]])*scale
        num,mass=scoring.soft4_sums(logits,y)
        loss=(num/(mass+1e-8)).mean()
        pred=logits>0
        tp=int((pred&y).sum());fp=int((pred&~y).sum());fn=int((~pred&y).sum())
        return logits,dict(tp=tp,fp=fp,fn=fn,tn=int((~pred&~y).sum()),f1=2*tp/(2*tp+fp+fn),
                           perfect=fp==0 and fn==0,edge_soft4=float(loss),objective_edge_over4=float(loss/4),
                           min_margin_gt=float(logits[y].min()),min_margin_non_gt=float((-logits[~y]).min()))

    with torch.no_grad():
        baseline_raw=F.linear(hg,old_w.cuda(),bg)
        baseline_e=baseline_raw-baseline_raw.mean(0,keepdim=True)
        baseline_l,baseline_metrics=score(baseline_e)
        target_l,target_metrics=score(target.cuda())
        assert torch.equal(target_l,torch.from_numpy(target_file['logits']).cuda()) and target_metrics['perfect']
        raw32=F.linear(hg,wg,bg)
        fit32=raw32-raw32.mean(0,keepdim=True)
        logits,metrics=score(fit32)
        repeat_raw=F.linear(hg,wg,bg)
        repeat_e=repeat_raw-repeat_raw.mean(0,keepdim=True)
        repeat_l,_=score(repeat_e)
        assert torch.equal(fit32,repeat_e) and torch.equal(logits,repeat_l)
    weights=dict(original_frobenius=norm(old_w),fit_fp64_frobenius=norm(w64),
                 fit_fp32_frobenius=norm(w32),norm_ratio=norm(w32)/norm(old_w),
                 original_max_abs=float(old_w.abs().max()),fit_max_abs=float(w32.abs().max()),
                 fp32_rounding_relative_frobenius=norm(w32.double()-w64)/norm(w64),
                 bias='original bias retained; no new intercept fitted',bias_frobenius=norm(bias))
    result=dict(spectrum=spectrum,weights=weights,fp64_residual=residual(fit64,ed),
                fp32_residual=residual(fit32.cpu(),target),
                baseline=baseline_metrics,target=target_metrics,fitted_fp32=metrics,
                baseline_raw_vs_export=residual(baseline_raw.cpu(),torch.from_numpy(tensors['edge_head_raw'])),
                baseline_logits_vs_export=residual(baseline_l.cpu(),torch.from_numpy(source_edges['logits'])),
                repeated_fp32_forward_exact=True,optimizer_updates=0,
                interpretation='Fixed-H least-squares fit to this one target; failure does not rule out other valid embeddings.')
    manifest=dict(snapshot=str(args.snapshot),target=str(args.target),source_sha256=hashes,
                  script_sha256=sha(__file__),variable='one shared original-shape Linear(1024,32) weight',
                  h='decoder_hidden_after_final_ln from only804_mu step1000',
                  target_embedding='centered successful free Edge step2000 representation, used as stored',
                  solve='CPU FP64 SVD minimum-norm least squares on FP64 centered H',
                  rank_tolerance='max(m,n)*eps64*sigma_max; no ridge; no hyperparameter sweep',
                  evaluation='CUDA FP32 F.linear(H,W,b_original), FP32 center, exact original Edge scoring',
                  edge_logit_scale=scale,threshold=0,space_time_dims=[16,16],pairs=len(pairs),
                  tf32=False,autocast=False,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                  no_optimizer=True,original_checkpoint_modified=False)
    np.savez_compressed(args.out/'readout.npz',weight_fp64=w64.numpy(),weight_fp32=w32.numpy(),
                        original_weight=old_w.numpy(),bias=bias.numpy(),hidden=h.numpy(),
                        target=target.numpy(),fit_fp64=fit64.numpy(),fit_fp32=fit32.cpu().numpy(),
                        raw_fp32=raw32.cpu().numpy(),logits_fp32=logits.cpu().numpy(),
                        pairs=source_edges['pairs'],labels=source_edges['labels'])
    np.savez_compressed(args.out/'singular_values.npz',**spectra)
    torch.save(dict(weight=w32,bias=bias,manifest=manifest,metrics=metrics),args.out/'edge_head_candidate.pt')
    assert {str(path):sha(path) for path in paths}==hashes
    write(args.out/'manifest.json',manifest);write(args.out/'result.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
