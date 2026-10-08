"""Export fixed post-LayerNorm hidden and original shared head, without network updates."""
import json
from pathlib import Path
from runtime import ROOT, T, np, b, c, setup, sha, write, tensor_hash

def main():
    cfg=json.loads((ROOT/'config.json').read_text())
    source=Path(cfg['source_checkpoint']);free=Path(cfg['free_probe'])
    assert sha(source)==cfg['source_sha256']
    assert not (ROOT/'snapshots').exists(),'Preserve previous exports'
    model,opt,groups,uids,pools,forward,capture,args=setup()
    T.use_deterministic_algorithms(True)
    cp=T.load(source,map_location='cpu',mmap=True,weights_only=False)
    model.load_state_dict(cp['model'],strict=True);del cp,opt
    before=tensor_hash((k,v) for k,v in model.state_dict().items() if T.is_tensor(v))
    metadata=json.dumps({k:v for k,v in model.state_dict().items() if not T.is_tensor(v)},sort_keys=True)
    head=model.autoencoder.edge_embedding
    assert tuple(head.weight.shape)==tuple(cfg['head_shape']) and head.bias is not None
    (ROOT/'snapshots').mkdir()
    np.savez(ROOT/'snapshots/head_initial.npz',weight=head.weight.detach().cpu().numpy(),bias=head.bias.detach().cpu().numpy())
    grabbed={}
    pre=head.register_forward_pre_hook(lambda m,x:grabbed.update(hidden=x[0]))
    post=head.register_forward_hook(lambda m,x,y:grabbed.update(raw=y))
    records=[];g=c.m.h.soft4_loss.__globals__;old=g['soft4_sums']
    try:
        g['soft4_sums']=b.full_sums
        for uid,n in zip(cfg['uids'],cfg['vertices']):
            base=free/'snapshots'/uid
            summary=json.loads((base/'summary.json').read_text())
            assert summary['checkpoint_sha256']==cfg['source_sha256'] and summary['vertices']==n
            dst=ROOT/'snapshots'/uid;dst.mkdir()
            for name in ['edge_all_pairs.npz','representations_and_gradients.npz','summary.json','source_evaluation.json','effective_code']:
                (dst/name).symlink_to(base/name)
            rows=forward(uid);hidden=grabbed['hidden'];raw=grabbed['raw'];center=rows[2][0]
            assert hidden.shape==(n,1024) and hidden.dtype==T.float32
            assert T.equal(hidden,capture['hidden'])
            arrays=np.load(base/'representations_and_gradients.npz')
            assert np.array_equal(raw.detach().cpu().numpy(),arrays['edge_head_raw'])
            assert np.array_equal(center.detach().cpu().numpy(),arrays['edge_embedding_scoring'])
            contract=json.loads((base/'effective_code/runtime_contract.json').read_text())
            assert contract['scales']==model.scoring_contract()
            edges=np.sort(pools[uid]['edges'].T,axis=1)
            keys=T.as_tensor(np.unique(edges@np.array([n,1])),device='cuda',dtype=T.long)
            loss,_=c.m.h.soft4_loss(center,keys,contract['edge_chunk'],contract['scales']['edge_logit_scale'])
            actual_grad=T.autograd.grad(loss*cfg['outer_coefficient'],tuple(head.parameters()))
            # Same nn.Linear on fixed features, preserving shape, dtype, and bias placement.
            reproduced=head(hidden.detach());z=reproduced-reproduced.mean(0,keepdim=True)
            loss2,_=c.m.h.soft4_loss(z,keys,contract['edge_chunk'],contract['scales']['edge_logit_scale'])
            reproduced_grad=T.autograd.grad(loss2*cfg['outer_coefficient'],tuple(head.parameters()))
            record=dict(uid=uid,hidden_shape=list(hidden.shape),raw_bitwise=T.equal(raw,reproduced),
                centered_bitwise=T.equal(center,z),loss_bitwise=T.equal(loss,loss2),
                weight_gradient_bitwise=T.equal(actual_grad[0],reproduced_grad[0]),
                bias_gradient_bitwise=T.equal(actual_grad[1],reproduced_grad[1]),
                edge_loss=float(loss),expected_edge_loss=summary['parts']['edge'])
            write(dst/'hidden_verification.json',record)
            assert all(record[k] for k in ['raw_bitwise','centered_bitwise','loss_bitwise','weight_gradient_bitwise','bias_gradient_bitwise'])
            assert float(loss)==summary['parts']['edge']
            np.save(dst/'hidden.npy',hidden.detach().cpu().numpy())
            np.savez(dst/'head_gradient.npz',weight=actual_grad[0].cpu().numpy(),bias=actual_grad[1].cpu().numpy())
            records.append(record);print('EXPORTED',json.dumps(record),flush=True)
            del rows,hidden,raw,center,reproduced,z,loss,loss2,actual_grad,reproduced_grad
            capture.clear();grabbed.clear();T.cuda.empty_cache()
    finally:
        pre.remove();post.remove();g['soft4_sums']=old
    assert before==tensor_hash((k,v) for k,v in model.state_dict().items() if T.is_tensor(v))
    assert metadata==json.dumps({k:v for k,v in model.state_dict().items() if not T.is_tensor(v)},sort_keys=True)
    assert sha(source)==cfg['source_sha256']
    write(ROOT/'export_complete.json',dict(records=records,source_checkpoint=str(source),source_sha256=cfg['source_sha256'],
        original_model_unchanged=True,network_updates=0,weight_norm=float(head.weight.detach().double().norm()),
        bias_norm=float(head.bias.detach().double().norm()),one_shared_head=True))

if __name__=='__main__':main()
