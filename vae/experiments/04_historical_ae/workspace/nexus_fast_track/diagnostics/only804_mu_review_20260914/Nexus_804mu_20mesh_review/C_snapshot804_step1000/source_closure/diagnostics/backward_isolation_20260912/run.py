"""Retained-graph backward isolation; no updates, identical reconstruction objective."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import ast,fcntl,importlib.util,inspect,json,sys,textwrap,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('forward_c_backend',ROOT.parent/'forward_backend_20260911/run_C.py')
cbackend=importlib.util.module_from_spec(spec);spec.loader.exec_module(cbackend)
b=cbackend.b;c=b.c;torch=b.torch;np=b.np;flash=b.flash
from torch.utils.checkpoint import checkpoint
REPEATS=20
BRANCHES=['A_C_current','B_graph_backward','C_math_reference']
write=b.write

class DeterministicGather(torch.autograd.Function):
    @staticmethod
    def forward(ctx,nodes,source):
        ctx.save_for_backward(source);ctx.shape=nodes.shape
        return nodes[source]
    @staticmethod
    def backward(ctx,gradient):
        source,=ctx.saved_tensors
        with cbackend.deterministic_reduction():
            result=gradient.new_zeros(ctx.shape)
            result.index_add_(0,source,gradient)
        return result,None

class ReplaceGraphGather(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Subscript(self,node):
        if isinstance(node.value,ast.Name) and node.value.id=='nodes' and isinstance(node.slice,ast.Name) and node.slice.id=='source':
            self.count+=1
            return ast.copy_location(ast.Call(func=ast.Name(id='deterministic_gather',ctx=ast.Load()),args=[node.value,node.slice],keywords=[]),node)
        return self.generic_visit(node)

def install_graph_backward(model,out):
    # Start from the exact archived GraphSAGE body; apply same C forward reduction scope.
    original=next(iter(model.autoencoder.encoder_blocks)).graph.forward
    # install_graph already replaced the instance method: use saved original source, not generated inspect path.
    source=json.loads((out/'C_graph_only/graph_change.json').read_text())['original_graph_forward']
    tree=ast.parse(source);forward_scope=cbackend.ScopeReduction();tree=forward_scope.visit(tree)
    backward_scope=ReplaceGraphGather();tree=backward_scope.visit(tree);ast.fix_missing_locations(tree)
    assert forward_scope.count==2 and backward_scope.count==1
    namespace=dict(original.__func__.__globals__)
    namespace.update(deterministic_gather=DeterministicGather.apply,deterministic_reduction=cbackend.deterministic_reduction)
    exec(compile(tree,str(out/'graph_backward.py'),'exec'),namespace)
    for block in model.autoencoder.encoder_blocks:block.graph.forward=types.MethodType(namespace['forward'],block.graph)
    (out/'graph_backward.py').write_text(ast.unparse(tree)+'\n')

def groups(names):
    group={'encoder_all':[i for i,n in enumerate(names) if not n.startswith('boundary/')]}
    blocks=sorted({n.split('.')[1] for n in names if n.startswith('encoder_blocks.')},key=int)
    group['graphsage_parameters']=[i for i,n in enumerate(names) if '.graph.' in n]
    group['attention_parameters']=[i for i,n in enumerate(names) if '.self_attn.' in n]
    group['mu_head']=[i for i,n in enumerate(names) if n.startswith('mu.')]
    for k in blocks:
        prefix=f'encoder_blocks.{k}.'
        group[f'encoder_block_{int(k):02d}']=[i for i,n in enumerate(names) if n.startswith(prefix)]
        group[f'graph_block_{int(k):02d}']=[i for i,n in enumerate(names) if n.startswith(prefix+'graph.')]
        group[f'attention_block_{int(k):02d}']=[i for i,n in enumerate(names) if n.startswith(prefix+'transformer.self_attn.')]
    for i,n in enumerate(names):
        if n.startswith('boundary/'):group[n]=[i]
    assert all(group.values())
    return group

def compare(g,base,group):
    # FP64 statistics; norms and cosine refer to concatenated parameter vectors per group.
    values=[]
    for x,y in zip(g,base):
        a=x.detach().double();v=y.double()
        values.append(torch.stack([a.square().sum(),v.square().sum(),(a-v).square().sum(),(a*v).sum()]))
    values=torch.stack(values).cpu().numpy();result={}
    for name,indices in group.items():
        n2,b2,d2,dot=values[indices].sum(0)
        result[name]=dict(norm=float(np.sqrt(n2)),baseline_norm=float(np.sqrt(b2)),relative_l2=float(np.sqrt(d2)/(np.sqrt(b2)+1e-30)),cosine=None if n2==0 or b2==0 else float(np.clip(dot/np.sqrt(n2*b2),-1,1)),exact=bool(d2==0))
    return result

def main(branch):
    out=ROOT/branch;out.mkdir(exist_ok=True);assert not (out/'trace.jsonl').exists()
    cp,model,batch,enc,data=c.setup(0)
    cbackend.ROOT=out;(out/'C_graph_only').mkdir(exist_ok=True);cbackend.install_graph(model)
    if branch!='A_C_current':install_graph_backward(model,out)
    if branch=='C_math_reference':
        def math_with_recompute(attention,tokens,cu,maximum):
            return checkpoint(b.math_attention,attention,tokens,cu,maximum,use_reentrant=False,preserve_rng_state=False)
        flash._flash_self_attention=math_with_recompute
    mu=[];hook=model.autoencoder.mu.register_forward_hook(lambda mod,args,value:mu.append(value))
    rows,_,_=c.m.forward(model,batch,'sample',c.SEEDS);hook.remove()
    scales=model.scoring_contract();loss,parts,membership=c.objective(rows,data,scales)
    for i,uid in enumerate(c.UIDS):
        np.savez_compressed(out/(uid+'_forward.npz'),mu=rows[0][i].detach().cpu().numpy(),logvar=rows[1][i].detach().cpu().numpy(),edge_embedding=rows[2][i].detach().cpu().numpy(),face_embedding=rows[3][i].detach().cpu().numpy(),**membership[i],epsilon=model.autoencoder._diagnostic_eps[i].cpu().numpy())
    if branch=='B_graph_backward':
        for uid in c.UIDS:
            a=np.load(ROOT/'A_C_current'/(uid+'_forward.npz'));z=np.load(out/(uid+'_forward.npz'))
            assert all(np.array_equal(a[k],z[k]) for k in a.files),'Graph-backward-only branch changed forward'
        assert float(loss.detach())==json.loads((ROOT/'A_C_current/manifest.json').read_text())['loss']
    elif branch=='C_math_reference':
        for uid in c.UIDS:
            a=np.load(ROOT/'A_C_current'/(uid+'_forward.npz'));z=np.load(out/(uid+'_forward.npz'));assert np.array_equal(a['epsilon'],z['epsilon'])
    edge=c.edge_logits(rows[2][1],torch.tensor([[300,383]],device='cuda'),scales)[0]
    face=c.face_logits(rows[3][1],torch.tensor([[385,386,388]],device='cuda'),scales)[0]
    # edge is a GT non-edge, face is GT; signed margins as in the line search.
    edges=data[1]['edges'];edges=edges.T if edges.shape[0]==2 else edges
    assert not np.any((np.sort(edges,axis=1)==[300,383]).all(1))
    assert np.any((np.sort(data[1]['positive'],axis=1)==[385,386,388]).all(1))
    targets={'reconstruction':loss,'face_385_386_388':face,'edge_300_383':-edge}
    names=list(enc)+['boundary/mu_output']+['boundary/edge_embedding_'+u for u in c.UIDS]+['boundary/face_embedding_'+u for u in c.UIDS]
    inputs=tuple(enc.values())+(mu[0],)+tuple(rows[2])+tuple(rows[3])
    group=groups(names)
    write(out/'manifest.json',dict(branch=branch,checkpoint_sha=c.m.START_SHA,source_sha256=cp['diagnostic_run']['source_sha256'],gpu=0,seeds=c.SEEDS,mode='sample',loss=float(loss.detach()),loss_parts=parts,repeats=REPEATS,targets={k:float(v.detach()) for k,v in targets.items()},groups={k:[names[i] for i in ix] for k,ix in group.items()},graph_backward='deterministic nodes[source] VJP index_add' if branch!='A_C_current' else 'original',attention='FP32 SDPA math, nonreentrant activation checkpoint to bound memory' if branch=='C_math_reference' else 'unchanged BF16 Flash forward/backward',other_backward_ops='unchanged, including scoring/embedding indexing; reference does not silently enable global deterministic algorithms',retained_graph=True,optimizer=False,script_sha=c.m.probe.digest(Path(__file__))))
    print(branch,'forward loss',float(loss.detach()),flush=True)
    started=time.monotonic()
    with (out/'trace.jsonl').open('w',buffering=1) as f:
        for target,scalar in targets.items():
            base=None
            for k in range(REPEATS):
                rng=torch.cuda.get_rng_state();g=torch.autograd.grad(scalar,inputs,retain_graph=True,allow_unused=True)
                unused=[names[i] for i,x in enumerate(g) if x is None]
                if target=='reconstruction':assert not unused,unused
                g=tuple(torch.zeros_like(inputs[i]) if x is None else x for i,x in enumerate(g))
                if base is None:
                    base=tuple(x.detach().clone() for x in g)
                    if target=='reconstruction':torch.save({n:x.cpu() for n,x in zip(names,base)},out/'reconstruction_g0.pt')
                result=compare(g,base,group)
                assert torch.equal(rng,torch.cuda.get_rng_state())
                f.write(json.dumps(dict(target=target,repeat=k,groups=result,unused=unused,seconds=time.monotonic()-started))+'\n')
                if k in [0,1,4,9,19]:print(branch,target,k,result['encoder_all'],flush=True)
                del g
            del base
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(p.cpu(),cp['model'][n]) for n,p in model.named_parameters())
    write(out/'complete.json',dict(repeats=REPEATS,targets=list(targets),parameters_unchanged=True,no_optimizer=True,retained_graph=True,seconds=time.monotonic()-started))

if __name__=='__main__':
    lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert sys.argv[1] in BRANCHES;main(sys.argv[1])
