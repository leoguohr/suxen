"""C: determinism scoped to the two GraphSAGE index_add_ reductions only."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import ast,contextlib,fcntl,importlib.util,inspect,json,textwrap,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('backend_reference',ROOT/'run.py')
b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)
torch=b.torch;np=b.np;c=b.c

@contextlib.contextmanager
def deterministic_reduction():
    previous=torch.are_deterministic_algorithms_enabled()
    warn=torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True,warn_only=False)
    try:yield
    finally:torch.use_deterministic_algorithms(previous,warn_only=warn)

class ScopeReduction(ast.NodeTransformer):
    def __init__(self):self.count=0
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='index_add_':
            self.count+=1
            return ast.copy_location(ast.With(items=[ast.withitem(context_expr=ast.Call(func=ast.Name(id='deterministic_reduction',ctx=ast.Load()),args=[],keywords=[]))],body=[node]),node)
        return self.generic_visit(node)

def install_graph(model):
    assert not torch.are_deterministic_algorithms_enabled()
    methods=set()
    for block in model.autoencoder.encoder_blocks:
        original=block.graph.forward.__func__;methods.add(original)
    assert len(methods)==1
    original=next(iter(methods));source=textwrap.dedent(inspect.getsource(original))
    transformer=ScopeReduction();tree=transformer.visit(ast.parse(source));ast.fix_missing_locations(tree)
    assert transformer.count==2,'Expected neighbor_sum and count index_add_ only'
    patched=ast.unparse(tree)
    namespace=dict(original.__globals__);namespace['deterministic_reduction']=deterministic_reduction
    exec(compile(tree,str(ROOT/'C_graph_only/scoped_graph_forward.py'),'exec'),namespace)
    for block in model.autoencoder.encoder_blocks:block.graph.forward=types.MethodType(namespace['forward'],block.graph)
    (ROOT/'C_graph_only/scoped_graph_forward.py').write_text(patched+'\n')
    b.write(ROOT/'C_graph_only/graph_change.json',dict(scope='Only the two GraphSAGE index_add_ calls; flags restored immediately before projections and attention',blocks=len(model.autoencoder.encoder_blocks),changed_reductions_per_block=2,global_deterministic_outside_graph=False,flash_replaced=False,original_graph_forward=source,script_sha256=c.m.probe.digest(Path(__file__))))

def actual_reconstruction(face_embedding,ep,el,ft,fy,scales,d):
    assert not torch.are_deterministic_algorithms_enabled()
    n=len(d['vertices']);adj=[set() for _ in range(n)]
    for a,z in ep[el.detach().cpu().numpy()>0]:adj[a].add(int(z))
    triangles=[]
    for a in range(n):
        for z in sorted(adj[a]):triangles.extend((a,z,k) for k in sorted(adj[a].intersection(adj[z])))
    triangles=np.asarray(triangles,dtype=np.int64).reshape(-1,3)
    # Directed upper-triangle adjacency gives sorted, unique a<b<c triples.
    if len(triangles):assert np.all(triangles[:,0]<triangles[:,1]) and np.all(triangles[:,1]<triangles[:,2])
    labels=np.isin(c.m.probe.keys(triangles,n),c.m.probe.keys(d['positive'],n))
    logits=c.face_logits(face_embedding,torch.as_tensor(triangles,device='cuda'),scales).cpu().numpy() if len(triangles) else np.empty(0,np.float32)
    metric=c.metrics(labels,logits,len(d['positive']))
    return dict(face=metric,candidate_count=len(triangles),same_candidates_as_baseline=bool(np.array_equal(triangles,ft)),gt_faces_missing_from_edge_candidates=int(len(d['positive'])-labels.sum()),errors=[dict(vertices=triangles[j].tolist(),label=int(labels[j]),logit=float(logits[j])) for j in np.flatnonzero((logits>0)!=labels)])

# Reuse the exact A/B runner, recorder, noise and scoring; add only C and actual-candidate validation.
source=inspect.getsource(b.main)
old="    if backend=='B_fp32_math':"
assert source.count(old)==1
source=source.replace(old,"    if backend=='C_graph_only':\n        install_graph(model)\n    elif backend=='B_fp32_math':")
old="                metrics.append(dict(uid=uid,edge=c.metrics(ey,el.cpu().numpy()),face_fixed_candidates=c.metrics(fy,fl.cpu().numpy())))"
assert source.count(old)==1
source=source.replace(old,old+"\n                metrics[-1]['actual_reconstruction']=actual_reconstruction(rows[3][i],ep,el,ft,fy,scales,data[i])")
source=source.replace("precision='A: original FP32 except BF16 Flash IO; B: FP32 SDPA math, deterministic algorithms enabled including supported graph reductions'", "precision='C: original FP32 with BF16 Flash IO; only GraphSAGE index_add_ determinism changed'")
b.__dict__.update(install_graph=install_graph,actual_reconstruction=actual_reconstruction)
exec(compile(source,str(ROOT/'run_C.py'),'exec'),b.__dict__)
if __name__=='__main__':
    lock=(ROOT/'run.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    b.main('C_graph_only')
