"""Exact 00 backend/loss helpers copied from the verified cast experiment."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,importlib.util,json,time,fcntl
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'encoder_gradient_linesearch_20260911'))
import common as c
import torch
import numpy as np
spec=importlib.util.spec_from_file_location('c_backend',ROOT.parent/'forward_backend_20260911/run_C.py')
backend=importlib.util.module_from_spec(spec);spec.loader.exec_module(backend)
spec2=importlib.util.spec_from_file_location('backward_backend',ROOT.parent/'backward_isolation_20260912/run.py')
bw=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(bw)
def install_deterministic(model):
    backend.install_graph(model)
    bw.install_graph_backward(model,ROOT)
original_flash=backend.b.flash.flash_attn_varlen_qkvpacked_func
def deterministic_flash(*args,**kwargs):
    kwargs['deterministic']=True
    return original_flash(*args,**kwargs)
backend.b.flash.flash_attn_varlen_qkvpacked_func=deterministic_flash
BASE=ROOT
BRANCH='00'
assert BRANCH in ['00','10','01','11']
INPUT_CAST=BRANCH[0]=='1';OUTPUT_CAST=BRANCH[1]=='1'
ROOT=BASE;ROOT.mkdir(exist_ok=True)
backend.ROOT=ROOT
(ROOT/'C_graph_only').mkdir(exist_ok=True)
write=c.write
R=[0.,1e-10,3e-10,1e-9,3e-9,1e-8,3e-8,1e-7,3e-7,1e-6]


# Only remove membership detach; preserve archived reduction/accumulation order.
import inspect,csv
old_weights=c.weights;sums_globals=c.m.h.soft4_loss.__globals__;old_sums=sums_globals['soft4_sums']
src=inspect.getsource(old_sums).replace('s.sigmoid().detach()', 's.sigmoid()').replace('    assert not weights.requires_grad\n','')
ns=dict(old_sums.__globals__);exec(src,ns);full_sums=ns['soft4_sums']
src_weights=inspect.getsource(old_weights).replace('logits.detach().float()', 'logits.float()')
ns=dict(old_weights.__globals__);exec(src_weights,ns);full_weights=ns['weights']
(ROOT/'loss_changes.txt').write_text(src+'\n'+src_weights)
def full_objective(rows,data,scales,base=None):
    if base is not None:return c.objective(rows,data,scales,base)
    c.weights=full_weights;sums_globals['soft4_sums']=full_sums
    try:return c.objective(rows,data,scales)
    finally:c.weights=old_weights;sums_globals['soft4_sums']=old_sums

def frozen_rows(model,batch):
    # Keep grad-mode/parameter flags identical to original step0 kernel dispatch, then detach.
    rows,stats,zs=c.m.forward(model,batch,'sample',c.SEEDS)
    result=tuple(tuple(x.detach() for x in group) for group in rows)
    del rows,zs
    return result

def norm(xs):return float(sum(x.detach().double().square().sum() for x in xs).sqrt())

def repeat_check(rows,rows2,tables,scales):
    result={}
    for i,uid in enumerate(c.UIDS):
        for j,name in enumerate(['mu','logvar','edge_embedding','face_embedding']):
            result[uid+'/'+name]=float((rows[j][i]-rows2[j][i]).abs().max())
        t=tables[i];ls=[]
        for k in range(0,len(t['edge_pairs']),65536):ls.append(c.edge_logits(rows2[2][i],torch.as_tensor(t['edge_pairs'][k:k+65536],device='cuda'),scales).cpu().numpy())
        el=np.concatenate(ls)
        fl=c.face_logits(rows2[3][i],torch.as_tensor(t['face_triples'],device='cuda'),scales).cpu().numpy()
        result[uid+'/edge_logits']=float(np.max(np.abs(el-t['edge_logits'])))
        result[uid+'/face_logits']=float(np.max(np.abs(fl-t['face_logits'])))
    return result


from contextlib import contextmanager
from torch.nn.attention import sdpa_kernel,SDPBackend
from torch.utils.checkpoint import checkpoint,set_checkpoint_early_stop
import torch.nn.functional as F
AUDIT={'contexts':{},'completed_calls':{},'dtype_routes':{},'attention_semantics':dict(scale='default 1/sqrt(head_dim)',mask=None,causal=False,dropout=0.,packed='each cu_seqlens interval independently'), 'checkpoint':'nonreentrant; early_stop=False; identical explicit precision context on forward and recompute'}
STAGE=[]
@contextmanager
def precision_context(stage):
    with torch.autocast(device_type='cuda',enabled=False),sdpa_kernel(SDPBackend.MATH):
        assert not torch.is_autocast_enabled() and not torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cuda.math_sdp_enabled() and not torch.backends.cuda.flash_sdp_enabled() and not torch.backends.cuda.mem_efficient_sdp_enabled()
        AUDIT['contexts'][stage]=AUDIT['contexts'].get(stage,0)+1
        STAGE.append(stage)
        try:yield
        finally:STAGE.pop()
class ReusablePrecisionContext:
    # A retained graph re-enters the recompute context on each backward.
    def __init__(self,stage):self.stage=stage
    def __enter__(self):
        self.active=precision_context(self.stage)
        return self.active.__enter__()
    def __exit__(self,*exc):return self.active.__exit__(*exc)
def core(qkv32,cu,maximum):
    assert qkv32.dtype==torch.float32
    bounds=cu.cpu().tolist();outputs=[]
    for start,end in zip(bounds[:-1],bounds[1:]):
        q,k,v=[qkv32[start:end,j].transpose(0,1).unsqueeze(0) for j in range(3)]
        o=F.scaled_dot_product_attention(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,scale=None)
        assert all(x.dtype==torch.float32 for x in [q,k,v,o])
        outputs.append(o.squeeze(0).transpose(0,1))
    out=torch.cat(outputs,dim=0)
    stage=STAGE[-1];AUDIT['completed_calls'][stage]=AUDIT['completed_calls'].get(stage,0)+1
    AUDIT['dtype_routes'][stage]=['core_input:float32','math_output:float32']
    return out
ATTENTION_NAMES={};SEEN=set()
def unified_attention(attention,tokens,cu,maximum):
    assert tokens.dtype==torch.float32
    SEEN.add(ATTENTION_NAMES[id(attention)])
    head_count=attention.num_heads;hidden_dim=attention.embed_dim;head_dim=hidden_dim//head_count
    with torch.autocast(device_type='cuda',enabled=False):
        qkv=F.linear(tokens,attention.in_proj_weight,attention.in_proj_bias).reshape(len(tokens),3,head_count,head_dim)
        assert qkv.dtype==torch.float32
        if INPUT_CAST:qkv=qkv.to(torch.bfloat16).to(torch.float32)
        qkv=qkv.contiguous()
        with set_checkpoint_early_stop(False):
            attended=checkpoint(core,qkv,cu,maximum,use_reentrant=False,preserve_rng_state=False,context_fn=lambda:(ReusablePrecisionContext('forward'),ReusablePrecisionContext('recompute')))
        assert attended.dtype==torch.float32
        if OUTPUT_CAST:attended=attended.to(torch.bfloat16).to(torch.float32)
        return F.linear(attended.reshape(len(tokens),hidden_dim),attention.out_proj.weight,attention.out_proj.bias)
backend.b.flash._flash_self_attention=unified_attention
