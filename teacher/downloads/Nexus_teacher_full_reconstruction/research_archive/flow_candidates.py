"""Weight-compatible candidate flow semantics; hypothesis testing, not original code."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from teacher_ae import fourier_positions

def time_embedding(t, scale=1000., order='cs', den=32):
    f=torch.exp(-math.log(10000)*torch.arange(32,device=t.device,dtype=t.dtype)/den)
    a=t.reshape(-1,1)*scale*f
    return torch.cat((a.cos(),a.sin()) if order=='cs' else (a.sin(),a.cos()),-1)

def rope(x,xyz,mode='pair',scale=1.,freq_kind='inverse'):
    if mode=='none':return x
    # B,H,N,D -> [B,H,N,3,D/3]
    d=x.shape[-1]; a=d//3
    r=x.reshape(*x.shape[:-1],3,a)
    freq=torch.exp(-math.log(10000)*torch.arange(0,a,2,device=x.device,dtype=x.dtype)/a)
    if freq_kind=='pow2':freq=2.**torch.arange(a//2,device=x.device,dtype=x.dtype)
    if freq_kind=='linear':freq=torch.arange(1,a//2+1,device=x.device,dtype=x.dtype)
    angle=xyz[:,None,:,:,None]*scale*freq
    c,s=angle.cos(),angle.sin()
    if mode=='pair':
        u,v=r[...,0::2],r[...,1::2]
        return torch.stack((u*c-v*s,u*s+v*c),-1).flatten(-2).reshape_as(x)
    u,v=r[...,:a//2],r[...,a//2:]
    return torch.cat((u*c-v*s,u*s+v*c),-1).reshape_as(x)

class DiTBlock(nn.Module):
    def __init__(self,w=144):
        super().__init__();self.qkv=nn.Linear(w,3*w);self.out=nn.Linear(w,w)
        self.ff=nn.Sequential(nn.Linear(w,4*w),nn.GELU(),nn.Linear(4*w,w))
        self.ada=nn.Sequential(nn.SiLU(),nn.Linear(w,6*w))
    def forward(self,x,c,xyz,mask,op):
        ada=self.ada(c).chunk(6,-1)
        sh1,sc1,g1,sh2,sc2,g2=ada
        if op.get('ada_order')=='scale_shift':sc1,sh1,g1,sc2,sh2,g2=ada
        b,n,d=x.shape;hh=op['heads']
        y=F.layer_norm(x,(d,),eps=op.get('eps',1e-5))*(1+sc1[:,None,:])+sh1[:,None,:]
        q,k,v=self.qkv(y).reshape(b,n,3,hh,d//hh).permute(2,0,3,1,4).unbind(0)
        q=rope(q,xyz,op['rope'],op['rope_scale'],op.get('freq_kind','inverse'));k=rope(k,xyz,op['rope'],op['rope_scale'],op.get('freq_kind','inverse'))
        a=F.scaled_dot_product_attention(q,k,v,attn_mask=None if mask is None else mask[:,None,None,:],dropout_p=0.)
        x=x+g1[:,None,:]*self.out(a.transpose(1,2).reshape(b,n,d))
        y=F.layer_norm(x,(d,),eps=op.get('eps',1e-5))*(1+sc2[:,None,:])+sh2[:,None,:]
        f=F.gelu(self.ff[0](y)) if op.get('ff','gelu')=='gelu' else F.silu(self.ff[0](y))
        return x+g2[:,None,:]*self.ff[2](f)

class TopologyFlow(nn.Module):
    def __init__(self,layers=10):
        super().__init__();w=144
        self.input=nn.Linear(64,w);self.position=nn.Linear(39,w)
        self.time=nn.Sequential(nn.Linear(64,w),nn.SiLU(),nn.Linear(w,w))
        self.blocks=nn.ModuleList([DiTBlock(w) for _ in range(layers)])
        self.out=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,64))
        self.op=dict(heads=6,rope='pair',rope_scale=1.,time_scale=1000.,time_order='cs',time_den=32)
    def forward(self,z,t,xyz,mask=None):
        op=self.op
        x=self.input(z)+self.position(fourier_positions(xyz))
        tem=time_embedding(t,op['time_scale'],op['time_order'],op['time_den'])
        c=self.time(tem)
        for blk in self.blocks:x=blk(x,c,xyz,mask,op)
        return self.out(x)

class PointFlow(nn.Module):
    def __init__(self,layers=18,prior_width=512):
        super().__init__();w=144
        self.register_buffer('residual_scale',torch.tensor(1.))
        self.text=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,w),nn.SiLU(),nn.Linear(w,w))
        self.count=nn.Sequential(nn.SiLU(),nn.Linear(w,275))
        self.slot=nn.Embedding(274,w);self.input=nn.Linear(39,w)
        self.time=nn.Sequential(nn.Linear(64,w),nn.SiLU(),nn.Linear(w,w))
        self.blocks=nn.ModuleList([DiTBlock(w) for _ in range(layers)])
        self.out=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,3))
        self.coordinate_prior=nn.Sequential(nn.LayerNorm(2048),nn.Linear(2048,prior_width),nn.SiLU(),nn.Linear(prior_width,822))
        self.op=dict(heads=4,rope='pair',rope_scale=math.pi,time_scale=1000.,time_order='cs',time_den=32,eps=1e-5)
    def text_hidden(self,text):
        x=self.text[1](self.text[0](text))
        x=F.silu(x) if self.op.get('text_act','silu')=='silu' else F.gelu(x)
        return self.text[3](x)
    def predict_counts(self,text):return self.count(self.text_hidden(text))
    def prior(self,text):return self.coordinate_prior(text).reshape(-1,274,3)
    def denoise(self,x,t,text,mask=None):
        op=self.op;b,n,d=x.shape
        h=self.input(fourier_positions(x))+self.slot.weight[:n][None]
        text=self.text_hidden(text)
        tem=time_embedding(t,op['time_scale'],op['time_order'],op['time_den']);c=self.time(tem)+text
        if op.get('add_text'):h=h+text[:,None,:]
        for block in self.blocks:h=block(h,c,x,mask,op)
        return self.out(h)
    def forward(self,x,t,text,mask=None):
        return self.prior(text)[:,:x.shape[1]]+self.residual_scale*self.denoise(x,t,text,mask)
