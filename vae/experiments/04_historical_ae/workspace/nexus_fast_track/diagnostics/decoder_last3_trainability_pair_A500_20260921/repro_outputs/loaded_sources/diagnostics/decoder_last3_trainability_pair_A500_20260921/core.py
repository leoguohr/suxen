"""Same arithmetic as the approved last2 scorer, with existing block13 prepended."""
from runtime import T,b
import face_core

def score(tail,d,sc,scales):
    x=d['x'];cu=T.tensor([0,len(x)],dtype=T.int32,device=x.device);h=x
    for name in ['third','penultimate','block']:
        block=tail[name]
        h=h+b.unified_attention(block.attention,block.norm(h),cu,len(h))
    h=tail['final_norm'](h)
    raw=tail['head'](h);center=raw-raw.mean(0,keepdim=True);pairs=d['pairs']
    logits=sc.first_order_interval(center[pairs[:,0]],center[pairs[:,1]])*scales['edge_logit_scale']
    numerator,mass=sc.soft4_sums(logits,d['labels'])
    edge=(raw,center,logits,(numerator/(mass+1e-8)).mean())
    fr,fe=face_core.center(tail['face'],h)
    fl,fs=face_core.objective(fe,d['face_tris'],d['face_labels'],scales)
    return h,edge,(fr,fe,fs,fl)
