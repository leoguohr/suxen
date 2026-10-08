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
