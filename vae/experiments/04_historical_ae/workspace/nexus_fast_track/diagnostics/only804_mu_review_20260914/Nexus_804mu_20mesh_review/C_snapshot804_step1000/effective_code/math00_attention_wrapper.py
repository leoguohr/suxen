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
