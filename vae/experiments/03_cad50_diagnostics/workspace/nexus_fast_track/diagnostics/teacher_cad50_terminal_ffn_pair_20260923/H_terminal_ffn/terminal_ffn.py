"""The only new trainable structure: identity-initialized terminal FFN.

Persisted as autoencoder.terminal_ffn.* in every model state_dict. Reinstall
the named prehook with load_extended_state when loading an extended checkpoint.
No original model source or original decoder block is patched.
"""
import torch
from torch import nn

CONTRACT=dict(name='autoencoder.terminal_ffn',width=1024,hidden=4096,
    placement='decoder_output_norm forward_pre_hook; after last attention residual',
    formula='x + out_proj(GELU(in_proj(LayerNorm(x))))',
    layer_norm_eps=1e-5,dropout=0,zero_init=['out_proj.weight','out_proj.bias'])


class TerminalFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=nn.LayerNorm(1024,eps=1e-5)
        self.in_proj=nn.Linear(1024,4096)
        self.activation=nn.GELU()
        self.out_proj=nn.Linear(4096,1024)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self,x):
        return x+self.out_proj(self.activation(self.in_proj(self.norm(x))))

    def before_output_norm(self,module,inputs):
        assert len(inputs)==1
        return (self(inputs[0]),)


def install_terminal_ffn(autoencoder):
    assert not hasattr(autoencoder,'terminal_ffn'),'Refuse duplicate terminal FFN/hook'
    reference=autoencoder.decoder_output_norm.weight
    assert tuple(reference.shape)==(1024,)
    block=TerminalFFN().to(device=reference.device,dtype=reference.dtype)
    block.train(autoencoder.training)
    autoencoder.terminal_ffn=block
    autoencoder._terminal_ffn_hook_handle=autoencoder.decoder_output_norm.register_forward_pre_hook(block.before_output_norm)
    return block


def load_extended_state(model,checkpoint):
    """Restore the persistent module AND its execution placement, strictly."""
    assert checkpoint['config']['terminal_ffn']==CONTRACT
    if not hasattr(model.autoencoder,'terminal_ffn'):
        install_terminal_ffn(model.autoencoder)
    assert hasattr(model.autoencoder,'_terminal_ffn_hook_handle')
    model.load_state_dict(checkpoint['model'],strict=True)
    return model


def gradient_details(block):
    result={}
    for name,parameter in block.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(),name
        result[name]=dict(norm=float(parameter.grad.detach().double().square().sum().sqrt()),
                          nonzero=int(torch.count_nonzero(parameter.grad)))
    return result
