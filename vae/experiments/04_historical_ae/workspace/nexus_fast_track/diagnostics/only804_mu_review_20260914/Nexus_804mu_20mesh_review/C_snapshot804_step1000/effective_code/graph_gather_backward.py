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
