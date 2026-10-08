def forward(self, nodes: Tensor, source: Tensor, target: Tensor) -> Tensor:
    neighbor_sum = torch.zeros_like(nodes)
    with deterministic_reduction():
        neighbor_sum.index_add_(0, target, deterministic_gather(nodes, source))
    counts = torch.zeros((len(nodes), 1), dtype=nodes.dtype, device=nodes.device)
    with deterministic_reduction():
        counts.index_add_(0, target, torch.ones((len(target), 1), dtype=nodes.dtype, device=nodes.device))
    neighbor_mean = neighbor_sum / counts.clamp_min(1.0)
    return self.self_projection(nodes) + self.neighbor_projection(neighbor_mean)
