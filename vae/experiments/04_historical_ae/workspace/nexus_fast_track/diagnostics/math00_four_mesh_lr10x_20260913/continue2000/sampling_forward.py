def _flash_varlen_autoencoder_forward(
    autoencoder: nn.Module,
    vertices: Tensor,
    vertex_mask: Tensor,
    faces: tuple[Tensor, ...],
    incidence_index: tuple[Tensor, ...],
    sample_seeds: tuple[int, ...] | None,
) -> tuple[
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    tuple[Tensor, ...],
    tuple[Tensor, ...],
]:
    """完成一次 packed 前向，并返回逐 mesh 的四组结果。

    输入 ``vertices`` 为 ``[B,Vmax,3]``；返回的每个 tuple 长度为 B，而第 b
    项长度为该 UID 的真实 V。这样后续 loss 仍然逐 mesh 计算并等权平均。
    """

    if vertices.dtype != torch.float32:
        raise TypeError(f"Topology AE vertices must be FP32: {vertices.dtype}")
    # mask 恢复每个对象的真实顶点数；face 数来自未 padding 的 tuple。
    vertex_counts = [int(value) for value in vertex_mask.sum(dim=1).tolist()]
    node_lengths = [count + len(face) for count, face in zip(vertex_counts, faces)]
    node_rows = []
    source_rows = []
    target_rows = []
    node_offset = 0
    # 先把各 mesh 的 vertex nodes 与 face-centroid nodes 物理拼成一条长序列。
    # incidence 加 node_offset 后成为这条长序列中的索引，但不会跨 mesh 连边。
    for batch_index, (vertex_count, face, incidence) in enumerate(
        zip(vertex_counts, faces, incidence_index)
    ):
        sample_vertices = vertices[batch_index, :vertex_count]
        vertex_features = autoencoder.vertex_input(sample_vertices)
        centroids = sample_vertices[face].mean(dim=1)
        face_features = autoencoder.face_input(centroids)
        node_rows.append(torch.cat((vertex_features, face_features), dim=0))
        source_rows.append(incidence[0] + node_offset)
        target_rows.append(incidence[1] + node_offset)
        node_offset += vertex_count + len(face)

    nodes = torch.cat(node_rows, dim=0)
    source = torch.cat(source_rows)
    target = torch.cat(target_rows)
    node_cu = _cu_seqlens(node_lengths, nodes.device)
    maximum_nodes = max(node_lengths)
    # 每个 composite block 仍执行一层局部 MeanSAGEConv 和一层全局 Transformer。
    for block in autoencoder.encoder_blocks:
        graph_update = block.graph(block.graph_norm(nodes), source, target)
        nodes = nodes + block.graph_activation(graph_update)
        nodes = _encoder_transformer_varlen(
            block.transformer, nodes, node_cu, maximum_nodes
        )

    # Encoder 序列包含 vertex+face nodes；VAE latent 只取每段开头的 vertex nodes。
    vertex_indices = []
    node_offset = 0
    for vertex_count, node_count in zip(vertex_counts, node_lengths):
        vertex_indices.append(
            torch.arange(
                node_offset,
                node_offset + vertex_count,
                dtype=torch.long,
                device=nodes.device,
            )
        )
        node_offset += node_count
    vertex_hidden = autoencoder.encoder_output_norm(nodes[torch.cat(vertex_indices)])
    mu = autoencoder.mu(vertex_hidden)
    log_variance = autoencoder.log_variance(vertex_hidden).clamp(autoencoder.diagnostic_logvar_min, 10.0)

    # 训练时执行 z = mu + sigma * epsilon；评估时直接用 posterior mean。
    # UID-stable seed 保证改变 rank 分配或 pack 顺序不会改变某 UID 的 epsilon。
    if autoencoder.training:
        mu_rows = _split_rows(mu, vertex_counts)
        logvar_rows = _split_rows(log_variance, vertex_counts)
        latent_rows = []
        for sample_index, (sample_mu, sample_logvar) in enumerate(
            zip(mu_rows, logvar_rows)
        ):
            if sample_seeds is None:
                noise = torch.randn(sample_mu.shape, dtype=sample_mu.dtype, device=sample_mu.device, generator=autoencoder.diagnostic_train_rng)
                autoencoder.diagnostic_train_draws += 1
            else:
                generator = torch.Generator(device=sample_mu.device)
                generator.manual_seed(sample_seeds[sample_index])
                noise = torch.randn(
                    sample_mu.shape,
                    dtype=sample_mu.dtype,
                    device=sample_mu.device,
                    generator=generator,
                )
            autoencoder._diagnostic_eps.append(noise.detach())
            latent_rows.append(
                sample_mu + torch.exp(0.5 * sample_logvar) * noise
            )
        latent = torch.cat(latent_rows, dim=0)
    else:
        latent = mu

    # Decoder 只处理 V 个 vertex latent；edge 与 face 使用分开的输出 head。
    hidden = autoencoder.latent_input(latent)
    vertex_cu = _cu_seqlens(vertex_counts, hidden.device)
    maximum_vertices = max(vertex_counts)
    for block in autoencoder.decoder_blocks:
        update = _flash_self_attention(
            block.attention,
            block.norm(hidden),
            vertex_cu,
            maximum_vertices,
        )
        hidden = hidden + update

    hidden = autoencoder.decoder_output_norm(hidden)

    # Center spacetime embeddings in FP32. Translation does not change an exact
    # interval; the explicit cast also guards the reconstruction boundary.
    edge_rows = tuple(
        row.float()
        for row in _split_rows(autoencoder.edge_embedding(hidden), vertex_counts)
    )
    face_rows = tuple(
        row.float()
        for row in _split_rows(autoencoder.face_embedding(hidden), vertex_counts)
    )
    if autoencoder.normalize_spacetime_embeddings:
        edge_rows = tuple(
            normalize_embedding(row, autoencoder.embedding_normalization_eps)
            for row in edge_rows
        )
        face_rows = tuple(
            normalize_embedding(row, autoencoder.embedding_normalization_eps)
            for row in face_rows
        )
    else:
        edge_rows = tuple(row - row.mean(dim=0, keepdim=True) for row in edge_rows)
        face_rows = tuple(row - row.mean(dim=0, keepdim=True) for row in face_rows)
    outputs = (
        _split_rows(mu, vertex_counts),
        _split_rows(log_variance, vertex_counts),
        edge_rows,
        face_rows,
    )
    if any(value.dtype != torch.float32 for rows in outputs for value in rows):
        raise TypeError("all non-Flash Topology AE outputs must be FP32")
    return outputs
