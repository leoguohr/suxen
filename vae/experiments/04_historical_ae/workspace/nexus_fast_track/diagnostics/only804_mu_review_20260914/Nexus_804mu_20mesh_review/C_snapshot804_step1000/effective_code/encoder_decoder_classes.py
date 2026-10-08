class MeanSAGEConv(nn.Module):
    """不依赖图学习库的 GraphSAGE mean aggregation。

    输出为 ``W_self*x_i + W_neighbor*mean(x_j for j -> i)``，负责 encoder 中
    局部、知道拓扑连接关系的信息传递。
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_projection = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, nodes: Tensor, source: Tensor, target: Tensor) -> Tensor:
        # nodes 是 [V+F,C]。index_add_ 按 target 把所有 source feature 相加，
        # 再除以入度得到 mean neighbor feature，不需要 torch-geometric。
        neighbor_sum = torch.zeros_like(nodes)
        neighbor_sum.index_add_(0, target, nodes[source])
        counts = torch.zeros(
            (len(nodes), 1), dtype=nodes.dtype, device=nodes.device
        )
        counts.index_add_(
            0,
            target,
            torch.ones((len(target), 1), dtype=nodes.dtype, device=nodes.device),
        )
        neighbor_mean = neighbor_sum / counts.clamp_min(1.0)
        return self.self_projection(nodes) + self.neighbor_projection(neighbor_mean)


class GraphTransformerBlock(nn.Module):
    """先做一层局部 GraphSAGE，再做一层全局 Transformer。

    GraphSAGE 只沿 vertex-face incidence edge 通信；全局 self-attention 再让
    远处节点和不同组件交换信息。二者交替是论文 topology encoder 的核心。
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.graph_norm = nn.LayerNorm(hidden_dim)
        self.graph = MeanSAGEConv(hidden_dim)
        self.graph_activation = nn.SiLU()
        self.transformer = nn.TransformerEncoderLayer(
            hidden_dim,
            num_heads,
            4 * hidden_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )

    def forward(self, nodes: Tensor, source: Tensor, target: Tensor) -> Tensor:
        # Pre-normalize, aggregate local neighbors, and add a residual update.
        graph_update = self.graph(self.graph_norm(nodes), source, target)
        nodes = nodes + self.graph_activation(graph_update)
        # The leading batch dimension is temporary: this function handles one mesh.
        return self.transformer(nodes.unsqueeze(0)).squeeze(0)

    def forward_packed(
        self,
        nodes: Tensor,
        node_mask: Tensor,
        source: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Process padded, independent graphs without cross-mesh attention."""

        batch_size, maximum_nodes, hidden_dim = nodes.shape
        flat_nodes = nodes.reshape(batch_size * maximum_nodes, hidden_dim)
        graph_update = self.graph(self.graph_norm(flat_nodes), source, target)
        nodes = nodes + self.graph_activation(graph_update.reshape_as(nodes))
        nodes = nodes.masked_fill(~node_mask.unsqueeze(-1), 0.0)
        nodes = self.transformer(nodes, src_key_padding_mask=~node_mask)
        return nodes.masked_fill(~node_mask.unsqueeze(-1), 0.0)


class AttentionBlock(nn.Module):
    """用于纯 attention decoder 的 pre-norm self-attention 残差块。

    论文只公开 decoder 是 pure attention、hidden=1024，但未公开 Q/K/V 投影
    维度、head 数或内部 bottleneck。这里采用 PyTorch 标准 full-width MHA：
    Q/K/V 和输出都保持 hidden_dim；不加入 FFN。两者都是明确的独立实现选择。
    使用论文公开宽度时，总参数量也与论文约 100M 的量级一致。
    """

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )

    def forward(self, nodes: Tensor) -> Tensor:
        normalized = self.norm(nodes).unsqueeze(0)
        update, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        return nodes + update.squeeze(0)

    def forward_packed(self, nodes: Tensor, node_mask: Tensor) -> Tensor:
        """Apply full attention inside each padded mesh and nowhere else."""

        normalized = self.norm(nodes)
        update, _ = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~node_mask,
            need_weights=False,
        )
        nodes = nodes + update
        return nodes.masked_fill(~node_mask.unsqueeze(-1), 0.0)


class TopologyAutoencoder(nn.Module):
    """带混合图 encoder 的“每顶点 latent”变分自编码器。

    ``hidden_dim`` 控制 encoder 宽度；``encoder_layers`` 统计单独层数，必须为
    偶数，因为每两层组成一组 GraphSAGE+Transformer；``decoder_layers`` 统计
    attention block；decoder 可以比 encoder 更宽。论文公开的 24/512 encoder
    与 16/1024 decoder 由同一个类实例化。
    """

    def __init__(
        self,
        hidden_dim: int = 96,
        latent_dim: int = 16,
        spacetime_dim: int = 16,
        num_heads: int = 4,
        num_layers: int | None = None,
        *,
        encoder_layers: int | None = None,
        decoder_layers: int | None = None,
        decoder_hidden_dim: int | None = None,
        encoder_dropout: float = 0.1,
        normalize_spacetime_embeddings: bool = False,
        embedding_normalization_eps: float = 1e-6,
    ):
        super().__init__()
        if spacetime_dim % 2:
            raise ValueError("spacetime_dim must be even")
        if encoder_layers is None:
            encoder_layers = 24 if num_layers is None else 2 * num_layers
        if decoder_layers is None:
            decoder_layers = 16 if num_layers is None else num_layers
        if decoder_hidden_dim is None:
            decoder_hidden_dim = 1024 if num_layers is None else hidden_dim
        if encoder_layers <= 0 or encoder_layers % 2:
            raise ValueError("encoder_layers must be a positive even number")
        if decoder_layers <= 0:
            raise ValueError("decoder_layers must be positive")
        if hidden_dim % num_heads or decoder_hidden_dim % num_heads:
            raise ValueError("encoder and decoder dimensions must divide num_heads")
        if embedding_normalization_eps <= 0 or not math.isfinite(
            embedding_normalization_eps
        ):
            raise ValueError("embedding_normalization_eps must be positive")
        self.normalize_spacetime_embeddings = normalize_spacetime_embeddings
        self.embedding_normalization_eps = embedding_normalization_eps
        self.vertex_input = nn.Linear(3, hidden_dim)
        # A face node starts from its geometric centroid, not from a learned face ID.
        self.face_input = nn.Linear(3, hidden_dim)
        self.encoder_blocks = nn.ModuleList(
            GraphTransformerBlock(hidden_dim, num_heads, dropout=encoder_dropout)
            for _ in range(encoder_layers // 2)
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_variance = nn.Linear(hidden_dim, latent_dim)
        # Isolated LayerNorm/no-RMS experiment: normalize the final residual stream.
        self.encoder_output_norm = nn.LayerNorm(hidden_dim)

        # Nexus Eq. (6), PDF p.5 L033-L040 写作 Z=Dec(H_V)，没有再次注入 V。
        # 因而 paper-aligned 路径只把每顶点 latent 投影到 decoder hidden width。
        self.latent_input = nn.Linear(latent_dim, decoder_hidden_dim)
        self.decoder_blocks = nn.ModuleList(
            AttentionBlock(decoder_hidden_dim, num_heads)
            for _ in range(decoder_layers)
        )
        self.edge_embedding = nn.Linear(decoder_hidden_dim, spacetime_dim)
        self.face_embedding = nn.Linear(decoder_hidden_dim, spacetime_dim)
        self.decoder_output_norm = nn.LayerNorm(decoder_hidden_dim)

    def encode(
        self,
        vertices: Tensor,
        faces: Tensor,
        incidence_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """把 ``vertices[V,3], faces[F,3]`` 编码为 ``mu/logvar[V,L]``。"""

        # Create V vertex-node features directly from normalized coordinates.
        vertex_features = self.vertex_input(vertices)
        # Create F face-node features from the mean of each face's three vertices.
        centroids = vertices[faces].mean(dim=1)
        face_features = self.face_input(centroids)
        # The encoder sequence has V+F nodes; vertex identity remains 0..V-1.
        nodes = torch.cat([vertex_features, face_features], dim=0)
        if incidence_index is None:
            source, target = build_vertex_face_incidence(faces, len(vertices))
        else:
            if incidence_index.ndim != 2 or incidence_index.shape[0] != 2:
                raise ValueError("incidence_index must have shape [2,I]")
            source, target = incidence_index[0], incidence_index[1]
        for block in self.encoder_blocks:
            nodes = block(nodes, source, target)
        # Face nodes help encode topology but only vertex nodes receive latents.
        vertex_hidden = self.encoder_output_norm(nodes[: len(vertices)])
        return self.mu(vertex_hidden), self.log_variance(vertex_hidden).clamp(-10.0, 10.0)

    def sample(self, mu: Tensor, log_variance: Tensor) -> Tensor:
        """训练时使用 VAE 重参数化技巧采样 latent。"""

        if self.training:
            return mu + torch.exp(0.5 * log_variance) * torch.randn_like(mu)
        return mu

    def decode(self, latent: Tensor) -> tuple[Tensor, Tensor]:
        """按论文 Eq. (6) 把每顶点 latent 解码成 edge/face 两套 embedding。"""

        hidden = self.latent_input(latent)
        for block in self.decoder_blocks:
            hidden = block(hidden)
        hidden = self.decoder_output_norm(hidden)
        # Spacetime intervals只依赖顶点间的坐标差，所以两条路径都先移除
        # 每个 mesh 的公共平移。新路径还用一个全局正 RMS 因子固定尺度；
        # 兼容路径只中心化。两者都不改变精确算术中的 interval 正负号。
        edge_embedding = self.edge_embedding(hidden)
        face_embedding = self.face_embedding(hidden)
        if self.normalize_spacetime_embeddings:
            edge_embedding = normalize_embedding(
                edge_embedding, self.embedding_normalization_eps
            )
            face_embedding = normalize_embedding(
                face_embedding, self.embedding_normalization_eps
            )
        else:
            edge_embedding = edge_embedding - edge_embedding.mean(dim=0, keepdim=True)
            face_embedding = face_embedding - face_embedding.mean(dim=0, keepdim=True)
        return edge_embedding, face_embedding

    def forward(
        self,
        vertices: Tensor,
        faces: Tensor,
        incidence_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        mu, log_variance = self.encode(vertices, faces, incidence_index)
        latent = self.sample(mu, log_variance)
        edge_embedding, face_embedding = self.decode(latent)
        return latent, mu, log_variance, edge_embedding, face_embedding

    def encode_packed(
        self,
        vertices: Tensor,
        vertex_mask: Tensor,
        faces: tuple[Tensor, ...],
        incidence_index: tuple[Tensor, ...],
    ) -> tuple[Tensor, Tensor]:
        """Encode several variable-length meshes in one padded Transformer call."""

        batch_size, maximum_vertices, _ = vertices.shape
        vertex_counts = vertex_mask.sum(dim=1).tolist()
        node_counts = [count + len(face) for count, face in zip(vertex_counts, faces)]
        maximum_nodes = max(node_counts)
        nodes = vertices.new_zeros((batch_size, maximum_nodes, self.mu.in_features))
        node_mask = torch.zeros(
            (batch_size, maximum_nodes), dtype=torch.bool, device=vertices.device
        )
        source_rows = []
        target_rows = []
        for batch_index, (vertex_count, face, incidence) in enumerate(
            zip(vertex_counts, faces, incidence_index)
        ):
            vertex_count = int(vertex_count)
            vertex_features = self.vertex_input(vertices[batch_index, :vertex_count])
            centroids = vertices[batch_index, :vertex_count][face].mean(dim=1)
            face_features = self.face_input(centroids)
            node_count = vertex_count + len(face)
            nodes[batch_index, :node_count] = torch.cat(
                [vertex_features, face_features], dim=0
            )
            node_mask[batch_index, :node_count] = True
            offset = batch_index * maximum_nodes
            source_rows.append(incidence[0] + offset)
            target_rows.append(incidence[1] + offset)
        source = torch.cat(source_rows)
        target = torch.cat(target_rows)
        for block in self.encoder_blocks:
            nodes = block.forward_packed(nodes, node_mask, source, target)

        vertex_hidden = self.encoder_output_norm(nodes[:, :maximum_vertices])
        mu = self.mu(vertex_hidden).masked_fill(~vertex_mask.unsqueeze(-1), 0.0)
        log_variance = self.log_variance(vertex_hidden).clamp(-10.0, 10.0)
        log_variance = log_variance.masked_fill(~vertex_mask.unsqueeze(-1), 0.0)
        return mu, log_variance

    def sample_packed(
        self,
        mu: Tensor,
        log_variance: Tensor,
        vertex_mask: Tensor,
        sample_seeds: tuple[int, ...] | None,
    ) -> Tensor:
        """Sample only valid vertices, optionally with UID-stable seeds."""

        if not self.training:
            return mu
        latent = torch.zeros_like(mu)
        for batch_index, mask in enumerate(vertex_mask):
            if sample_seeds is None:
                noise = torch.randn_like(mu[batch_index, mask])
            else:
                generator = torch.Generator(device=mu.device)
                generator.manual_seed(sample_seeds[batch_index])
                noise = torch.randn(
                    mu[batch_index, mask].shape,
                    dtype=mu.dtype,
                    device=mu.device,
                    generator=generator,
                )
            latent[batch_index, mask] = (
                mu[batch_index, mask]
                + torch.exp(0.5 * log_variance[batch_index, mask]) * noise
            )
        return latent

    def decode_packed(
        self, latent: Tensor, vertex_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Decode padded per-vertex latents with mesh-isolated attention."""

        hidden = self.latent_input(latent)
        hidden = hidden.masked_fill(~vertex_mask.unsqueeze(-1), 0.0)
        for block in self.decoder_blocks:
            hidden = block.forward_packed(hidden, vertex_mask)
        hidden = self.decoder_output_norm(hidden)
        edge_embedding = self.edge_embedding(hidden)
        face_embedding = self.face_embedding(hidden)
        if self.normalize_spacetime_embeddings:
            edge_embedding = _normalize_packed_embedding(
                edge_embedding, vertex_mask, self.embedding_normalization_eps
            )
            face_embedding = _normalize_packed_embedding(
                face_embedding, vertex_mask, self.embedding_normalization_eps
            )
        else:
            counts = vertex_mask.sum(dim=1, keepdim=True).unsqueeze(-1)
            edge_mean = (
                edge_embedding * vertex_mask.unsqueeze(-1)
            ).sum(dim=1, keepdim=True) / counts
            face_mean = (
                face_embedding * vertex_mask.unsqueeze(-1)
            ).sum(dim=1, keepdim=True) / counts
            edge_embedding = (edge_embedding - edge_mean).masked_fill(
                ~vertex_mask.unsqueeze(-1), 0.0
            )
            face_embedding = (face_embedding - face_mean).masked_fill(
                ~vertex_mask.unsqueeze(-1), 0.0
            )
        return edge_embedding, face_embedding

    def forward_packed(
        self,
        vertices: Tensor,
        vertex_mask: Tensor,
        faces: tuple[Tensor, ...],
        incidence_index: tuple[Tensor, ...],
        *,
        sample_seeds: tuple[int, ...] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Packed equivalent of :meth:`forward` for independent meshes."""

        mu, log_variance = self.encode_packed(
            vertices, vertex_mask, faces, incidence_index
        )
        latent = self.sample_packed(mu, log_variance, vertex_mask, sample_seeds)
        edge_embedding, face_embedding = self.decode_packed(latent, vertex_mask)
        return latent, mu, log_variance, edge_embedding, face_embedding
