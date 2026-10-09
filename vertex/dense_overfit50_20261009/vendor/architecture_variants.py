"""Independent S0--S6 Vertex variants and explicit A24000 Adam migration.

Only fixed point features and FPS indices may be cached. Learned VecSet outputs
and queries are always evaluated from current parameters. No training runs here.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys

import torch
from torch import Tensor, nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent
BASE_CODE = ROOT  # vendored copy of the S0 mini_nexus (byte-identical)
sys.path.insert(0, str(BASE_CODE))
from mini_nexus.training import VertexStageSystem
from mini_nexus.vertex import (
    VertexConditionEncoder, VertexDiT, _LayerNorm, _valid_mask,
    _fourier_xyz, farthest_point_sample, parent_centers, checkpoint,
)

SOURCE_SHA256 = "2505189c1ab0aa8e3cf9f0cdb2b801619b21aae20d64c3fa77c8455790006aba"
SOURCE_STEP = 24000
INIT_SEED = 20261006
BRANCHES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6")
BRANCH_CONFIG = {
    "S0": dict(final_time_adaln=False, cross_qk="rms", self_qk="rms",
               learned_queries=False, hunyuan_vecset=False, global_condition=False),
    "S1": dict(final_time_adaln=True, cross_qk="rms", self_qk="rms",
               learned_queries=False, hunyuan_vecset=False, global_condition=False),
    "S2": dict(final_time_adaln=False, cross_qk="none", self_qk="rms",
               learned_queries=False, hunyuan_vecset=False, global_condition=False),
    "S3": dict(final_time_adaln=False, cross_qk="rms", self_qk="rms",
               learned_queries=True, hunyuan_vecset=False, global_condition=False),
    "S4": dict(final_time_adaln=True, cross_qk="none", self_qk="trellis",
               learned_queries=False, hunyuan_vecset=False, global_condition=False),
    "S5": dict(final_time_adaln=False, cross_qk="shared_rms", self_qk="shared_rms",
               learned_queries=False, hunyuan_vecset=True, global_condition=False),
    "S6": dict(final_time_adaln=False, cross_qk="none", self_qk="none",
               learned_queries=False, hunyuan_vecset=False, global_condition=True),
}


class TrellisHeadNorm(nn.Module):
    """TRELLIS L2 normalization: default normalize eps=1e-12, per-channel gain."""
    def __init__(self, heads: int, head_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(heads, head_dim))

    def forward(self, x: Tensor) -> Tensor:
        return (F.normalize(x.float(), dim=-1) * self.weight.float()[None, :, None, :]
                * math.sqrt(x.shape[-1])).to(x.dtype)


class SharedHeadRMSNorm(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: Tensor) -> Tensor:
        values = x.float()
        values = values * torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + 1e-6)
        return (values * self.weight.float()).to(x.dtype)


def _set_qk(attention, kind: str) -> None:
    for name in ("query_norm", "key_norm"):
        if kind == "none":
            module = nn.Identity()
        elif kind == "trellis":
            module = TrellisHeadNorm(attention.num_heads, attention.head_dim)
        elif kind == "shared_rms":
            module = SharedHeadRMSNorm(attention.head_dim)
        elif kind == "shared_layer":
            module = _LayerNorm(attention.head_dim)
        elif kind == "rms":
            continue
        else:
            raise ValueError(f"unknown QK normalization: {kind}")
        setattr(attention, name, module)


def _remove_qkv_bias(attention) -> None:
    for name in ("qkv", "query", "key_value"):
        if hasattr(attention, name):
            getattr(attention, name).register_parameter("bias", None)


def _fourier_no_pi(positions: Tensor) -> Tensor:
    frequencies = 2.0 ** torch.arange(8, device=positions.device, dtype=torch.float32)
    xyz = positions.float()
    angles = xyz.unsqueeze(-1) * frequencies
    return torch.cat([xyz, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)


class SweepConditionEncoder(VertexConditionEncoder):
    def __init__(self, *, learned_queries=False, hunyuan_vecset=False, **kwargs):
        super().__init__(**kwargs)
        self.learned_query_mode = learned_queries
        self.hunyuan_vecset = hunyuan_vecset
        self._fixed_features = {}
        self._last_group_id = None
        if learned_queries:
            self.learned_queries = nn.Parameter(torch.empty(self.num_tokens, self.hidden_dim))
            nn.init.normal_(self.learned_queries, std=0.02)
        if hunyuan_vecset:
            for block in self.blocks:
                _set_qk(block.attention, "shared_layer")
                _remove_qkv_bias(block.attention)
                block.ffn[1] = nn.GELU()
            self.output_norm.eps = 1e-5

    def prepare_inputs(self, points_and_normals, mask=None):
        if points_and_normals.ndim != 3 or points_and_normals.shape[-1] != self.input_dim:
            raise ValueError("point cloud must have shape [B,P,input_dim]")
        valid = _valid_mask(mask, points_and_normals.shape[:2], points_and_normals.device)
        if not valid.any(dim=1).all():
            raise ValueError("each object needs at least one valid condition point")
        points = points_and_normals.masked_fill(~valid.unsqueeze(-1), 0)
        indices = (None if self.learned_query_mode else
                   farthest_point_sample(points[..., :3], self.num_tokens, valid))
        fourier = _fourier_no_pi if self.hunyuan_vecset else _fourier_xyz
        features = torch.cat([fourier(points[..., :3]), points[..., 3:].float()], dim=-1)
        return features, indices, valid

    def forward(self, points_and_normals: Tensor, mask: Tensor | None = None) -> Tensor:
        saved = self._fixed_features.get(id(points_and_normals)) if mask is None else None
        if saved is None:
            features, indices, valid = self.prepare_inputs(points_and_normals, mask)
        else:
            reference, version, features, indices, valid = saved
            if reference is not points_and_normals or reference._version != version:
                raise ValueError("registered condition tensor was modified")
        embedded = self.point_embedding(features.to(self.point_embedding.weight.dtype))
        if self.learned_query_mode:
            queries = self.learned_queries.unsqueeze(0).expand(embedded.shape[0], -1, -1)
        else:
            queries = embedded.gather(1, indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim))
        if self.use_checkpoint and self.training and torch.is_grad_enabled():
            x = checkpoint(self.blocks[0], queries, embedded, valid, use_reentrant=False)
            for block in self.blocks[1:]:
                x = checkpoint(block, x, use_reentrant=False)
        else:
            x = self.blocks[0](queries, embedded, valid)
            for block in self.blocks[1:]:
                x = block(x)
        return self.output_norm(x)


class SweepVertexDiT(VertexDiT):
    def __init__(self, branch: str, **kwargs):
        super().__init__(**kwargs)
        options = BRANCH_CONFIG[branch]
        self.branch = branch
        for block in self.blocks:
            _set_qk(block.self_attention, options["self_qk"])
            _set_qk(block.cross_attention, options["cross_qk"])
            if branch == "S5":
                _remove_qkv_bias(block.self_attention)
                _remove_qkv_bias(block.cross_attention)
            if branch in ("S5", "S6"):
                block.ffn[1] = nn.GELU()
            if branch == "S6":
                block.self_norm.eps = block.ffn_norm.eps = 1e-5
        if options["final_time_adaln"]:
            self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(self.hidden_dim, 2 * self.hidden_dim))
            nn.init.zeros_(self.final_modulation[-1].weight)
            nn.init.zeros_(self.final_modulation[-1].bias)
        if branch in ("S5", "S6"):
            self.output_norm = _LayerNorm(self.hidden_dim)
            self.output_norm.eps = 1e-6 if branch == "S5" else 1e-5
        if options["global_condition"]:
            self.global_condition = nn.Sequential(
                _LayerNorm(self.condition_dim), nn.Linear(self.condition_dim, self.hidden_dim),
                nn.SiLU(), nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            self.global_condition[0].eps = 1e-5
            nn.init.xavier_uniform_(self.global_condition[1].weight)
            nn.init.zeros_(self.global_condition[1].bias)
            nn.init.zeros_(self.global_condition[-1].weight)
            nn.init.zeros_(self.global_condition[-1].bias)

    def forward(self, noisy, time, parent_codes, depths, condition_tokens,
                mask=None, condition_mask=None):
        if noisy.ndim != 3 or noisy.shape[-1] != 8:
            raise ValueError("noisy occupancy must have shape [B,N,8]")
        batch_size, token_count = noisy.shape[:2]
        if parent_codes.shape != (batch_size, token_count, 3) or parent_codes.dtype != torch.long:
            raise ValueError("parent_codes must be long with shape [B,N,3]")
        if time.shape != (batch_size,):
            raise ValueError("time must have shape [B]")
        if depths.shape not in [(batch_size,), (batch_size, token_count)] or depths.dtype != torch.long:
            raise ValueError("depths must be long with shape [B] or [B,N]")
        if (condition_tokens.ndim != 3 or condition_tokens.shape[0] != batch_size
                or condition_tokens.shape[-1] != self.condition_dim):
            raise ValueError("condition_tokens must have shape [B,M,condition_dim]")
        valid = _valid_mask(mask, noisy.shape[:2], noisy.device)
        condition_valid = _valid_mask(condition_mask, condition_tokens.shape[:2], condition_tokens.device)
        if not condition_valid.any(dim=1).all():
            raise ValueError("each object needs at least one valid condition token")
        if depths.ndim == 1:
            depths = depths[:, None].expand(-1, token_count)
        if (((depths < 1) | (depths > self.max_depth)) & valid).any():
            raise ValueError("valid target depths must lie in [1,max_depth]")
        safe_depths = depths.masked_fill(~valid, 1)
        safe_codes = parent_codes.masked_fill(~valid.unsqueeze(-1), 0)
        centers = parent_centers(safe_codes, safe_depths)
        rope_positions = centers * (2 ** (self.rope_reference_depth - 1))
        clean_noisy = noisy.masked_fill(~valid.unsqueeze(-1), 0)
        condition = condition_tokens.masked_fill(~condition_valid.unsqueeze(-1), 0)
        dtype = self.data_embedding.weight.dtype
        x = self.data_embedding(clean_noisy.to(dtype))
        x = x + self.position_embedding(_fourier_xyz(centers).to(dtype))
        x = x + self.depth_embedding(safe_depths - 1)
        x = x.masked_fill(~valid.unsqueeze(-1), 0)
        frequencies = 10_000.0 ** (-torch.arange(128, device=time.device, dtype=torch.float32) / 128)
        angles = (1000 * time.float()).unsqueeze(-1) * frequencies
        time_features = torch.cat([angles.cos(), angles.sin()], dim=-1)
        time_embedding = self.time_embedding(time_features.to(dtype))
        condition = condition.to(dtype)
        if hasattr(self, "global_condition"):
            pooled = condition.float().sum(dim=1) / condition_valid.sum(dim=1, keepdim=True)
            global_embedding = self.global_condition(pooled.to(dtype))
            x = (x + global_embedding.unsqueeze(1)).masked_fill(~valid.unsqueeze(-1), 0)
            time_embedding = time_embedding + global_embedding
        for block in self.blocks:
            args = (x, time_embedding, condition, rope_positions, valid, condition_valid)
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                x = checkpoint(block, *args, use_reentrant=False)
            else:
                x = block(*args)
        x = self.output_norm(x)
        if hasattr(self, "final_modulation"):
            shift, scale = self.final_modulation(time_embedding).unsqueeze(1).chunk(2, dim=-1)
            x = x * (1 + scale) + shift
        return self.output(x).masked_fill(~valid.unsqueeze(-1), 0)


class SweepVertexStageSystem(VertexStageSystem):
    architecture = "nexus_vertex_arch_sweep_20261006"

    def __init__(self, branch: str, *, small=False, use_checkpoint=False):
        nn.Module.__init__(self)
        self.branch = branch
        self.small = small
        hidden, condition, layers, heads = (24, 32, 2, 3) if small else (1536, 2048, 36, 12)
        condition_heads = (8 if small else 32) if branch == "S5" else (4 if small else 16)
        self.condition_encoder = SweepConditionEncoder(
            hidden_dim=condition, num_tokens=4 if small else 1024,
            num_heads=condition_heads, num_layers=2 if small else 8,
            use_checkpoint=use_checkpoint,
            learned_queries=BRANCH_CONFIG[branch]["learned_queries"],
            hunyuan_vecset=BRANCH_CONFIG[branch]["hunyuan_vecset"],
        )
        self.flow = SweepVertexDiT(
            branch, hidden_dim=hidden, condition_dim=condition, num_layers=layers,
            num_heads=heads, max_depth=9, rope_reference_depth=9,
            use_checkpoint=use_checkpoint,
        )


def construct_model(branch: str, small: bool = False, *, use_checkpoint=False):
    if branch not in BRANCHES:
        raise ValueError(f"branch must be one of {BRANCHES}")
    # Construction must not consume the training draw stream. Meta/CPU callers
    # are supported; CUDA is selected only after restoration below.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(INIT_SEED)
        return SweepVertexStageSystem(branch, small=small, use_checkpoint=use_checkpoint)


def cache_fixed_features(model, conditions):
    """Register immutable B=1 or pre-grouped tensors; uncached tensors still work."""
    encoder = model.condition_encoder
    records = []
    with torch.no_grad():
        for condition in conditions:
            features, indices, valid = encoder.prepare_inputs(condition)
            encoder._fixed_features[id(condition)] = (condition, condition._version, features, indices, valid)
            records.append({"shape": list(condition.shape),
                            "fps_indices_cached": indices is not None,
                            "indices_sha256": (None if indices is None else
                                hashlib.sha256(indices.cpu().numpy().tobytes()).hexdigest())})
    return {"immutable_point_features_cached": True,
            "fourier_pi": not encoder.hunyuan_vecset,
            "learned_queries_cached": False, "learned_condition_features_cached": False,
            "records": records}


def cached_condition_batch(model, conditions, mesh_indices):
    """Build a real condition batch plus fixed-feature cache, without FPS reruns."""
    encoder = model.condition_encoder
    selected = [conditions[index] for index in mesh_indices]
    batch = torch.cat(selected, dim=0)
    prepared = []
    for condition in selected:
        entry = encoder._fixed_features.get(id(condition))
        if entry is None or entry[0] is not condition or entry[1] != condition._version:
            raise ValueError("cache_fixed_features must register immutable source conditions first")
        prepared.append(entry[2:])
    features = torch.cat([entry[0] for entry in prepared], dim=0)
    indices = (None if encoder.learned_query_mode else
               torch.cat([entry[1] for entry in prepared], dim=0))
    valid = torch.cat([entry[2] for entry in prepared], dim=0)
    if encoder._last_group_id is not None:
        encoder._fixed_features.pop(encoder._last_group_id, None)
    encoder._fixed_features[id(batch)] = (batch, batch._version, features, indices, valid)
    encoder._last_group_id = id(batch)
    return batch


def _new_parameter(name, expected, branch):
    """Name-seeded initialization on CPU, independent of construction/RNG order."""
    seed = int.from_bytes(hashlib.sha256(f"{INIT_SEED}|{branch}|{name}".encode()).digest()[:8], "little")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    result = torch.empty(expected.shape, dtype=expected.dtype, device="cpu")
    if name == "condition_encoder.learned_queries":
        result.normal_(std=0.02, generator=generator)
    elif name.startswith("flow.final_modulation.") or name.startswith("flow.global_condition.3."):
        result.zero_()
    elif name == "flow.global_condition.1.weight":
        nn.init.xavier_uniform_(result, generator=generator)
    elif name.endswith(".bias"):
        result.zero_()
    elif name.endswith(".weight") and result.ndim == 1:
        result.fill_(1)
    else:
        raise ValueError(f"no explicit initialization policy for new parameter: {name}")
    return result


def _source_parameter_map(source, small):
    if source.get("step") != SOURCE_STEP or source.get("scheduler") is not None:
        raise ValueError("requires complete A24000 source, step=24000 and no scheduler")
    config = source.get("config", {})
    if config.get("max_depth") != 9 or config.get("rope_reference_depth") != 9:
        raise ValueError("A24000 source must retain D9 and reference RoPE9")
    with torch.device("meta"):
        dimensions = (dict(hidden_dim=24, condition_dim=32, num_layers=2, num_heads=3,
                           condition_heads=4, condition_layers=2, condition_tokens=4)
                      if small else {})
        reference = VertexStageSystem(max_depth=9, rope_reference_depth=9,
                                      use_checkpoint=False, **dimensions)
    expected = dict(reference.named_parameters())
    state = source.get("model", {})
    if set(state) != set(reference.state_dict()):
        raise ValueError("source model keys do not exactly match S0")
    for name, param in expected.items():
        if state[name].shape != param.shape or state[name].dtype != param.dtype:
            raise ValueError(f"source name/shape/dtype mismatch: {name}")
    saved = source.get("optimizer", {})
    groups = saved.get("param_groups", [])
    if len(groups) != 1 or len(groups[0].get("params", [])) != len(expected):
        raise ValueError("source Adam group does not cover every S0 parameter")
    group = groups[0]
    for name, expected_value in {"lr": 1e-5, "weight_decay": 0, "betas": (0.9, 0.999),
                                  "eps": 1e-8, "amsgrad": False, "maximize": False}.items():
        if group.get(name, False if name == "maximize" else None) != expected_value:
            raise ValueError(f"source Adam hyperparameter differs: {name}")
    ids = group["params"]
    if len(set(ids)) != len(ids) or set(saved.get("state", {})) != set(ids):
        raise ValueError("source Adam missing/duplicate/orphan parameter states")
    if "param_names" in group and group["param_names"] != list(expected):
        raise ValueError("saved Adam param_names do not match S0 semantic order")
    mapping = dict(zip(expected, ids))
    for name, param_id in mapping.items():
        moment = saved["state"][param_id]
        if set(moment) != {"step", "exp_avg", "exp_avg_sq"}:
            raise ValueError(f"incomplete source Adam state: {name}")
        step = moment["step"]
        if not torch.is_tensor(step) or step.numel() != 1 or float(step.item()) != SOURCE_STEP:
            raise ValueError(f"source Adam step must be 24000: {name}")
        for key in ("exp_avg", "exp_avg_sq"):
            if moment[key].shape != expected[name].shape or moment[key].dtype != state[name].dtype:
                raise ValueError(f"source Adam name/shape/dtype mismatch: {name}.{key}")
    return mapping


def restore_source(source, branch: str, device, *, small: bool = False):
    """Restore from A24000 only; old moments retained, new/transformed Adam step=0."""
    device = torch.device(device)
    source_ids = _source_parameter_map(source, small)
    with torch.device("meta"):
        model = construct_model(branch, small=small)
    expected = model.state_dict()
    source_state = source["model"]
    inherited, added, transformed = [], [], []
    migrated = {}
    for name, param in expected.items():
        value = source_state.get(name)
        if value is None:
            value = _new_parameter(name, param, branch)
            added.append(name)
        elif value.shape == param.shape and value.dtype == param.dtype:
            inherited.append(name)
        elif (branch == "S5" and name.endswith(("query_norm.weight", "key_norm.weight"))
              and name.startswith("flow.blocks.") and value.ndim == 2
              and value.shape[1:] == param.shape and value.dtype == param.dtype):
            value = value.mean(dim=0)
            transformed.append(name)
        else:
            raise ValueError(f"unsupported semantic migration: {name}: {tuple(value.shape)} -> {tuple(param.shape)}")
        migrated[name] = value.to(device=device, copy=True)
        if not torch.equal(migrated[name].detach().cpu(), value.detach().cpu()):
            raise ValueError(f"model value changed during migration copy: {name}")
    removed = sorted(set(source_state) - set(expected))
    model.load_state_dict(migrated, strict=True, assign=True)
    model.train().requires_grad_(True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0, foreach=False)
    current = dict(model.named_parameters())
    for name, param in current.items():
        if name in inherited:
            old = source["optimizer"]["state"][source_ids[name]]
            optimizer.state[param] = {
                "step": old["step"].to(device="cpu", copy=True),
                "exp_avg": old["exp_avg"].to(device=device, copy=True),
                "exp_avg_sq": old["exp_avg_sq"].to(device=device, copy=True),
            }
            for key in ("step", "exp_avg", "exp_avg_sq"):
                if not torch.equal(optimizer.state[param][key].detach().cpu(), old[key].detach().cpu()):
                    raise ValueError(f"inherited Adam value changed during copy: {name}.{key}")
        else:
            optimizer.state[param] = {"step": torch.tensor(0.0, device="cpu"),
                                      "exp_avg": torch.zeros_like(param),
                                      "exp_avg_sq": torch.zeros_like(param)}
    def records(names, state):
        return [{"name": name, "shape": list(state[name].shape), "elements": state[name].numel()}
                for name in names]
    audit = {
        "branch": branch, "source_step": SOURCE_STEP, "source_sha256_required": SOURCE_SHA256,
        "model_keys_strict": True, "migration_by": "semantic parameter name and shape",
        "actual_value_verification": {"method": "torch.equal after copy, CPU comparison",
                                      "model_tensors_checked": len(migrated),
                                      "inherited_adam_tensors_checked": len(inherited) * 3,
                                      "all_equal": True},
        "source_optimizer_mapping": "validated original S0 named_parameters order",
        "inherited": records(inherited, expected), "added": records(added, expected),
        "removed": records(removed, source_state),
        "transformed": [{"name": name, "source_shape": list(source_state[name].shape),
                          "target_shape": list(expected[name].shape),
                          "operation": "mean over source heads", "adam_reset": True}
                         for name in transformed],
        "optimizer": {"class": "Adam", "lr": 1e-5, "weight_decay": 0,
                      "clip_required": 1.0, "betas": [0.9, 0.999], "eps": 1e-8,
                      "foreach": False, "parameters": len(current), "states": len(optimizer.state),
                      "inherited_step24000": len(inherited), "new_step0": len(added),
                      "transformed_step0": len(transformed), "removed_states": len(removed),
                      "coverage_complete": len(current) == len(optimizer.state)},
        "initialization": {"seed": INIT_SEED, "policy": "SHA256(seed|branch|parameter-name), local CPU generator",
                           "source_rng_consumed": False},
        "activation_checkpoint": False,
        "equivalent_at_initialization_expected": branch in ("S0", "S1"),
        "parameter_elements": sum(param.numel() for param in current.values()),
        "parameter_elements_by_module": {
            "dit": sum(param.numel() for param in model.flow.parameters()),
            "vecset": sum(param.numel() for param in model.condition_encoder.parameters()),
        },
        "options": BRANCH_CONFIG[branch],
    }
    return model, optimizer, audit
