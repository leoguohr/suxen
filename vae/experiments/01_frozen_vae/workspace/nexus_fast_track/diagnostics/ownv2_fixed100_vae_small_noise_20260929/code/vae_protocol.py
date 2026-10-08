"""Small, explicit additions to the frozen fixed100 AE numerical protocol."""
import hashlib
import math
import torch

BETA = 1e-6
START = 34220
UPDATES = 500
MILESTONES = (0, 100, 250, 500)
SOURCE_SHA = 'c9cd45dbd011c4adec9f3916bcdeb85dbe4bb2dcd3949db313356f1909f5ce95'
SOURCE_MODEL_SHA = '508cb0f40560a2ea060d0f6cfbd6131a4a7aa34b42e02898a0c6280d063409d5'


def kl_parts(mu, logvar):
    """Mean over vertices and latent channels, then sum the two KL terms."""
    k_mu = 0.5 * mu.square().mean()
    k_sigma = 0.5 * (logvar.exp() - 1 - logvar).mean()
    return k_mu, k_sigma


def uid_noise_seed(group_seed, uid):
    key = f'ownv2-fixed100-vae/eval/{group_seed}/{uid}'.encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') & ((1 << 63) - 1)


def evaluation_generator(device, group_seed, uid):
    return torch.Generator(device=device).manual_seed(uid_noise_seed(group_seed, uid))


def posterior_stats(rows):
    with torch.no_grad():
        mu, lv, raw = rows['mu'], rows['log_variance'], rows['raw_log_variance']
        sigma = (0.5 * lv).exp()
        sigma_quantiles = torch.quantile(sigma.flatten(), torch.tensor(
            [.01, .5, .99], device=sigma.device, dtype=sigma.dtype))
        k_mu, k_sigma = kl_parts(mu, lv)
        delta = rows['latent'] - mu
        epsilon = rows.get('epsilon')
        result = dict(vertices=len(mu), channels=mu.shape[1],
            mu_rms=float(mu.square().mean().sqrt()),
            raw_logvar_mean=float(raw.mean()), raw_logvar_min=float(raw.min()), raw_logvar_max=float(raw.max()),
            logvar_mean=float(lv.mean()), logvar_min=float(lv.min()), logvar_max=float(lv.max()),
            clamp_low_fraction=float((raw <= -20).float().mean()),
            clamp_high_fraction=float((raw >= 10).float().mean()),
            clamp_fraction=float(((raw <= -20) | (raw >= 10)).float().mean()),
            sigma_mean=float(sigma.mean()), sigma_rms=float(sigma.square().mean().sqrt()),
            sigma_p01=float(sigma_quantiles[0]), sigma_p50=float(sigma_quantiles[1]),
            sigma_p99=float(sigma_quantiles[2]),
            sigma_min=float(sigma.min()), sigma_max=float(sigma.max()),
            perturbation_rms=float(delta.square().mean().sqrt()),
            perturbation_max=float(delta.abs().max()),
            k_mu=float(k_mu), k_sigma=float(k_sigma), kl=float(k_mu+k_sigma),
            epsilon_sha256=None if epsilon is None else hashlib.sha256(
                epsilon.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        assert all(math.isfinite(v) for v in result.values() if isinstance(v, float))
        return result


def initialize_logvar(model):
    with torch.no_grad():
        model.log_variance.weight.zero_()
        model.log_variance.bias.fill_(-6.)
    model.log_variance.requires_grad_(True)


def old_parameter_names(model):
    return [name for name, p in model.named_parameters() if p.requires_grad]


def new_parameter_names(model):
    return [name for name, _ in model.log_variance.named_parameters(prefix='log_variance')]


def make_optimizer(model, source_optimizer=None, expected_old_count=412, expected_step=START):
    """Load the AE group before activating logvar; preserve all inherited moments."""
    old = [p for p in model.parameters() if p.requires_grad]
    assert len(old) == expected_old_count and not model.log_variance.weight.requires_grad
    opt = torch.optim.AdamW(old, lr=1e-4, betas=(.9, .999), eps=1e-8,
                             weight_decay=.01, foreach=True)
    if source_optimizer is not None:
        assert len(source_optimizer['param_groups']) == 1
        opt.load_state_dict(source_optimizer)
        assert len(opt.state) == len(old)
        assert all(int(opt.state[p]['step']) == expected_step for p in old)
    initialize_logvar(model)
    new = list(model.log_variance.parameters())
    opt.add_param_group(dict(params=new, lr=1e-4, betas=(.9, .999), eps=1e-8,
                             weight_decay=.01, foreach=True))
    assert len({id(p) for group in opt.param_groups for p in group['params']}) == len(old)+2
    if source_optimizer is not None:
        assert all(p not in opt.state for p in new)
    for group in opt.param_groups:
        assert group['lr'] == 1e-4 and group['betas'] == (.9, .999)
        assert group['eps'] == 1e-8 and group['weight_decay'] == .01 and group['foreach'] is True
    return opt


def group_steps(opt):
    values = []
    for group in opt.param_groups:
        steps = {int(opt.state[p]['step']) for p in group['params'] if p in opt.state}
        values.append(sorted(steps))
    return values


def gradient_norm(parameters):
    return math.sqrt(sum(float(p.grad.detach().double().square().sum()) for p in parameters if p.grad is not None))


def snapshot_parameters(opt):
    return [[p.detach().clone() for p in g['params']] for g in opt.param_groups]


def measured_deltas(model, opt, before):
    names = optimizer_group_names(model, opt)
    groups = [0., 0.]
    modules = {key: 0. for key in ('encoder_mu', 'decoder', 'edge_head', 'face_head', 'logvar')}
    for index, (group, group_before, group_names) in enumerate(zip(opt.param_groups, before, names)):
        for p, old, name in zip(group['params'], group_before, group_names):
            squared = float((p.detach()-old).double().square().sum())
            groups[index] += squared
            if name.startswith('log_variance.'):
                key = 'logvar'
            elif name.startswith('edge_embedding.'):
                key = 'edge_head'
            elif name.startswith('face_embedding.'):
                key = 'face_head'
            elif name.startswith(('latent_input.', 'decoder_blocks.', 'decoder_output_norm.')):
                key = 'decoder'
            else:
                key = 'encoder_mu'
            modules[key] += squared
    return [math.sqrt(x) for x in groups], {key: math.sqrt(x) for key, x in modules.items()}


def optimizer_group_names(model, opt):
    lookup = {id(p): name for name, p in model.named_parameters()}
    return [[lookup[id(p)] for p in group['params']] for group in opt.param_groups]
