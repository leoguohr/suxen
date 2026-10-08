#!/usr/bin/env python3
"""CPU/std-library reporting only. Never imports training code or opens weights.

python report_results.py [--root EXPERIMENT_ROOT] [--output-dir DIRECTORY]
python report_results.py --self-test
Exit codes: 0 = complete or honestly pending; 2 = inconsistent/invalid evidence.
"""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

START, UPDATES, BETA = 34220, 500, 1e-6
MU_STEPS = (0, 100, 250, 500)
SEEDS = (861001, 861002, 861003, 861004, 861005)
MODULES = ('encoder_mu', 'decoder', 'edge_head', 'face_head', 'logvar')
POSTERIOR = ('sigma_mean', 'sigma_rms', 'sigma_p01', 'sigma_p50', 'sigma_p99',
             'sigma_min', 'sigma_max', 'logvar_mean', 'raw_logvar_mean',
             'clamp_low_fraction', 'clamp_high_fraction', 'clamp_fraction',
             'perturbation_rms', 'perturbation_max', 'kl', 'k_mu', 'k_sigma')
QUANTILE_NOTE = 'sigma p01/p50/p99: each mesh first computes its latent-element quantile; then mesh quantiles are averaged equally. These are not pooled quantiles.'
SCOPE = ('仅描述原 AE 28/100 工作点上的小噪声、固定 beta=1e-6 的 KL 适应；'
         '不能据此宣称成功 AE、全 100 严格成功或保证全 Gaussian 扰动下重建。')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def close(a, b):
    return finite(a) and finite(b) and math.isclose(a, b, rel_tol=2e-6, abs_tol=1e-8)


def is_sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def condition_paths():
    return [(step, group) for step in MU_STEPS for group in
            (['mu'] + ([f'noise-{seed}' for seed in SEEDS] if step in (0, 500) else []))]


def uid_seed(seed, uid):
    text = f'ownv2-fixed100-vae/eval/{seed}/{uid}'.encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'little') & ((1 << 63)-1)


def metrics(meshes):
    result = {}
    for task in ('edge', 'face'):
        totals = {k: 0 for k in ('tp', 'fp', 'fn', 'tn')}
        for mesh in meshes:
            for k in totals:
                value = mesh[task][k]
                require(type(value) is int and value >= 0, f'{mesh["uid"]}: invalid {task}.{k}')
                totals[k] += value
        totals['micro_f1'] = 2*totals['tp']/max(2*totals['tp']+totals['fp']+totals['fn'], 1)
        result[task] = totals
    result['perfect_uids'] = [m['uid'] for m in meshes
        if all(m[t]['fp'] == m[t]['fn'] == 0 for t in ('edge', 'face'))]
    result['joint_perfect'] = len(result['perfect_uids'])
    return result


def validated_metrics(document, expected_uids=None):
    require(document.get('complete') is True, 'complete flag is not true')
    meshes = document['meshes']
    ids = [m['uid'] for m in meshes]
    require(len(ids) == len(set(ids)) == 100, 'evaluation must contain 100 unique UIDs')
    if expected_uids is not None:
        require(set(ids) == set(expected_uids), 'evaluation UID set differs from source100')
    require(all(m.get('complete') is True for m in meshes), 'incomplete mesh in complete evaluation')
    actual = metrics(meshes)
    for task in ('edge', 'face'):
        for key, value in actual[task].items():
            matches = close(document[task][key], value) if key == 'micro_f1' else document[task][key] == value
            require(matches, f'aggregate {task}.{key} disagrees with per-UID counts')
    require(document['joint_perfect'] == actual['joint_perfect'], 'joint_perfect disagrees with per-UID counts')
    require(set(document['perfect_uids']) == set(actual['perfect_uids']), 'perfect_uids disagree with per-UID counts')
    return actual


def checkpoint_metadata(value, step):
    require(value['completed_updates'] == START+step, 'checkpoint global step mismatch')
    require(value['new_updates'] == step, 'checkpoint new update mismatch')
    require(isinstance(value['path'], str) and value['path'], 'checkpoint path missing')
    require(type(value['bytes']) is int and value['bytes'] > 0, 'checkpoint declared size invalid')
    require(is_sha(value['sha256']) and is_sha(value['model_state_sha256']), 'checkpoint declared SHA invalid')
    return {key: value[key] for key in
            ('path', 'bytes', 'sha256', 'model_state_sha256', 'completed_updates', 'new_updates')}


def posterior_summary(meshes):
    values = [m['posterior'] for m in meshes]
    for mesh, row in zip(meshes, values):
        for key in POSTERIOR:
            require(finite(row.get(key)), f'{mesh["uid"]}: missing/nonfinite posterior {key}')
        for key in ('clamp_low_fraction', 'clamp_high_fraction', 'clamp_fraction'):
            require(0 <= row[key] <= 1, f'{mesh["uid"]}: invalid {key}')
        require(close(row['kl'], row['k_mu']+row['k_sigma']), 'KL != Kmu + Ksigma')
        require(0 <= row['sigma_min'] <= row['sigma_p01'] <= row['sigma_p50'] <=
                row['sigma_p99'] <= row['sigma_max'], 'unordered sigma quantiles')
    return dict(mesh_equal_mean={key: sum(row[key] for row in values)/len(values) for key in POSTERIOR},
                sigma_global_min=min(row['sigma_min'] for row in values),
                sigma_global_max=max(row['sigma_max'] for row in values),
                perturbation_global_max=max(row['perturbation_max'] for row in values),
                quantile_definition=QUANTILE_NOTE,
                rms_definition='Arithmetic mean of within-mesh RMS, not a pooled RMS.',
                clamp_definition='raw logvar <= -20 / >= 10 (inclusive), then equal mesh mean.')


def numeric_summary(values):
    if not values:
        return None
    require(all(finite(x) for x in values), 'nonfinite summary input')
    return dict(count=len(values), first=values[0], last=values[-1],
                mean=sum(values)/len(values), min=min(values), max=max(values))


class Evidence:
    def __init__(self, root):
        self.root = root
        self.checks = []
        self.inputs = []

    def check(self, name, status, detail):
        self.checks.append(dict(name=name, status=status, detail=detail))

    def read(self, path):
        if not path.exists():
            return None
        raw = path.read_bytes()
        self.inputs.append(dict(path=str(path.relative_to(self.root)), bytes=len(raw),
                                sha256=hashlib.sha256(raw).hexdigest()))
        try:
            return json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
        except (ValueError, UnicodeError) as exc:
            self.check(str(path.relative_to(self.root)), 'failed', f'Invalid JSON: {exc}')
            return None

    def validate(self, name, fn):
        try:
            result = fn()
            self.check(name, 'passed', 'checked')
            return result
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            self.check(name, 'failed', str(exc))
            return None


def read_updates(evidence):
    path = evidence.root/'run/updates.jsonl'
    if not path.exists():
        return [], False
    raw = path.read_bytes()
    evidence.inputs.append(dict(path='run/updates.jsonl', bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    lines = raw.splitlines(keepends=True)
    partial = bool(lines and not lines[-1].endswith(b'\n'))
    if partial:
        evidence.check('updates_live_tail', 'pending', 'Trailing uncommitted line ignored; rerun after writer commits it.')
        lines = lines[:-1]
    rows = []
    for index, line in enumerate(lines, 1):
        try:
            rows.append(json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x))))
        except (ValueError, UnicodeError) as exc:
            evidence.check('updates_jsonl', 'failed', f'Line {index}: {exc}')
            break
    return rows, partial


def validate_training(rows, uids):
    require(len(rows) <= UPDATES, 'more than 500 update records')
    participation = Counter({uid: 0 for uid in uids})
    loss_keys = ('reconstruction_edge_mean', 'reconstruction_face_mean', 'kl_mean',
                 'k_mu_mean', 'k_sigma_mean', 'total_mean', 'gradient_norm_before_clip', 'clip_coefficient')
    for index, row in enumerate(rows, 1):
        require(row['update'] == START+index and row['new_update'] == index,
                f'record {index}: updates are not consecutive 34221..34720')
        require(row['optimizer_group_steps'] == [[START+index], [index]], f'update {index}: Adam steps mismatch')
        require(row['optimizer_group_lr'] == [1e-4, 1e-4], f'update {index}: LR mismatch')
        require(row['beta'] == BETA, f'update {index}: beta mismatch')
        require((row['epoch'], row['group']) == divmod(START+index-1, 20), 'logged epoch/group mismatch')
        batch = row['uids']
        require(len(batch) == len(set(batch)) == 5 and set(batch) <= set(uids), f'update {index}: invalid batch')
        require([m['uid'] for m in row['meshes']] == batch, 'per-mesh records differ from batch UIDs')
        participation.update(batch)
        for key in loss_keys:
            require(finite(row[key]), f'update {index}: missing/nonfinite {key}')
        require(row['gradient_norm_before_clip'] >= 0, 'negative gradient norm')
        require(0 < row['clip_coefficient'] <= 1, 'invalid clip coefficient')
        require(close(row['clip_coefficient'], min(1., 1./(row['gradient_norm_before_clip']+1e-6))), 'clip coefficient mismatch')
        require(close(row['kl_mean'], row['k_mu_mean']+row['k_sigma_mean']), 'training KL reduction mismatch')
        require(close(row['total_mean'], row['reconstruction_edge_mean']+row['reconstruction_face_mean']+BETA*row['kl_mean']), 'training total loss mismatch')
        for module in MODULES:
            delta = row['module_delta_l2_fp32'][module]
            require(finite(delta) and delta >= 0, f'update {index}: invalid {module} displacement')
        posterior_summary(row['meshes'])
    require(all(n <= 25 for n in participation.values()), 'a UID participated more than 25 times')
    if len(rows) == UPDATES:
        require(set(participation.values()) == {25}, 'not every UID participated exactly 25 times')
    if not rows:
        return dict(record_count=0, participation=dict(participation), optimizer_steps=None)
    modules = {}
    for module in MODULES:
        vals = [row['module_delta_l2_fp32'][module] for row in rows]
        modules[module] = dict(numeric_summary(vals), nonzero_updates=sum(v > 0 for v in vals),
                              sum_step_l2=sum(vals))
    return dict(record_count=len(rows), first_global_step=rows[0]['update'], last_global_step=rows[-1]['update'],
        participation=dict(participation), optimizer_steps=rows[-1]['optimizer_group_steps'],
        loss_and_clip={key: numeric_summary([row[key] for row in rows]) for key in loss_keys},
        clipping_active_updates=sum(row['clip_coefficient'] < 1 for row in rows),
        semantic_module_delta=modules,
        delta_definition='Actual FP32 before/after subtraction per optimizer update. sum_step_l2 is path length, not endpoint displacement.',
        posterior=posterior_summary([mesh for row in rows for mesh in row['meshes']]))


def validate_audit(audit, rows, uids):
    require(audit['scope'] == 'next_optimizer_batch_5', 'audit scope is not next_optimizer_batch_5')
    require(audit['optimizer_updates'] == 0 and audit['beta'] == BETA, 'audit updated optimizer or beta differs')
    require(audit['epoch'] == 1711 and audit['group'] == 0, 'audit cursor mismatch')
    require(len(audit['uids']) == len(set(audit['uids'])) == 5 and set(audit['uids']) <= set(uids), 'audit UIDs invalid')
    require([m['uid'] for m in audit['meshes']] == audit['uids'], 'audit mesh UIDs mismatch')
    for key in ('reconstruction_gradient_norm', 'kl_gradient_norm', 'beta_kl_gradient_norm',
                'reconstruction_beta_kl_gradient_inner_product', 'total_gradient_norm'):
        require(finite(audit[key]), f'missing/nonfinite audit {key}')
    require(audit['logvar_reconstruction_gradient_nonzero'] is True and audit['logvar_kl_gradient_nonzero'] is True,
            'logvar gradient evidence is not positive')
    require(close(audit['beta_kl_gradient_norm'], BETA*audit['kl_gradient_norm']), 'weighted KL gradient norm mismatch')
    squared = audit['reconstruction_gradient_norm']**2+audit['beta_kl_gradient_norm']**2+2*audit['reconstruction_beta_kl_gradient_inner_product']
    require(math.isclose(audit['total_gradient_norm']**2, squared, rel_tol=1e-4, abs_tol=1e-8), 'gradient norm / inner product identity mismatch')
    if rows:
        require(audit['uids'] == rows[0]['uids'], 'audit is not first real optimizer batch')
        for a, b in zip(audit['meshes'], rows[0]['meshes']):
            ha, hb = a['posterior']['epsilon_sha256'], b['posterior']['epsilon_sha256']
            require(is_sha(ha) and ha == hb, f'{a["uid"]}: audit and first train epsilon differ')
    return audit


def build_report(root):
    root = Path(root).resolve()
    evidence = Evidence(root)
    baseline = evidence.read(root/'source/ae_mu_baseline.json')
    base = None
    if baseline is not None:
        def validate_base():
            value = validated_metrics(baseline)
            require(value['joint_perfect'] == 28, 'source baseline is not 28/100')
            require(baseline['checkpoint']['completed_updates'] == START, 'source baseline step mismatch')
            require(is_sha(baseline['checkpoint']['sha256']), 'source checkpoint metadata SHA missing')
            return value
        base = evidence.validate('source_baseline', validate_base)
    else:
        evidence.check('source_baseline', 'pending', 'source/ae_mu_baseline.json unavailable')
    uids = [m['uid'] for m in baseline['meshes']] if base is not None else []
    base_success = set(base['perfect_uids']) if base is not None else set()
    checkpoints = {}
    for step in MU_STEPS:
        doc = evidence.read(root/'run/checkpoints'/f'vae-{step:04d}.json')
        if doc is None:
            evidence.check(f'checkpoint_{step}', 'pending', 'Checkpoint metadata not available')
        else:
            checked = evidence.validate(f'checkpoint_{step}', lambda d=doc,s=step: checkpoint_metadata(d,s))
            if checked is not None:
                checkpoints[str(step)] = checked

    evaluations = []
    expected_paths = set()
    for step, group in condition_paths():
        relpath = f'run/evaluations/update-{START+step:08d}/{group}/complete.json'
        expected_paths.add(relpath)
        row = dict(new_update=step, global_step=START+step, condition=group, path=relpath, status='pending')
        doc = evidence.read(root/relpath)
        if doc is None or base is None:
            evidence.check(f'eval_{step}_{group}', 'pending', 'Complete 100-mesh evidence not available')
            evaluations.append(row)
            continue
        def validate_eval():
            actual = validated_metrics(doc, uids)
            expected_seed = None if group == 'mu' else int(group.split('-')[1])
            identity = doc['identity']
            cp = checkpoint_metadata(doc['checkpoint'], step)
            require(identity['group'] == group and identity['evaluation_seed'] == expected_seed, 'condition identity mismatch')
            require(identity['manifest_sha256'] == baseline['identity']['manifest_sha256'], 'data manifest mismatch')
            require(identity['checkpoint_sha256'] == cp['sha256'], 'evaluation checkpoint SHA mismatch')
            if str(step) in checkpoints:
                require(cp == checkpoints[str(step)], 'evaluation and checkpoint metadata differ')
            require(doc.get('native_forward') is True and doc.get('optimizer_updates_during_evaluation') == 0,
                    'evaluation forward/update evidence mismatch')
            post = posterior_summary(doc['meshes'])
            for mesh in doc['meshes']:
                require(mesh['identity'] == dict(identity, uid=mesh['uid']), 'per-UID condition binding mismatch')
                p = mesh['posterior']
                require(p['evaluation_seed'] == expected_seed, 'per-UID evaluation seed mismatch')
                if expected_seed is None:
                    require(p['epsilon_sha256'] is None and p['uid_noise_seed'] is None, 'mu evaluation contains epsilon')
                else:
                    require(is_sha(p['epsilon_sha256']), 'sampled evaluation epsilon SHA missing')
                    require(p['uid_noise_seed'] == uid_seed(expected_seed, mesh['uid']), 'noise seed is not fixed by group/UID')
            current = set(actual['perfect_uids'])
            retention = dict(retained_uids=sorted(base_success & current), lost_uids=sorted(base_success-current),
                             added_uids=sorted(current-base_success))
            retention.update(retained=len(retention['retained_uids']), lost=len(retention['lost_uids']),
                             added=len(retention['added_uids']))
            if step == 0 and group == 'mu':
                reference = {m['uid']:m for m in baseline['meshes']}
                for mesh in doc['meshes']:
                    for task in ('edge', 'face'):
                        require(mesh[task] == reference[mesh['uid']][task], 'step0 mu does not reproduce source per-UID counts')
            return dict(actual, posterior=post, retention_from_original28=retention,
                        checkpoint=cp, per_uid=doc['meshes'])
        validated = evidence.validate(f'eval_{step}_{group}', validate_eval)
        if validated is None:
            row['status'] = 'failed'
        else:
            row.update(validated, status='complete')
        evaluations.append(row)
    unexpected = sorted(str(p.relative_to(root)) for p in (root/'run/evaluations').glob('**/complete.json')
                        if str(p.relative_to(root)) not in expected_paths)
    if unexpected:
        evidence.check('unexpected_evaluation_paths', 'failed', unexpected)

    valid = {(r['new_update'], r['condition']):r for r in evaluations if r['status'] == 'complete'}
    pairs = []
    for seed in SEEDS:
        first, last = valid.get((0,f'noise-{seed}')), valid.get((500,f'noise-{seed}'))
        start_map = {m['uid']:m for m in first['per_uid']} if first else {}
        end_map = {m['uid']:m for m in last['per_uid']} if last else {}
        for uid in uids:
            a = start_map.get(uid, {}).get('posterior', {}).get('epsilon_sha256')
            b = end_map.get(uid, {}).get('posterior', {}).get('epsilon_sha256')
            status = 'pending' if a is None or b is None else ('matched' if a == b else 'failed')
            pairs.append(dict(seed=seed,uid=uid,epsilon_sha256_step0=a,epsilon_sha256_step500=b,status=status))
    counts = Counter(p['status'] for p in pairs)
    pair_status = 'failed' if counts['failed'] else ('passed' if len(pairs) == counts['matched'] == 500 else 'pending')
    evidence.check('500_paired_epsilon_hashes', pair_status, dict(expected=500, **dict(counts)))

    rows, partial = read_updates(evidence)
    training = evidence.validate('training_record_consistency', lambda: validate_training(rows,uids)) if base is not None else None
    if training is not None:
        evidence.check('500_contiguous_updates_and_25_participations',
                       'passed' if len(rows) == UPDATES and not partial else 'pending',
                       dict(records=len(rows), expected=UPDATES, final_optimizer_steps=training['optimizer_steps']))
    audit = evidence.read(root/'run/step0_gradient_audit.json')
    if audit is None:
        evidence.check('step0_gradient_audit', 'pending', 'Audit JSON not yet available')
    elif base is not None:
        audit = evidence.validate('step0_gradient_audit', lambda: validate_audit(audit,rows,uids))
        if audit is not None and not rows:
            evidence.check('audit_matches_first_train_epsilon', 'pending', 'No first training record yet')

    statuses = [c['status'] for c in evidence.checks]
    status = 'invalid' if 'failed' in statuses else ('pending' if 'pending' in statuses else 'complete')
    return dict(schema_version=1, generated_at_utc=datetime.now(timezone.utc).isoformat(),
        status=status, scope=SCOPE, root=str(root), checks=evidence.checks, inputs=evidence.inputs,
        baseline=None if base is None else dict(base, checkpoint=baseline['checkpoint']),
        checkpoints=checkpoints, checkpoint_verification='Path/bytes/SHA are declarations from JSON metadata; weight files were neither loaded nor rehashed.',
        evaluations=evaluations, epsilon_pairing=dict(expected=500,counts=dict(counts),pairs=pairs),
        training=training, step0_gradient_audit=audit, quantile_definition=QUANTILE_NOTE)


def render_markdown(report):
    def fmt(x):
        return 'pending' if x is None else (f'{x:.7g}' if isinstance(x,float) else str(x))
    lines = ['# Fixed100 小噪声 VAE 结果', '', f'状态：**{report["status"]}**。生成时间：{report["generated_at_utc"]}。',
             '', report['scope'], '', '完整性以 JSON 与已提交的 updates.jsonl 为准；pending 不作完整结果解释。',
             '', '## 完整性检查', '', '| 检查 | 状态 | 详情 |', '|---|---|---|']
    for check in report['checks']:
        detail = json.dumps(check['detail'], ensure_ascii=False) if not isinstance(check['detail'],str) else check['detail']
        lines.append(f'| {check["name"]} | {check["status"]} | {detail.replace(chr(10), " ").replace("|", "/")} |')
    lines += ['', '## 原工作点与各评价路径', '',
        'microF1 由逐 UID TP/FP/FN 汇总重算。保留/丢失均相对原 28 个 strict joint 成功 UID；新增来自原 72 个失败 UID。完整 UID 名单见 comparison.json。',
        '', '| 新增步/路径 | 状态 | Edge FP | Edge FN | Face FP | Face FN | Edge microF1 | Face microF1 | joint | 保留 | 丢失 | 新增 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    base = report['baseline']
    if base:
        lines.append(f'| source AE / {START} | baseline | {base["edge"]["fp"]} | {base["edge"]["fn"]} | {base["face"]["fp"]} | {base["face"]["fn"]} | {base["edge"]["micro_f1"]:.10f} | {base["face"]["micro_f1"]:.10f} | 28/100 | 28 | 0 | 0 |')
    for row in report['evaluations']:
        label = f'{row["new_update"]}/{row["condition"]}'
        if row['status'] != 'complete':
            lines.append(f'| {label} | {row["status"]} | — | — | — | — | — | — | — | — | — | — |')
            continue
        e,f,r = row['edge'],row['face'],row['retention_from_original28']
        lines.append(f'| {label} | complete | {e["fp"]} | {e["fn"]} | {f["fp"]} | {f["fn"]} | {e["micro_f1"]:.10f} | {f["micro_f1"]:.10f} | {row["joint_perfect"]}/100 | {r["retained"]} | {r["lost"]} | {r["added"]} |')
    lines += ['', '## 后验统计', '',
        '**sigma p01/p50/p99 均为先在每个 mesh 的 latent 元素中计算分位数，再对 mesh 等权平均；不是合并所有 mesh 后的分位数。** sigma RMS 与扰动 RMS 也先在 mesh 内计算再平均。KL/Kmu/Ksigma 在 mesh 内按 vertex × channel 取均值，外层 mesh 等权。',
        '', '| 新增步/路径 | sigma均值 | sigma RMS | p01 | p50 | p99 | Kmu | Ksigma | raw≤−20 | raw≥10 | 扰动RMS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in report['evaluations']:
        if row['status'] == 'complete':
            p = row['posterior']['mesh_equal_mean']
            keys = ('sigma_mean','sigma_rms','sigma_p01','sigma_p50','sigma_p99','k_mu','k_sigma','clamp_low_fraction','clamp_high_fraction','perturbation_rms')
            lines.append('| '+f'{row["new_update"]}/{row["condition"]} | '+' | '.join(fmt(p[k]) for k in keys)+' |')
    lines += ['', '## 训练与梯度', '']
    training = report['training']
    if training and training['record_count']:
        lines += [f'已提交 {training["record_count"]}/500 次更新；最终已记录 optimizer steps={training["optimizer_steps"]}。每 UID 参与次数及完整统计见 comparison.json。',
            '', f'clip 系数 < 1 的更新：{training["clipping_active_updates"]}。下表是更新前 loss/gradient/clip 的逐步统计。',
            '', '| 字段 | 首条 | 末条 | 均值 | 最小 | 最大 |', '|---|---:|---:|---:|---:|---:|']
        for key, value in training['loss_and_clip'].items():
            lines.append('| '+key+' | '+' | '.join(fmt(value[k]) for k in ('first','last','mean','min','max'))+' |')
        lines += ['', '各模块位移来自每次更新前后 FP32 参数实差。sum_step_l2 是各步范数之和，不是起末参数净位移。', '',
            '| 模块 | 非零更新数 | 首步L2 | 末步L2 | 平均L2 | sum_step_l2 |', '|---|---:|---:|---:|---:|---:|']
        for key, value in training['semantic_module_delta'].items():
            lines.append('| '+key+' | '+' | '.join(fmt(value[k]) for k in ('nonzero_updates','first','last','mean','sum_step_l2'))+' |')
        lines += ['', '训练后验为已提交训练 mesh-forward 的等权统计；sigma 分位数同样先取 mesh 内分位数再平均。',
                  '', '```json', json.dumps(training['posterior']['mesh_equal_mean'], indent=2), '```']
    else:
        lines.append('训练记录 pending 或未通过一致性检查。')
    audit = report['step0_gradient_audit']
    if audit:
        lines += ['', 'step0 梯度审计 scope=next_optimizer_batch_5：先对下一真实 batch 的五个 mesh 梯度按 1/5 累积，再求范数/内积；不是 full100 梯度，也不是 mesh 范数平均。', '', '```json',
                  json.dumps({k:v for k,v in audit.items() if k != 'meshes'}, indent=2, ensure_ascii=False), '```']
    else:
        lines += ['', 'step0 梯度审计 pending 或无效。']
    lines += ['', '## 固定 epsilon 配对', '',
              f'预期 5×100=500 对；当前：{json.dumps(report["epsilon_pairing"]["counts"],ensure_ascii=False)}。逐对 UID、seed、起末 SHA 和不匹配项见 comparison.json。',
              '', '## Checkpoint 元数据', '',
              '以下路径、大小、SHA 仅引用 JSON 元数据，未读取或重新 hash 大权重文件。', '',
              '| 新增步 | 路径 | bytes | SHA-256 |', '|---|---|---:|---|']
    for step in MU_STEPS:
        cp = report['checkpoints'].get(str(step))
        lines.append(f'| {step} | {cp["path"] if cp else "pending"} | {cp["bytes"] if cp else "—"} | {cp["sha256"] if cp else "—"} |')
    return '\n'.join(lines)+'\n'


def write_reports(report, directory):
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {'comparison.json': json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n',
               'REPORT.md': render_markdown(report)}
    for name, text in outputs.items():
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=directory, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, directory/name)


def self_test():
    """Small scalar JSON fixtures exercise the actual reader/validator/renderer."""
    def digest(text):
        return hashlib.sha256(str(text).encode()).hexdigest()
    def put(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
    def posterior(uid, seed=None, train_step=None):
        value = dict(sigma_mean=.05, sigma_rms=.05, sigma_p01=.05, sigma_p50=.05,
            sigma_p99=.05, sigma_min=.05, sigma_max=.05, logvar_mean=-6., raw_logvar_mean=-6.,
            clamp_low_fraction=0., clamp_high_fraction=0., clamp_fraction=0.,
            perturbation_rms=0. if seed is None and train_step is None else .05,
            perturbation_max=0. if seed is None and train_step is None else .1,
            kl=3., k_mu=1., k_sigma=2., evaluation_seed=seed,
            uid_noise_seed=None if seed is None else uid_seed(seed,uid),
            epsilon_sha256=None if seed is None else digest(f'eval/{seed}/{uid}'))
        if train_step is not None:
            value['epsilon_sha256'] = digest(f'train/{train_step}/{uid}')
        return value
    with tempfile.TemporaryDirectory(prefix='ownv2-report-cpu-') as temporary:
        root = Path(temporary)
        ids = [f'u{i:03d}' for i in range(100)]
        meshes = [dict(uid=uid, complete=True, vertices=7,
            edge=dict(tp=3091,fp=0,fn=0,tn=100),
            face=dict(tp=2042,fp=0 if i < 28 else 1,fn=0,tn=2)) for i,uid in enumerate(ids)]
        meshes[0]['edge']['tp'] += 30; meshes[0]['face']['tp'] += 7
        meshes[28]['edge'].update(fp=51,fn=64)
        meshes[28]['face'].update(fp=1437,fn=123)
        baseline = dict(complete=True, meshes=meshes, **metrics(meshes),
            identity=dict(manifest_sha256=digest('manifest')),
            checkpoint=dict(completed_updates=START,sha256=digest('source'),path='/synthetic/source.pt',bytes=123))
        put(root/'source/ae_mu_baseline.json',baseline)
        for step,group in condition_paths():
            cp = dict(path=f'/synthetic/run/checkpoints/vae-{step:04d}.pt',bytes=123,
                sha256=digest(f'checkpoint/{step}'), model_state_sha256=digest(f'model/{step}'),
                completed_updates=START+step,new_updates=step)
            put(root/'run/checkpoints'/f'vae-{step:04d}.json',cp)
            seed = None if group == 'mu' else int(group.split('-')[1])
            identity = dict(checkpoint_sha256=cp['sha256'],manifest_sha256=digest('manifest'),
                            group=group,evaluation_seed=seed)
            current = copy.deepcopy(meshes)
            for mesh in current:
                mesh.update(identity=dict(identity,uid=mesh['uid']),posterior=posterior(mesh['uid'],seed))
            if step == 500 and group == 'mu':
                current[0]['face']['fp'] = 1
                for task in ('edge','face'): current[28][task].update(fp=0,fn=0)
            doc = dict(complete=True,identity=identity,checkpoint=cp,meshes=current,
                       native_forward=True,optimizer_updates_during_evaluation=0,**metrics(current))
            put(root/f'run/evaluations/update-{START+step:08d}/{group}/complete.json',doc)
        rows = []
        for index in range(1,501):
            batch = ids[((index-1)%20)*5:((index-1)%20)*5+5]
            epoch,group = divmod(START+index-1,20)
            rows.append(dict(update=START+index,new_update=index,epoch=epoch,group=group,uids=batch,
                optimizer_group_steps=[[START+index],[index]],optimizer_group_lr=[1e-4,1e-4],beta=BETA,
                meshes=[dict(uid=u,posterior=posterior(u,train_step=index)) for u in batch],
                reconstruction_edge_mean=.2,reconstruction_face_mean=.3,kl_mean=3.,k_mu_mean=1.,k_sigma_mean=2.,
                total_mean=.5+BETA*3,gradient_norm_before_clip=2.,clip_coefficient=1/(2+1e-6),
                module_delta_l2_fp32={key:.1 for key in MODULES}))
        log = root/'run/updates.jsonl'
        def write_rows(values):
            log.write_text(''.join(json.dumps(row)+'\n' for row in values),encoding='utf-8')
        write_rows(rows)
        audit = dict(scope='next_optimizer_batch_5',optimizer_updates=0,beta=BETA,epoch=1711,group=0,
            uids=rows[0]['uids'],meshes=rows[0]['meshes'],reconstruction_gradient_norm=1.,kl_gradient_norm=2.,
            beta_kl_gradient_norm=2*BETA,reconstruction_beta_kl_gradient_inner_product=0.,
            total_gradient_norm=math.sqrt(1+(2*BETA)**2),
            logvar_reconstruction_gradient_nonzero=True,logvar_kl_gradient_nonzero=True)
        put(root/'run/step0_gradient_audit.json',audit)
        report = build_report(root)
        require(report['status'] == 'complete', str([c for c in report['checks'] if c['status'] != 'passed']))
        require(report['epsilon_pairing']['counts'] == {'matched':500}, 'pairing test failed')
        require(set(report['training']['participation'].values()) == {25}, 'participation test failed')
        end = next(r for r in report['evaluations'] if r['new_update'] == 500 and r['condition'] == 'mu')
        require((end['retention_from_original28']['retained'],end['retention_from_original28']['lost'],
                 end['retention_from_original28']['added']) == (27,1,1), 'retention test failed')
        write_reports(report,root/'output')
        require('不是合并所有 mesh 后的分位数' in (root/'output/REPORT.md').read_text(), 'quantile label missing')
        require(not list(root.rglob('*.pt')), 'test unexpectedly required checkpoint weights')
        print('PASS complete report: 500 updates, UID counts, Adam steps, 14 evals, 500 epsilon pairs, retention, rendering; no weights')
        noisy_path = root/f'run/evaluations/update-{START+500:08d}/noise-{SEEDS[0]}/complete.json'
        original = json.loads(noisy_path.read_text())
        corrupt = copy.deepcopy(original); corrupt['meshes'][0]['posterior']['epsilon_sha256'] = digest('wrong-noise')
        put(noisy_path,corrupt)
        bad = build_report(root)
        require(bad['status'] == 'invalid' and bad['epsilon_pairing']['counts']['failed'] == 1, 'noise mismatch accepted')
        put(noisy_path,original)
        altered = copy.deepcopy(rows); altered[-1]['optimizer_group_steps'] = [[34720],[499]]
        write_rows(altered)
        require(build_report(root)['status'] == 'invalid', 'wrong final Adam step accepted')
        write_rows(rows[:1]+rows[2:])
        require(build_report(root)['status'] == 'invalid', 'update gap accepted')
        altered = copy.deepcopy(rows); altered[-1]['uids'] = ids[:5]
        altered[-1]['meshes'] = [dict(uid=u,posterior=posterior(u,train_step=500)) for u in ids[:5]]
        write_rows(altered)
        require(build_report(root)['status'] == 'invalid', 'unequal participation accepted')
        print('PASS invalid evidence: epsilon mismatch, wrong Adam step, update gap, unequal participation')
        write_rows(rows[:7])
        with log.open('a') as handle: handle.write('{"update":')
        for step,group in condition_paths():
            if step == 500:
                (root/f'run/evaluations/update-{START+step:08d}/{group}/complete.json').unlink()
        partial = build_report(root)
        require(partial['status'] == 'pending' and partial['training']['record_count'] == 7, 'live partial input misclassified')
        require(partial['epsilon_pairing']['counts'] == {'pending':500}, 'missing endpoints misclassified')
        write_reports(partial,root/'partial-output')
        print('PASS pending evidence: missing endpoints and partial live log tail; reports generated')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output-dir', type=Path, help='Default: experiment root; only REPORT.md and comparison.json are written')
    parser.add_argument('--self-test', action='store_true', help='Run temporary synthetic CPU fixtures; no training code imports')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    report = build_report(args.root)
    out = (args.output_dir or args.root).resolve()
    write_reports(report, out)
    print(json.dumps(dict(status=report['status'], report=str(out/'REPORT.md'), comparison=str(out/'comparison.json')), ensure_ascii=False))
    return 2 if report['status'] == 'invalid' else 0


if __name__ == '__main__':
    raise SystemExit(main())
