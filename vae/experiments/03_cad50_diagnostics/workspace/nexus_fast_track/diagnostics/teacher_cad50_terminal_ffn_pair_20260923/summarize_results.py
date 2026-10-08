"""Derive comparison tables from completed evaluations without touching models."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'repro_outputs'
STEPS = [0, 25, 50, 75, 100]


def read(path):
    return json.loads(path.read_text())


def metrics(evaluation):
    result = {'strict': evaluation['joint_perfect'], 'edge_strict': evaluation['edge_perfect'],
        'face_strict': evaluation['face_perfect'], 'retained': evaluation['retained'],
        'lost': evaluation['lost'], 'new': evaluation['new']}
    for task in ['edge', 'face']:
        for key in ['tp', 'fp', 'fn', 'micro_f1']:
            result[f'{task}_{key}'] = evaluation['counts'][task][key]
    for key in ['face_fn_missing', 'face_fn_present', 'face_fp_inside_pool', 'face_fp_outside_pool']:
        result[key] = evaluation[key]
    return result


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    evaluations = {}
    trajectory, per_mesh, step_edge = [], [], []
    for branch, directory in [('S_original', 'Control_reused'), ('H_terminal_ffn', 'H_terminal_ffn')]:
        root = ROOT / directory
        done = read(root / 'complete.json')
        assert done['new_updates'] == 100 and done['completed_updates'] == 2600
        evaluations[branch] = {}
        for step in STEPS:
            e = read(root / f'eval-new{step:04d}.json')
            values = metrics(e)
            large = metrics(e['size_groups']['66_to_274'])
            evaluations[branch][str(step)] = dict(all50=values, large16=large)
            trajectory.append(dict(branch=branch, new_update=step,
                **{k:v for k,v in values.items() if not isinstance(v, list)},
                large16_face_f1=large['face_micro_f1'], large16_strict=large['strict']))
            for m in e['meshes']:
                row = dict(branch=branch, new_update=step, uid=m['uid'], vertices=m['vertices'], strict=m['joint_perfect'])
                for task in ['edge', 'face']:
                    counts = m[task]
                    row.update({f'{task}_{k}':counts[k] for k in ['tp', 'fp', 'fn']})
                    row[f'{task}_f1'] = 2*counts['tp']/max(2*counts['tp']+counts['fp']+counts['fn'], 1)
                row.update(face_fn_missing=m['missing_gt_face_candidates'],
                    face_fn_present=m['face']['fn_present_but_negative'],
                    face_fp_inside_pool=m['face']['actual_fp_inside_training_pool'],
                    face_fp_outside_pool=m['face']['actual_fp_outside_training_pool'])
                per_mesh.append(row)
        for text in (root / 'updates.jsonl').read_text().splitlines():
            row = json.loads(text)
            step_edge.append(dict(branch=branch, new_update=row['new_update'],
                measured_state_before_update=row['state_before'],
                edge_strict_before=len(row['edge_perfect_uids_before']),
                edge_fp_before=sum(x['fp'] for x in row['meshes']),
                edge_fn_before=sum(x['fn'] for x in row['meshes']),
                grad_norm=row['total_grad_norm'], clip_coefficient=row['clip_coefficient']))
    s = evaluations['S_original']['100']
    h = evaluations['H_terminal_ffn']['100']
    delta = {scope:{k:h[scope][k]-v for k,v in s[scope].items() if isinstance(v,(int,float))}
        for scope in ['all50','large16']}
    result = dict(evaluations=evaluations, final_H_minus_S=delta,
        control_new_updates_this_turn=0, treatment_new_updates_this_turn=100,
        comparison_budget_per_branch=100, face_f1_target=.997,
        h_meets_face_f1_target=h['all50']['face_micro_f1']>=.997,
        h_strict_50=h['all50']['strict']==50,
        step_edge_semantics='Metrics BEFORE each optimizer update; not per-step actual Face evaluations')
    (OUT / 'COMPARISON.json').write_text(json.dumps(result,indent=2)+'\n')
    write_csv('trajectory.csv', trajectory)
    write_csv('per_mesh.csv', per_mesh)
    write_csv('step_edge.csv', step_edge)
    print(json.dumps(result['final_H_minus_S'],indent=2))


if __name__ == '__main__':
    main()
