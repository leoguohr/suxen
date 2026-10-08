"""Recovered scoring hypothesis; complete edge graph then predicted 3-cycles.

Spacetime split/zero sign rule are validated recovery conventions. Configuration
identifies equation 5; serialization alone does not encode score formulas.
Canonical face sets below are evaluation only, not mesh repair.
"""
import itertools
import torch
from torch.nn import functional as F


def gram_area(u, v):
    return u.square().sum(-1) * v.square().sum(-1) - (u * v).sum(-1).square()


def edge_scores(embedding, pairs, method='spacetime', threshold=1., cosine_scale=10.):
    a, b = embedding[pairs[:, 0]], embedding[pairs[:, 1]]
    if method == 'cosine':
        return cosine_scale * (F.cosine_similarity(a, b, dim=-1) - threshold)
    delta = a - b
    if method == 'euclidean':
        return delta.square().sum(-1) - threshold
    if method != 'spacetime':
        raise ValueError(method)
    positive, negative = delta.chunk(2, dim=-1)
    return positive.square().sum(-1) - negative.square().sum(-1)


def face_scores(embedding, triangles, method='spacetime', threshold=1., cosine_scale=10.):
    a, b, c = (embedding[triangles[:, i]] for i in range(3))
    if method == 'cosine':
        similarity = (F.cosine_similarity(a, b, dim=-1) + F.cosine_similarity(b, c, dim=-1)
                      + F.cosine_similarity(c, a, dim=-1)) / 3
        return cosine_scale * (similarity - threshold)
    u, v = a - c, b - c
    if method == 'euclidean':
        return gram_area(u, v) - threshold
    if method != 'spacetime':
        raise ValueError(method)
    up, un = u.chunk(2, -1)
    vp, vn = v.chunk(2, -1)
    return gram_area(up, vp) - gram_area(un, vn)


def decode_mesh(edge_embedding, face_embedding, method='spacetime', edge_threshold=1., face_threshold=1.):
    count = edge_embedding.shape[0]
    pairs = torch.triu_indices(count, count, offset=1, device=edge_embedding.device).T
    logits = edge_scores(edge_embedding, pairs, method, edge_threshold)
    edges = pairs[logits > 0]
    neighbors = [set() for _ in range(count)]
    for a, b in edges.tolist():
        neighbors[a].add(b)
        neighbors[b].add(a)
    triples = [(a, b, c) for a in range(count) for b in sorted(neighbors[a]) if a < b
               for c in sorted(neighbors[a] & neighbors[b]) if b < c]
    candidates = torch.tensor(triples, dtype=torch.long, device=edges.device).reshape(-1, 3)
    selected = []
    for chunk in candidates.split(16384):
        selected.append(chunk[face_scores(face_embedding, chunk, method, face_threshold) > 0])
    faces = torch.cat(selected) if selected else candidates
    return {'edges':edges, 'faces':faces, 'candidates':candidates}


def canonical_set(tensor):
    return {tuple(sorted(row)) for row in tensor.tolist()}


def metrics(prediction, true_faces):
    truth_faces = canonical_set(true_faces)
    truth_edges = {edge for face in truth_faces for edge in itertools.combinations(face, 2)}
    answer = {}
    for kind, truth in [('edge', truth_edges), ('face', truth_faces)]:
        predicted = canonical_set(prediction[kind + 's'])
        tp, fp, fn = len(predicted & truth), len(predicted - truth), len(truth - predicted)
        answer[kind] = {'tp':tp, 'fp':fp, 'fn':fn, 'f1':2 * tp / max(2 * tp + fp + fn, 1)}
    answer['candidate_count'] = len(prediction['candidates'])
    answer['gt_faces_missing_candidates'] = len(truth_faces - canonical_set(prediction['candidates']))
    return answer
