"""Independent numpy re-derivation of the CAD50 objective. No torch needed.

Re-implements edge_logits / face_logits / hard4 straight from their algebraic
definitions and compares against the formulas used in code/objective.py.
Run this FIRST if you want to convince yourself the reconstruction is right.

    python verify_math.py

Requires only numpy.
"""
import numpy as np

EDGE_SCALE = 0.9306077080970389
FACE_SCALE = 0.39804385828730726
FACE_FACTOR = 0.25


# ---------------- transcription of code/objective.py ------------------------
def edge_logits(emb, pairs):
    d = emb[pairs[:, 0]] - emb[pairs[:, 1]]
    s, t = np.split(d, 2, axis=-1)
    return (np.square(s).sum(-1) - np.square(t).sum(-1)) * EDGE_SCALE


def face_logits(emb, triples):
    a, b, c = emb[triples[:, 0]], emb[triples[:, 1]], emb[triples[:, 2]]
    areas = []
    for x, y, z in zip(np.split(a, 2, -1), np.split(b, 2, -1), np.split(c, 2, -1)):
        u, v = y - x, z - x
        areas.append(np.clip(np.square(u).sum(-1) * np.square(v).sum(-1)
                             - np.square((u * v).sum(-1)), 0, None))
    return FACE_SCALE * (FACE_FACTOR * (areas[0] - areas[1]))


# ---------------- independent re-derivation ---------------------------------
def area(P, Q, R):
    """True Euclidean triangle area, valid in any dimension."""
    u, v = Q - P, R - P
    gram = np.square(u).sum() * np.square(v).sum() - np.square((u * v).sum())
    return 0.5 * np.sqrt(np.clip(gram, 0, None))


def hard4(logits, labels):
    """4-group BCE, mean of per-group means."""
    pos, truth = logits > 0, labels == 1
    masks = [truth & pos, ~truth & ~pos, ~truth & pos, truth & ~pos]
    bce = np.maximum(logits, 0) - logits * labels + np.log1p(np.exp(-np.abs(logits)))
    num = np.array([bce[m].sum() for m in masks])
    cnt = np.array([m.sum() for m in masks])
    return num, cnt, (num / np.maximum(cnt, 1)).sum() / 4


# ---------------- verification ----------------------------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    emb = rng.standard_normal((12, 32)).astype(np.float32)
    emb = emb - emb.mean(0, keepdims=True)

    tri = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]], dtype=np.int64)
    lab = np.array([1, 0, 1, 0])
    half = 16

    print("=" * 70)
    print("1) face_logits  ==  FACE_SCALE * (Area_space^2 - Area_time^2)")
    fl = face_logits(emb, tri)
    derived = np.array([
        FACE_SCALE * (area(*[emb[i][:half] for i in t]) ** 2
                      - area(*[emb[i][half:] for i in t]) ** 2)
        for t in tri], dtype=np.float32)
    print("   source  :", np.round(fl, 6).tolist())
    print("   derived :", np.round(derived, 6).tolist())
    print("   MATCH   :", np.allclose(fl, derived, atol=1e-4))

    print()
    print("2) edge_logits  ==  EDGE_SCALE * (||delta||^2_space - ||delta||^2_time)")
    pairs = np.array([[0, 1], [0, 2], [3, 7], [5, 9]], dtype=np.int64)
    el = edge_logits(emb, pairs)
    d = emb[pairs[:, 0]] - emb[pairs[:, 1]]
    d2 = EDGE_SCALE * (np.square(d[:, :half]).sum(-1) - np.square(d[:, half:]).sum(-1))
    print("   source  :", np.round(el, 6).tolist())
    print("   derived :", np.round(d2, 6).tolist())
    print("   MATCH   :", np.allclose(el, d2, atol=1e-5))

    print()
    print("3) Lagrange identity: |u|^2|v|^2 - (u.v)^2 == |u x v|^2  (n=32)")
    a, b, c = emb[tri[0, 0]], emb[tri[0, 1]], emb[tri[0, 2]]
    u, v = b - a, c - a
    lag = np.square(u).sum() * np.square(v).sum() - np.square((u * v).sum())
    n = len(u)
    cross2 = sum((u[i] * v[j] - u[j] * v[i]) ** 2 for i in range(n) for j in range(i + 1, n))
    print("   MATCH   :", np.allclose(lag, cross2), f"({n * (n - 1) // 2} coordinate pairs)")

    print()
    print("4) Hard4 group structure")
    num, cnt, loss = hard4(fl, lab)
    print("   groups [tp,tn,fp,fn] counts :", cnt.tolist())
    print("   groups [tp,tn,fp,fn] bce_sum:", np.round(num, 4).tolist())
    print("   loss (mean of 4 group means):", round(float(loss), 6))

    print()
    print("5) Why equal-group weighting tilts the boundary")
    fp_total, tn_total = 47, 168 * 50
    w_fp, w_tn = (1 / 4) / fp_total, (1 / 4) / tn_total
    print(f"   fp={fp_total}, tn={tn_total}")
    print(f"   per-sample weight fp = {w_fp:.3e}")
    print(f"   per-sample weight tn = {w_tn:.3e}")
    print(f"   amplification        = x{w_fp / w_tn:.1f}")
    print("   -> matches observed drift: face_fp 47->41 down, face_fn 6->17 up")
    print("=" * 70)
