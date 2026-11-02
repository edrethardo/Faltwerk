"""P16 trainer helpers used by train/e7_curves_train.py:
adaptive K per splat (more control points only for fast-moving splats) and the section-boundary smoothness term.
Everything here is off unless the trainer's switches (E7_KADAPT / E7_SMOOTH) are set.

Adaptive K: the control-point tensors keep K = E7_K (e.g. 9) columns. All splats start a section with K_lo = E7_KLO (e.g. 5)
columns in use (the other columns are copies of the last used one, so the trainer's densify/prune code, which looks at the
whole tensor, behaves like the K_lo run). At iteration E7_KADAPT_AT x ITS (after the densify phase) the splats with the
highest motion score (top E7_KADAPT_FRAC) get all K columns: their curve is refitted to the K_lo curve (exact for the hat
basis when (K-1) % (K_lo-1) == 0, least squares with fixed end points otherwise) and trained on. S['kn'] carries the K of
every splat; curves.npz stores it as g<i>_kn, codec.prep reads it (_reduce_adaptive).
"""
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from codec.falt import bspline_weights  # noqa: E402


def basis_matrix(K, frames, span, bspline):
    """[F, K] weights of the K control points at integer frames 0..frames-1 of a section whose time axis spans `span` frames."""
    if K == 1:
        return np.ones((frames, 1))
    if bspline:
        return np.array([bspline_weights(K, min(max(f / span, 0.0), 1.0)) for f in range(frames)])
    B = np.zeros((frames, K))
    for f in range(frames):
        tau = min(max(f * (K - 1) / span, 0.0), K - 1)
        j = min(int(math.floor(tau)), K - 2)
        B[f, j], B[f, j + 1] = 1 - (tau - j), tau - j
    return B


@torch.no_grad()
def refit_columns(C_lo, K_hi, frames, span, bspline):
    """C_lo [n,K_lo,d] torch -> [n,K_hi,d]: control points of a K_hi curve that reproduces the K_lo curve at the section's
    frames (end points fixed, least squares for the interior)."""
    squeeze = C_lo.dim() == 2                                  # opacities [n, K]
    if squeeze:
        C_lo = C_lo[..., None]
    n, K_lo, d = C_lo.shape
    B_lo = torch.tensor(basis_matrix(K_lo, frames, span, bspline), dtype=torch.float64, device=C_lo.device)
    B_hi = torch.tensor(basis_matrix(K_hi, frames, span, bspline), dtype=torch.float64, device=C_lo.device)
    target = torch.einsum('fk,nkd->nfd', B_lo, C_lo.double())
    c0, cl = C_lo[:, 0].double(), C_lo[:, -1].double()
    rhs = target - B_hi[None, :, 0, None] * c0[:, None] - B_hi[None, :, -1, None] * cl[:, None]
    inner = torch.einsum('kf,nfd->nkd', torch.linalg.pinv(B_hi[:, 1:-1]), rhs)
    out = torch.cat([c0[:, None], inner, cl[:, None]], 1).to(C_lo.dtype)
    return out[..., 0] if squeeze else out


@torch.no_grad()
def motion_score(P_means, K_lo, mode='path'):
    """per splat: 'path' = length of the control polygon of the position curve in the section (scene units; fast-moving
    splats), 'curv' = sum of the second differences of the control points (how far the K_lo curve bends)."""
    v = P_means[:, :K_lo].float()
    if mode == 'curv' and K_lo > 2:
        return (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]).norm(dim=-1).sum(-1)
    return (v[:, 1:] - v[:, :-1]).norm(dim=-1).sum(-1)


@torch.no_grad()
def select_hi(P, S, K_lo, frac, mode, min_alpha=0.05):
    """boolean [N]: top `frac` of the visible splats (opacity > min_alpha at some used knot) by motion score."""
    n = len(S['birth'])
    vis = torch.sigmoid(P['opacities'][:, :K_lo]).amax(-1) > min_alpha
    sc = motion_score(P['means'], K_lo, mode)
    sc = torch.where(vis, sc, torch.full_like(sc, -1.0))
    k = int(frac * int(vis.sum()))
    hi = torch.zeros(n, dtype=torch.bool, device=sc.device)
    if k > 0:
        thr = torch.topk(sc, k).values[-1]
        hi = sc >= thr
        hi &= vis
    return hi


@torch.no_grad()
def sync_pad(P, tracked, kn):
    """columns >= kn of a splat := its last used column (keeps densify/prune and the saved curves consistent)."""
    K = P['means'].shape[1]
    lo = kn < K
    if not bool(lo.any()):
        return
    cols = torch.arange(K, device=kn.device)[None]
    src = (kn[:, None] - 1).clamp_min(0)
    pad = cols >= kn[:, None]                                 # [N,K]
    for k in tracked:
        v = P[k].data
        last = v.gather(1, src.view(-1, 1, *([1] * (v.dim() - 2))).expand(-1, K, *v.shape[2:]))
        P[k].data = torch.where(pad.view(*pad.shape, *([1] * (v.dim() - 2))), last, v)


def curve_adaptive(curve_at, P, idx, t, g0, Lg, kn):
    """curve_at for splats with their own K (kn [N] torch long): evaluate per distinct K, restore the order."""
    kv = kn[idx]
    ks = torch.unique(kv).tolist()
    if len(ks) == 1:
        return curve_at(P, idx, t, g0, Lg, int(ks[0]))
    parts, pos = [], []
    for K in ks:
        m = torch.nonzero(kv == K).squeeze(1)
        parts.append(curve_at(P, idx[m], t, g0, Lg, int(K)))
        pos.append(m)
    inv = torch.argsort(torch.cat(pos))
    return {k: torch.cat([p[k] for p in parts])[inv] for k in parts[0]}


def deriv_factor(K, bspline):
    """d(curve)/d(frame) at the section start/end = deriv_factor / span * (cp1 - cp0) resp. (cp_{K-1} - cp_{K-2}); K tensor or int."""
    if bspline:
        p = torch.clamp(K - 1, max=3) if torch.is_tensor(K) else min(3, K - 1)
        return p * (K - p)
    return K - 1


@torch.no_grad()
def end_state(Pk, kn, span, bspline):
    """value and velocity (per frame) of the curve at the last frame of its section; Pk [N,K,d], kn [N] long."""
    N = Pk.shape[0]
    ar = torch.arange(N, device=Pk.device)
    last = Pk[ar, kn - 1]
    prev = Pk[ar, (kn - 2).clamp_min(0)]
    f = deriv_factor(kn.to(Pk.dtype), bspline) / span
    vel = torch.where((kn > 1)[:, None], f[:, None] * (last - prev), torch.zeros_like(last))
    return last, vel


def start_vel(Pk, kn, span, bspline):
    """velocity (per frame) of the new section's curve at its first frame (differentiable)."""
    f = deriv_factor(kn.to(Pk.dtype), bspline) / span
    return torch.where((kn > 1)[:, None], f[:, None] * (Pk[:, 1] - Pk[:, 0]), torch.zeros_like(Pk[:, 0]))


def greville(K, bspline):
    """abscissae (0..1) of the K control points: the points where control polygon = curve for a linear motion. Hat: uniform;
    clamped uniform cubic B-spline (codec.falt.bspline_weights knot vector): g_j = mean of knots j+1 .. j+p."""
    if K < 2:
        return np.zeros(K)
    if not bspline:
        return np.linspace(0.0, 1.0, K)
    p = min(3, K - 1)
    n_int = K - p - 1
    kn = [0.0] * (p + 1) + [(i + 1) / (n_int + 1) for i in range(n_int)] + [1.0] * (p + 1)
    return np.array([sum(kn[j + 1:j + p + 1]) / p for j in range(K)])


def curv_d2(v, K, bspline):
    """second-difference-like curvature of control points v [N,K,...] that vanishes for motion linear in time: residual of every
    interior control point against the linear interpolation of its neighbours at the Greville abscissae, x2 so that it equals
    the plain second difference v[j+1] - 2 v[j] + v[j-1] (up to sign) for uniform spacing (hat basis)."""
    g = greville(K, bspline)
    a = torch.tensor((g[2:] - g[1:-1]) / (g[2:] - g[:-2]), dtype=v.dtype, device=v.device)       # weight of the left neighbour
    b = torch.tensor((g[1:-1] - g[:-2]) / (g[2:] - g[:-2]), dtype=v.dtype, device=v.device)       # weight of the right neighbour
    sh = (1, K - 2) + (1,) * (v.dim() - 2)
    return 2 * (v[:, 1:-1] - a.view(sh) * v[:, :-2] - b.view(sh) * v[:, 2:])
