"""helpers for train/e7_curves_train.py : GT-flow guidance and motion-weighted
frame sampling. Imported only when the switches are on (defaults unchanged).

E7_FLOW=w (+ E7_FLOW_DIR=<dir of train/flow/gt_flow.py>, E7_FLOW_EDGE=e (0.75)): every step, for the first sampled
frame t and its cameras, the pair (s, s+1) inside the section (s = t, or t-1 at the section end) is taken; for every splat
alive at both frames the 2D motion of its projected centre d = pi_c(mu(s+1)) - pi_c(mu(s)) (from the curves, so the
gradient reaches the knots) is alpha-composited like a colour with the geometry of frame s DETACHED (visibility from the
renderer, nothing moves to improve the flow image by itself): F_r = sum_i w_i d_i. Loss
    w x sum_p conf(p) x ((1 - e) + e x edge(p)) x |F_r(p) - alpha(p) F_gt(p)|_1 / sum_p conf(p)
with conf = GT forward/backward consistency (x GT matte for matted scenes), edge = clamp(|grad luminance of the GT at s| /
0.1, 0, 1) at the flow scale (registration errors show up at edges). Units: pixels at the flow scale (e.g. 0.5 x GT).

E7_MSAMP=a, E7_TAIL_W=w, E7_TAIL_N=n: frame sampling weight per section (1 - a) + a x E_t / mean(E), E_t = GT motion
energy (mean |G_t - G_t-1| over every 4th train camera, 1/4 resolution, mean of both neighbour differences), the last n
frames of the section x w (the section ends f24-31 / f56-63 carry the flicker peaks in P22).
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fn


class FlowSection:
    """GT flow of the pairs (t, t+1), t in [g0, g1 - 1), all cameras, pinned CPU int16/uint8 (e.g. 31 x 81 x 512^2)."""

    def __init__(self, flow_dir, g0, g1, ncams, dev):
        m = json.load(open(os.path.join(flow_dir, 'meta.json')))
        self.w, self.h = m['size_wh']
        self.scale, self.q, self.g0, self.dev = float(m['scale']), float(m['flow_q']), g0, dev
        n = max(g1 - 1 - g0, 0)
        self.flow = torch.empty((n, ncams, self.h, self.w, 2), dtype=torch.int16).pin_memory()
        self.conf = torch.empty((n, ncams, self.h, self.w), dtype=torch.uint8).pin_memory()
        for j in range(n):
            z = np.load(os.path.join(flow_dir, f'f{g0 + j:03d}.npz'))
            self.flow[j] = torch.from_numpy(z['flow'][:ncams])
            self.conf[j] = torch.from_numpy(z['conf'][:ncams])

    def get(self, s, cams):
        j = s - self.g0
        c = torch.as_tensor(cams, dtype=torch.long)
        F = self.flow[j].index_select(0, c).to(self.dev, non_blocking=True).float() / self.q
        cf = self.conf[j].index_select(0, c).to(self.dev, non_blocking=True).float() / 255.0
        return F, cf


def project(X, VMc, Kc):
    """X [N,3] world -> [C,N,2] pixels (OpenCV camera, gsplat convention)."""
    Xc = torch.einsum('cij,nj->cni', VMc[:, :3, :3], X) + VMc[:, None, :3, 3]
    z = Xc[..., 2].clamp_min(1e-4)
    u = Kc[:, None, 0, 0] * Xc[..., 0] / z + Kc[:, None, 0, 2]
    v = Kc[:, None, 1, 1] * Xc[..., 1] / z + Kc[:, None, 1, 2]
    return torch.stack([u, v], -1)


def edge_map(gt, h, w):
    """GT [C,H,W,3] float in [0,1] -> edge weight [C,h,w] in [0,1] at the flow scale."""
    g = gt.permute(0, 3, 1, 2)
    if g.shape[-2:] != (h, w):
        g = Fn.interpolate(g, size=(h, w), mode='area')
    L = 0.299 * g[:, 0] + 0.587 * g[:, 1] + 0.114 * g[:, 2]
    gx = torch.zeros_like(L); gy = torch.zeros_like(L)
    gx[:, :, 1:-1] = (L[:, :, 2:] - L[:, :, :-2]) / 2
    gy[:, 1:-1] = (L[:, 2:] - L[:, :-2]) / 2
    return ((gx ** 2 + gy ** 2).sqrt() / 0.1).clamp(0, 1)


def flow_term(q0, m1, VMc, Kf, w, h, F, conf, gt_s, edge_w):
    """q0: params at frame s of the splats alive at s and s+1, m1: their means at s+1. Returns (loss, stats)."""
    from gsplat import rasterization
    d = project(m1, VMc, Kf) - project(q0['means'], VMc, Kf)                  # [C,N,2] px at the flow scale
    col = torch.cat([d, torch.zeros_like(d[..., :1])], -1)
    img, alp, _ = rasterization(q0['means'].detach(), q0['quats'].detach(), torch.exp(q0['scales']).detach(),
                                torch.sigmoid(q0['opacities']).detach(), col, VMc, Kf, w, h, sh_degree=None,
                                packed=False)
    a = alp[..., 0].detach()
    res = (img[..., :2] - a[..., None] * F).abs().sum(-1)                       # [C,h,w]
    wgt = conf
    if edge_w > 0:
        wgt = conf * ((1 - edge_w) + edge_w * edge_map(gt_s, h, w))
    den = conf.sum().clamp_min(1.0)
    loss = (wgt * res).sum() / den
    with torch.no_grad():
        sel = conf > 0.5
        st = {'res_px': float(res[sel].mean()) if bool(sel.any()) else 0.0,
              'gt_px': float(F.norm(dim=-1)[sel].mean()) if bool(sel.any()) else 0.0,
              'r_px': float((img[..., :2] / a[..., None].clamp_min(1e-3)).norm(dim=-1)[sel & (a > 0.5)].mean())
              if bool((sel & (a > 0.5)).any()) else 0.0}
    return loss, st


@torch.no_grad()
def motion_weights(gts, train_cams, a, tail_w, tail_n):
    """[Lg] frame sampling factors (see module doc) from the section GT gts [Lg, C, H, W, 3] uint8 (GPU)."""
    Lg = gts.shape[0]
    fw = np.ones(Lg)
    E = None
    if a > 0 and Lg >= 2:
        cams = list(train_cams)[::4]
        m = torch.stack([(gts[t + 1][cams, ::4, ::4].float() - gts[t][cams, ::4, ::4].float()).abs().mean()
                         for t in range(Lg - 1)]).cpu().numpy()
        E = np.zeros(Lg); n = np.zeros(Lg)
        E[1:] += m; n[1:] += 1; E[:-1] += m; n[:-1] += 1
        E = E / n
        fw = (1 - a) + a * E / max(E.mean(), 1e-9)
    if tail_w != 1 and tail_n > 0:
        fw[-tail_n:] *= tail_w
    return fw, E
