"""per-splat contribution of a trained run and a splat budget (keep.npz for codec.measure --keep).
Contribution of splat i at frame t = sum over training cameras (every CAMSTEP-th) and pixels of its blending weight
alpha_i * T_i (exact, from gsplat: gradient of the rendered image of constant colour 1 w.r.t. the per-splat colour).
Score = mean of that over the sampled frames where the splat is alive (short-lived fast splats - hands, hair - are not
penalised for their short life). Budget: the smallest score threshold s.t. every .csgs section holds <= B rows
(rows = ids alive in any frame of the section; the player sorts nextPow2(rows), so B = 2^k halves the sort).
  python contrib.py <run_k_dir> <out_dir> [--budgets 131072,98304,65536] [--fstep 2] [--camstep 3]
  writes <out_dir>/contrib.npz (score, alive_frames) and keep_<B>.npz (keep bool per global id) + contrib.json
  GPU, train venv; env P8_GT/P8_FRAMES like train/rig/rig_track.py (only the camera meta is read)."""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rig'))
import rig_track as e3  # noqa: E402
from gsplat import rasterization  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('kdir')
ap.add_argument('out')
ap.add_argument('--budgets', default='131072,98304,65536')
ap.add_argument('--fstep', type=int, default=2)
ap.add_argument('--camstep', type=int, default=3)
ap.add_argument('--batch', type=int, default=8)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
z = np.load(os.path.join(a.kdir, 'tracks.npz'))
birth, death = z['birth'], z['death']
TR = {k: z[k] for k in ('means', 'quats', 'scales', 'opacities')}   # npz members re-read on every access: load once
T, Ng = TR['means'].shape[:2]
cv = np.load(os.path.join(a.kdir, 'curves.npz'))
L = int(cv['L'])
cams = e3.train[::a.camstep]
dev = e3.dev
acc = np.zeros(Ng, np.float64)
cnt = np.zeros(Ng, np.int32)
frames = list(range(0, T, a.fstep))
for t in frames:
    alive = np.nonzero(~np.isnan(TR['means'][t, :, 0]))[0]
    m = torch.tensor(TR['means'][t, alive], device=dev)
    q = torch.nn.functional.normalize(torch.tensor(TR['quats'][t, alive].astype(np.float32), device=dev), dim=-1)
    s = torch.exp(torch.tensor(TR['scales'][t, alive].astype(np.float32), device=dev))
    o = torch.sigmoid(torch.tensor(TR['opacities'][t, alive].astype(np.float32), device=dev))
    w = torch.zeros(len(alive), device=dev)
    for i in range(0, len(cams), a.batch):
        cb = cams[i:i + a.batch]
        col = torch.ones((len(alive), 3), device=dev, requires_grad=True)
        img, _, _ = rasterization(m, q, s, o, col, e3.VM[cb], e3.K[cb], e3.W, e3.H, packed=False,
                                  near_plane=0.01)
        img.sum().backward()
        w += col.grad.sum(-1).detach() / 3.0
    acc[alive] += w.cpu().numpy() / len(cams)
    cnt[alive] += 1
    print(f'frame {t}: alive {len(alive)}, mean weight {float(w.mean()) / len(cams):.2f} px/view ({time.time() - t0:.0f} s)', flush=True)
score = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0)
np.savez(os.path.join(a.out, 'contrib.npz'), score=score.astype(np.float32), frames_seen=cnt, birth=birth, death=death)
# rows per section: alive in any frame of [g0, g1)
secs = [(g0, min(T, g0 + L)) for g0 in range(0, T, L)]
in_sec = [(birth < g1) & (death > g0) for g0, g1 in secs]
res = {'kdir': a.kdir, 'T': int(T), 'N_ids': int(Ng), 'cams': [int(c) for c in cams], 'frames': frames,
       'rows_full': [int(m_.sum()) for m_ in in_sec], 'score_quantiles': {str(p): float(np.quantile(score, p))
                                                                        for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
       'budgets': {}}
order = np.sort(score)
for B in [int(x) for x in a.budgets.split(',')]:
    lo, hi = 0, len(order) - 1                      # smallest threshold index with max rows <= B
    while lo < hi:
        mid = (lo + hi) // 2
        th = order[mid]
        if max(int((m_ & (score >= th)).sum()) for m_ in in_sec) <= B:
            hi = mid
        else:
            lo = mid + 1
    th = float(order[lo])
    keep = score >= th
    rows = [int((m_ & keep).sum()) for m_ in in_sec]
    frac_w = float(score[keep].sum() / max(score.sum(), 1e-12))
    np.savez(os.path.join(a.out, f'keep_{B}.npz'), keep=keep, score=score.astype(np.float32), threshold=th)
    res['budgets'][str(B)] = {'threshold_px_per_view': th, 'kept_ids': int(keep.sum()), 'rows': rows,
                              'score_mass_kept': frac_w}
    print(f'budget {B}: threshold {th:.3f} px/view, kept {int(keep.sum())} of {Ng}, rows {rows}, score mass kept {frac_w:.4f}',
          flush=True)
res['secs'] = round(time.time() - t0, 1)
json.dump(res, open(os.path.join(a.out, 'contrib.json'), 'w'), indent=1)
print('CONTRIB OK', a.out)
