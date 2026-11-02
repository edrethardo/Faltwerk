#!/usr/bin/env python3
"""stage 1 of the two-stage rig training: free per-frame splats with identity across frames.

Input: a matted multi-camera rig GT ($P8_GT) and a per-frame teacher model ($P43_TEACHER, one 3DGS PLY per frame, pattern
with {:05d}) fitted to the same views. Per-frame teacher models have no identity across frames (rows unrelated), so
they cannot be fed into a curve fit directly. Stage 1 therefore TRACKS them:

  frame 0   = the teacher's frame-0 PLY itself (opacity >= P43_MINOP), SH0 only.
  frame t   = warm start from the state of t-1 (same ids), P43_ITT tracking steps on all per-frame attributes against the
              targets of frame t; with P43_SWAP_TAU > 0 then the teacher swap (swap(): teacher splats of frame t that no
              splat of ours is within tau of are inserted, ours without a teacher splat within tau die -> fast parts are
              re-seeded from the teacher instead of being dragged) and P43_ITT2 more tracking steps; every P43_REFRESH
              frames (0 = never) additionally P43_ITR steps with densify/prune (gsplat
              DefaultStrategy, rig_track.fit_densify): clones/splits get new ids (birth t), pruned ids die at t.
  targets   = renders of the teacher PLY of frame t in every camera (consistent 3D teacher, rendered on the fly), blended
              with the real GT views: target = (1 - P43_MIX) * PLY render + P43_MIX * GT (RGB and alpha). Frames
              without a PLY (>= P43_PLYFRAMES) use the real GT only.
  loss      = rig_track.train_loss (L1 + D-SSIM, random background P8_RANDBG, alpha loss P8_ALPHA, optional hand term
              P8_HAND_W on the GT skin mask).

Output (same layout as rig_track, read by stage2_fit.py): <RUN>/tracks.npz (per-frame attributes over the global id
space, NaN when not alive; birth, death, shN [Ng,0,3]), <RUN>/s1.json (per frame: N, born, died, PSNR/fg-PSNR of the
test cameras against the real GT and against the PLY render, seconds).
  env: P43_RUN, P43_TEACHER, P43_FRAMES (64), P43_ITT (1500), P43_ITR (1000), P43_REFRESH (2), P43_MIX (0), P43_LR (1),
       P43_MINOP (0.005), P43_PLYFRAMES (50), P43_EXT (scene extent), P8_GT (real GT dir), P8_ALPHA (0.3),
       P8_RANDBG (1), P8_RANDBG_P (0.5)   GPU (torch, gsplat)
"""
import json
import math
import os
import sys
import time

os.environ.setdefault('E3_SH', '0')
os.environ.setdefault('P8_ALPHA', '0.3')
os.environ.setdefault('P8_RANDBG', '1')
os.environ.setdefault('P8_RANDBG_P', '0.5')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rig'))
import numpy as np                     # noqa: E402
import torch                           # noqa: E402
import rig_track as e3                     # noqa: E402
import rig_lib as L                     # noqa: E402

ROOT = os.environ.get('ROOT', 'work')
RUN = os.environ['P43_RUN']
T = int(os.environ.get('P43_FRAMES', '64'))
ITT = int(os.environ.get('P43_ITT', '1500'))
ITR = int(os.environ.get('P43_ITR', '1000'))
REFRESH = int(os.environ.get('P43_REFRESH', '2'))
MIX = float(os.environ.get('P43_MIX', '0'))
LRS = float(os.environ.get('P43_LR', '1'))
MINOP = float(os.environ.get('P43_MINOP', '0.005'))
PLYF = int(os.environ.get('P43_PLYFRAMES', '50'))
SWAP_TAU = float(os.environ.get('P43_SWAP_TAU', '0'))           # m; 0 = no teacher swap
ITT2 = int(os.environ.get('P43_ITT2', '300'))                    # tracking steps after the swap
EXT = float(os.environ.get('P43_EXT', '2.0'))                    # scene extent (the recipe sets it from the teacher)
PLY = os.environ['P43_TEACHER']                                   # teacher PLY pattern with {:05d}
TRACKED = e3.TRACKED
dev = e3.dev
assert e3.SH == 0 and e3.USE_ALPHA, 'stage 1 runs SH0 with the GT matte (E3_SH=0, P8_ALPHA/P8_RANDBG)'


def ply(t, minop=0.0):
    p = L.ply_params(L.read_ply(PLY.format(t)))
    keep = 1 / (1 + np.exp(-p['opacities'])) >= minop
    d = {k: torch.tensor(np.ascontiguousarray(v[keep]), device=dev).float() for k, v in p.items()}
    d['quats'] = torch.nn.functional.normalize(d['quats'], dim=-1)
    d['shN'] = torch.zeros((len(d['means']), 0, 3), device=dev)
    return d


def targets(t):
    """[C,H,W,4] for every camera: RGB premultiplied on black + alpha (PLY render / GT blend, see module doc)."""
    gt = e3.load_frame(t, rgba=True)
    if t >= PLYF:
        return gt, None, None
    q = ply(t)
    out = torch.empty_like(gt)
    with torch.no_grad():
        for c0 in range(0, len(e3.cams), 8):
            ids = list(range(c0, min(c0 + 8, len(e3.cams))))
            rgb, alpha, _ = e3.render(q, ids, 0)
            out[ids] = torch.cat([rgb, alpha], -1)
    teach = out.clone()
    if MIX > 0:
        out = (1 - MIX) * out + MIX * gt
    return out, teach, q


def track(p, tgt, iters):
    lrs = {'means': 4e-5 * EXT * LRS, 'quats': 5e-4 * LRS, 'scales': 1e-3 * LRS, 'opacities': 1e-2 * LRS, 'sh0': 1e-3 * LRS}
    for k in p:
        p[k].requires_grad_(k in TRACKED)
    opt = torch.optim.Adam([{'params': [p[k]], 'lr': lrs[k]} for k in TRACKED], eps=1e-15)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1 ** (1.0 / max(iters, 1)))
    for _ in range(iters):
        cam = list(np.random.choice(e3.train, 4, replace=False))
        rgb, alpha, _ = e3.render(p, cam, 0)
        loss = e3.train_loss(rgb, alpha, tgt[cam])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
    for k in p:
        p[k].requires_grad_(False)


def nn_dist(a, b, chunk=4096):
    """[len(a)] distance from every row of a to its nearest row of b (GPU, chunked cdist)."""
    out = torch.empty(len(a), device=dev)
    for i in range(0, len(a), chunk):
        out[i:i + chunk] = torch.cdist(a[i:i + chunk], b).min(1).values
    return out


def swap(p, ids, q, next_id, t, birth, death):
    """P43_SWAP_TAU > 0: where tracking did not follow the teacher (fast hands), take the teacher's splats.
    Teacher splats farther than tau from every splat of ours are inserted (new ids, born at t); splats of ours farther than
    tau from every teacher splat are removed (die at t). Elsewhere identity is kept."""
    keep_q = q['opacities'] > math.log(MINOP / (1 - MINOP))
    qm = q['means'][keep_q]
    dq = nn_dist(qm, p['means'])
    dp = nn_dist(p['means'], qm)
    ins = dq > SWAP_TAU
    rm = dp > SWAP_TAU
    keep = ~rm
    for i in ids[rm].tolist():
        death[int(i)] = t
    new = {k: torch.cat([p[k][keep], q[k][keep_q][ins]]) for k in p}
    n_ins = int(ins.sum())
    new_ids = torch.cat([ids[keep], torch.arange(next_id, next_id + n_ins, device=dev)])
    for i in range(next_id, next_id + n_ins):
        birth[i] = t
    return new, new_ids, next_id + n_ins, n_ins, int(rm.sum())


def evaluate(p, gt_test, teach):
    with torch.no_grad():
        rgb, _, _ = e3.render(p, e3.test, 0)
        g = gt_test[..., :3]
        r = {'psnr_gt': e3.psnr(rgb, g), 'fg_psnr_gt': e3.psnr_fg(rgb, g)}
        if teach is not None:
            r['psnr_teacher'] = e3.psnr(rgb, teach[e3.test][..., :3])
        return r


def main():
    os.makedirs(RUN, exist_ok=True)
    t0 = time.time()
    p = ply(0, MINOP)
    ids = torch.arange(len(p['means']), device=dev)
    next_id = len(ids)
    birth = {i: 0 for i in range(next_id)}
    death = {}
    frames = []
    log = {'cfg': {k: os.environ.get(k) for k in sorted(os.environ) if k.startswith(('P43_', 'P8_', 'E3_'))},
           'ext': EXT, 'frames': []}
    for t in range(T):
        ts = time.time()
        tgt, teach, q = targets(t)
        born = died = 0
        n_ins = n_rm = 0
        if t > 0:
            track(p, tgt, ITT)
            if SWAP_TAU > 0 and q is not None:
                p, ids, next_id, n_ins, n_rm = swap(p, ids, q, next_id, t, birth, death)
                track(p, tgt, ITT2)
            if REFRESH > 0 and t % REFRESH == 0 and ITR > 0:
                p, got = e3.fit_densify(p, ids, tgt, ITR, EXT, first=False)
                got = got.cpu().numpy()
                new_ids = np.empty_like(got)
                seen = set()
                for j, g in enumerate(got):
                    g = int(g)
                    if g in seen:
                        new_ids[j] = next_id
                        birth[next_id] = t
                        next_id += 1
                        born += 1
                    else:
                        new_ids[j] = g
                        seen.add(g)
                dl = [int(i) for i in ids.tolist() if int(i) not in seen]
                for i in dl:
                    death[i] = t
                died = len(dl)
                ids = torch.tensor(new_ids, device=dev)
        frames.append((ids.cpu().numpy(), {k: p[k].cpu().numpy().astype(np.float32) for k in TRACKED}))
        r = evaluate(p, tgt[e3.test] if teach is None else e3.load_frame(t, rgba=True)[e3.test], teach)
        r.update(t=t, N=len(ids), born=born + n_ins, died=died + n_rm, swap_in=n_ins, swap_out=n_rm,
                 secs=round(time.time() - ts, 1))
        log['frames'].append(r)
        print(f'frame {t}: N={len(ids)} +{born} -{died} swap +{n_ins} -{n_rm} ' + ' '.join(f'{k} {v:.2f}' for k, v in r.items()
                                                                                          if k.startswith(('psnr', 'fg'))),
              f'({r["secs"]} s)', flush=True)
        del tgt, teach, q
        torch.cuda.empty_cache()
    Ng = next_id
    out = {k: np.full((T, Ng) + frames[0][1][k].shape[1:], np.nan, np.float32) for k in TRACKED}
    for t, (fid, attrs) in enumerate(frames):
        for k in TRACKED:
            out[k][t, fid] = attrs[k]
    b = np.array([birth[i] for i in range(Ng)], np.int32)
    d = np.array([death.get(i, T) for i in range(Ng)], np.int32)
    np.savez(os.path.join(RUN, 'tracks.npz'), birth=b, death=d, shN=np.zeros((Ng, 0, 3), np.float16), **out)
    log.update(N_total=Ng, secs_total=round(time.time() - t0, 1),
               mean_fg_psnr_gt=float(np.mean([f['fg_psnr_gt'] for f in log['frames']])))
    json.dump(log, open(os.path.join(RUN, 's1.json'), 'w'), indent=1)
    print('S1 OK', RUN, f'ids {Ng}, mean fg-PSNR(GT) {log["mean_fg_psnr_gt"]:.2f} dB, {log["secs_total"]:.0f} s', flush=True)


if __name__ == '__main__':
    main()
